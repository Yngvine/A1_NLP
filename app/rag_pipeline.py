import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from config import DEFAULT_TOP_K, MAX_AGENT_ITERATIONS
from llm_client import LLMClient
from retriever import CaptionRetriever


class AgentState(TypedDict):
    query: str
    retrieved_chunks: Annotated[list, operator.add]
    search_queries: Annotated[list, operator.add]
    reasoning: Annotated[list, operator.add]
    answer: str
    iteration: int
    action: str


# Lazy singletons
_retriever = None
_llm = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = CaptionRetriever()
    return _retriever


def get_llm():
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


def plan_action(state: AgentState) -> AgentState:
    """Decide whether to search for more context or generate an answer."""
    if state["iteration"] == 0:
        return {
            **state,
            "action": "search",
            "reasoning": [
                "First iteration: performing initial search for relevant passages."
            ],
        }

    if state["iteration"] >= MAX_AGENT_ITERATIONS:
        return {
            **state,
            "action": "generate",
            "reasoning": [
                f"Reached max iterations ({MAX_AGENT_ITERATIONS}). Generating answer with available context."
            ],
        }

    # Ask the LLM if we have enough context
    context_summary = "\n".join(
        [
            f"- {chunk['document'][:200]}..."
            for chunk in state["retrieved_chunks"][:5]
        ]
    )

    prompt = f"""You are a search assistant. Given a question and the passages retrieved so far, decide if you have enough information to answer, or if you need to search for more.

Question: {state["query"]}

Retrieved passages so far:
{context_summary}

Do you have enough context to answer the question? Reply with exactly one word: "yes" or "no"."""

    try:
        decision = get_llm().generate(prompt, max_tokens=10, temperature=0.0)
        needs_more = "no" in decision.lower()
    except Exception:
        needs_more = False

    if needs_more:
        return {
            **state,
            "action": "search",
            "reasoning": [
                "LLM determined more context is needed. Performing additional search."
            ],
        }
    else:
        return {
            **state,
            "action": "generate",
            "reasoning": [
                "LLM determined sufficient context is available. Generating answer."
            ],
        }


def search(state: AgentState) -> AgentState:
    """Search ChromaDB for relevant captions."""
    search_query = state["query"]

    # On subsequent iterations, try to refine the search query
    if state["iteration"] > 0:
        try:
            prompt = f"""Given the original question and the passages already found, generate a short refined search query to find additional relevant information.

Original question: {state["query"]}
Already found {len(state["retrieved_chunks"])} passages.

Refined search query (one sentence only):"""
            refined = get_llm().generate(prompt, max_tokens=50, temperature=0.3)
            if refined.strip():
                search_query = refined.strip()
        except Exception:
            pass

    chunks = get_retriever().search(search_query, top_k=DEFAULT_TOP_K)

    # Deduplicate
    existing_ids = {c["id"] for c in state["retrieved_chunks"]}
    new_chunks = [c for c in chunks if c["id"] not in existing_ids]

    return {
        **state,
        "retrieved_chunks": new_chunks,
        "search_queries": [search_query],
        "iteration": state["iteration"] + 1,
        "reasoning": [
            f"Searched for: '{search_query}'. Retrieved {len(new_chunks)} new chunks "
            f"({len(chunks) - len(new_chunks)} duplicates filtered)."
        ],
    }


def generate(state: AgentState) -> AgentState:
    """Generate final answer using retrieved context."""
    context_parts = []
    for i, chunk in enumerate(state["retrieved_chunks"][:10]):
        meta = chunk["metadata"]
        context_parts.append(
            f"[Passage {i + 1}] (Image: {meta.get('image', 'N/A')}, "
            f"Tags: {meta.get('tags', 'N/A')})\n{chunk['document']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""Below are relevant passages about remote sensing images. Use ONLY the information in these passages to answer the question. If the passages do not contain the answer, say "I cannot answer this based on the available documents."

### Passages:
{context}

### Question:
{state["query"]}

### Answer:"""

    try:
        answer = get_llm().generate(prompt, max_tokens=512, temperature=0.1)
    except Exception:
        if state["retrieved_chunks"]:
            top_chunk = state["retrieved_chunks"][0]
            answer = (
                "The language model is still warming up. "
                "Based on the top retrieved passage, this is the most relevant evidence: "
                f"{top_chunk['document'][:350]}"
            )
        else:
            answer = (
                "The language model is still warming up and no supporting passages were found yet. "
                "Please retry in 30-60 seconds."
            )

    return {**state, "answer": answer}


def route_action(state: AgentState) -> str:
    if state["action"] == "search":
        return "search"
    return "generate"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("plan_action", plan_action)
    graph.add_node("search", search)
    graph.add_node("generate", generate)

    graph.set_entry_point("plan_action")

    graph.add_conditional_edges(
        "plan_action",
        route_action,
        {"search": "search", "generate": "generate"},
    )

    graph.add_edge("search", "plan_action")
    graph.add_edge("generate", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_rag_agent(question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    initial_state: AgentState = {
        "query": question,
        "retrieved_chunks": [],
        "search_queries": [],
        "reasoning": [],
        "answer": "",
        "iteration": 0,
        "action": "",
    }

    final_state = get_graph().invoke(initial_state)

    source_chunks = [
        {
            "document_id": chunk["id"],
            "text": chunk["document"][:500],
            "image": chunk["metadata"].get("image", ""),
            "tags": chunk["metadata"].get("tags", ""),
            "relevance_score": round(chunk.get("relevance_score", 0.0), 4),
        }
        for chunk in final_state["retrieved_chunks"][:top_k]
    ]

    return {
        "answer": final_state["answer"],
        "source_chunks": source_chunks,
        "reasoning_trace": final_state["reasoning"],
        "search_queries_used": final_state["search_queries"],
        "iterations": final_state["iteration"],
    }
