import os

import requests
import streamlit as st

FLASK_API_URL = os.getenv("FLASK_API_URL", "http://localhost:5000")

st.set_page_config(page_title="RAG Document Intelligence", layout="wide")
st.title("Remote Sensing Image Caption RAG System")

# Sidebar: Health check
with st.sidebar:
    st.header("System Status")
    if st.button("Check Health"):
        try:
            r = requests.get(f"{FLASK_API_URL}/health", timeout=10)
            data = r.json()
            st.write(f"**Overall:** {data['status']}")
            for service, status in data["services"].items():
                icon = ":white_check_mark:" if status == "ok" else ":x:"
                st.write(f"{icon} {service}: {status}")
        except Exception as e:
            st.error(f"API unreachable: {e}")

    st.divider()
    st.header("Collection Info")
    if st.button("Refresh Info"):
        try:
            r = requests.get(f"{FLASK_API_URL}/documents", timeout=30)
            if r.ok:
                data = r.json()
                st.metric("Total Documents", data["total"])
        except Exception as e:
            st.error(str(e))

# Tab layout
tab_ingest, tab_query, tab_docs = st.tabs(
    ["Ingest Documents", "Query", "Document List"]
)

with tab_ingest:
    st.header("Ingest Captions from Dataset")
    limit = st.number_input(
        "Number of captions to ingest", min_value=1, max_value=13820, value=1000
    )
    if st.button("Ingest Captions"):
        with st.spinner(f"Ingesting {limit} captions..."):
            try:
                r = requests.post(
                    f"{FLASK_API_URL}/documents",
                    json={"source": "dataset", "limit": int(limit)},
                    timeout=600,
                )
                if r.ok:
                    st.success(f"Done! {r.json()['message']}")
                else:
                    st.error(f"Error: {r.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

    st.divider()
    st.header("Upload Document (PDF/DOCX)")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])
    if uploaded_file and st.button("Upload & Index"):
        with st.spinner("Uploading and indexing..."):
            try:
                r = requests.post(
                    f"{FLASK_API_URL}/documents",
                    files={"file": (uploaded_file.name, uploaded_file.read())},
                    timeout=120,
                )
                if r.ok:
                    data = r.json()
                    st.success(
                        f"Uploaded! Document ID: {data['document_id']}, "
                        f"Chunks indexed: {data['chunks_indexed']}"
                    )
                else:
                    st.error(f"Error: {r.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

with tab_query:
    st.header("Ask a Question")
    question = st.text_input(
        "Enter your question about remote sensing images:",
        placeholder="e.g., What kind of area has a highway interchange?",
    )
    top_k = st.slider("Number of source passages", 1, 10, 5)

    if st.button("Submit Query") and question:
        with st.spinner("Agent is thinking..."):
            try:
                r = requests.post(
                    f"{FLASK_API_URL}/query",
                    json={"question": question, "top_k": top_k},
                    timeout=300,
                )
                if r.ok:
                    data = r.json()

                    st.subheader("Answer")
                    st.write(data["answer"])

                    st.subheader("Agent Reasoning Trace")
                    for step in data.get("reasoning_trace", []):
                        st.write(f"- {step}")

                    st.subheader(f"Source Passages ({len(data.get('source_chunks', []))})")
                    for i, chunk in enumerate(data.get("source_chunks", [])):
                        score = chunk.get("relevance_score", 0)
                        with st.expander(
                            f"Passage {i + 1} (Score: {score:.4f})"
                        ):
                            st.write(f"**Document ID:** {chunk['document_id']}")
                            st.write(f"**Image:** {chunk.get('image', 'N/A')}")
                            st.write(f"**Tags:** {chunk.get('tags', 'N/A')}")
                            st.divider()
                            st.write(chunk["text"])

                    with st.expander("Debug: Search queries used"):
                        for q in data.get("search_queries_used", []):
                            st.code(q)
                else:
                    st.error(f"Error: {r.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

with tab_docs:
    st.header("Indexed Documents")
    if st.button("Refresh Document List"):
        try:
            r = requests.get(f"{FLASK_API_URL}/documents", timeout=30)
            if r.ok:
                data = r.json()
                st.write(f"**Total documents:** {data['total']}")
                for doc in data["documents"][:100]:
                    col1, col2, col3 = st.columns([3, 2, 1])
                    col1.write(doc["id"])
                    col2.write(doc.get("source", "N/A"))
                    if col3.button("Delete", key=f"del-{doc['id']}"):
                        rd = requests.delete(
                            f"{FLASK_API_URL}/documents/{doc['id']}", timeout=30
                        )
                        if rd.ok:
                            st.success(f"Deleted {doc['id']}")
                            st.rerun()
                        else:
                            st.error(f"Failed: {rd.text}")
            else:
                st.error(f"Error: {r.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
