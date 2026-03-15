import json
import sys
from collections import defaultdict

sys.path.insert(0, "../app")

from retriever import CaptionRetriever


def _compute_metrics(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> tuple[int, float, float]:
    hit = 1 if relevant_ids & set(retrieved_ids) else 0

    rr = 0.0
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            rr = 1.0 / rank
            break

    precision = len(relevant_ids & set(retrieved_ids)) / k
    return hit, rr, precision


def evaluate(
    eval_path: str = "eval_dataset.json",
    output_path: str = "../results/eval_results.json",
):
    with open(eval_path) as f:
        questions = json.load(f)

    # Same question text can map to multiple relevant IDs in this dataset.
    # Relaxed evaluation counts any of those IDs as relevant.
    question_to_relevant_ids: dict[str, set[str]] = defaultdict(set)
    for q in questions:
        question_to_relevant_ids[q["question"]].update(q["relevant_document_ids"])

    retriever = CaptionRetriever()
    results = {"questions": [], "metrics": {}}

    for k in [1, 3, 5]:
        exact_hits = 0
        exact_rrs = []
        exact_precisions = []
        relaxed_hits = 0
        relaxed_rrs = []
        relaxed_precisions = []

        for q in questions:
            retrieved = retriever.search(q["question"], top_k=k)
            retrieved_ids = [r["id"] for r in retrieved]
            exact_relevant_ids = set(q["relevant_document_ids"])
            relaxed_relevant_ids = question_to_relevant_ids[q["question"]]

            exact_hit, exact_rr, exact_precision = _compute_metrics(
                retrieved_ids,
                exact_relevant_ids,
                k,
            )
            relaxed_hit, relaxed_rr, relaxed_precision = _compute_metrics(
                retrieved_ids,
                relaxed_relevant_ids,
                k,
            )

            exact_hits += exact_hit
            exact_rrs.append(exact_rr)
            exact_precisions.append(exact_precision)
            relaxed_hits += relaxed_hit
            relaxed_rrs.append(relaxed_rr)
            relaxed_precisions.append(relaxed_precision)

            if k == 5:
                results["questions"].append(
                    {
                        "question_id": q["question_id"],
                        "question": q["question"],
                        "ground_truth_ids": q["relevant_document_ids"],
                        "relaxed_ground_truth_ids": sorted(list(relaxed_relevant_ids)),
                        "retrieved_ids": retrieved_ids,
                        "hit_exact": bool(exact_hit),
                        "hit_relaxed": bool(relaxed_hit),
                        "reciprocal_rank_exact": exact_rr,
                        "reciprocal_rank_relaxed": relaxed_rr,
                        "precision_exact": exact_precision,
                        "precision_relaxed": relaxed_precision,
                        "type": q.get("question_type", "unknown"),
                    }
                )

        n = len(questions)
        results["metrics"][f"k={k}"] = {
            "exact": {
                "hit_rate": round(exact_hits / n, 4),
                "mrr": round(sum(exact_rrs) / n, 4),
                "precision": round(sum(exact_precisions) / n, 4),
            },
            "relaxed": {
                "hit_rate": round(relaxed_hits / n, 4),
                "mrr": round(sum(relaxed_rrs) / n, 4),
                "precision": round(sum(relaxed_precisions) / n, 4),
            },
        }

    # Qualitative analysis
    successes_exact = [q for q in results["questions"] if q["hit_exact"]]
    failures_exact = [q for q in results["questions"] if not q["hit_exact"]]
    successes_relaxed = [q for q in results["questions"] if q["hit_relaxed"]]
    failures_relaxed = [q for q in results["questions"] if not q["hit_relaxed"]]
    results["qualitative"] = {
        "success_examples_exact": successes_exact[:3],
        "failure_examples_exact": failures_exact[:2],
        "success_examples_relaxed": successes_relaxed[:3],
        "failure_examples_relaxed": failures_relaxed[:2],
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation Results:")
    print("=" * 50)
    for k_key, metrics in results["metrics"].items():
        print(
            f"  {k_key} exact: "
            f"Hit Rate={metrics['exact']['hit_rate']:.4f}, "
            f"MRR={metrics['exact']['mrr']:.4f}, "
            f"Precision={metrics['exact']['precision']:.4f}"
        )
        print(
            f"  {k_key} relaxed: "
            f"Hit Rate={metrics['relaxed']['hit_rate']:.4f}, "
            f"MRR={metrics['relaxed']['mrr']:.4f}, "
            f"Precision={metrics['relaxed']['precision']:.4f}"
        )
    print(f"\nSuccesses (k=5 exact): {len(successes_exact)}/{len(results['questions'])}")
    print(f"Failures  (k=5 exact): {len(failures_exact)}/{len(results['questions'])}")
    print(f"Successes (k=5 relaxed): {len(successes_relaxed)}/{len(results['questions'])}")
    print(f"Failures  (k=5 relaxed): {len(failures_relaxed)}/{len(results['questions'])}")
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    evaluate()
