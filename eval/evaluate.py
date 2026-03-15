import json
import sys

sys.path.insert(0, "../app")

from retriever import CaptionRetriever


def evaluate(
    eval_path: str = "eval_dataset.json",
    output_path: str = "../results/eval_results.json",
):
    with open(eval_path) as f:
        questions = json.load(f)

    retriever = CaptionRetriever()
    results = {"questions": [], "metrics": {}}

    for k in [1, 3, 5]:
        hits = 0
        reciprocal_ranks = []
        precisions = []

        for q in questions:
            retrieved = retriever.search(q["question"], top_k=k)
            retrieved_ids = [r["id"] for r in retrieved]
            relevant_ids = set(q["relevant_document_ids"])

            # Hit Rate: 1 if any relevant doc in top-k
            hit = 1 if relevant_ids & set(retrieved_ids) else 0
            hits += hit

            # MRR: reciprocal of rank of first relevant doc
            rr = 0.0
            for rank, rid in enumerate(retrieved_ids, 1):
                if rid in relevant_ids:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

            # Precision@k: fraction of retrieved that are relevant
            precision = len(relevant_ids & set(retrieved_ids)) / k
            precisions.append(precision)

            if k == 5:
                results["questions"].append(
                    {
                        "question_id": q["question_id"],
                        "question": q["question"],
                        "ground_truth_ids": q["relevant_document_ids"],
                        "retrieved_ids": retrieved_ids,
                        "hit": bool(hit),
                        "reciprocal_rank": rr,
                        "precision": precision,
                        "type": q.get("question_type", "unknown"),
                    }
                )

        n = len(questions)
        results["metrics"][f"k={k}"] = {
            "hit_rate": round(hits / n, 4),
            "mrr": round(sum(reciprocal_ranks) / n, 4),
            "precision": round(sum(precisions) / n, 4),
        }

    # Qualitative analysis
    successes = [q for q in results["questions"] if q["hit"]]
    failures = [q for q in results["questions"] if not q["hit"]]
    results["qualitative"] = {
        "success_examples": successes[:3],
        "failure_examples": failures[:2],
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation Results:")
    print("=" * 50)
    for k_key, metrics in results["metrics"].items():
        print(
            f"  {k_key}: Hit Rate={metrics['hit_rate']:.4f}, "
            f"MRR={metrics['mrr']:.4f}, Precision={metrics['precision']:.4f}"
        )
    print(f"\nSuccesses (k=5): {len(successes)}/{len(results['questions'])}")
    print(f"Failures  (k=5): {len(failures)}/{len(results['questions'])}")
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    evaluate()
