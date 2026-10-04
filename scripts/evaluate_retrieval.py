from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval.vector_store import get_connection
from retrieval.rag_pipeline import evaluate_retrieval


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality.")
    parser.add_argument("--qa_file", required=True, help="JSON file with Q&A pairs for evaluation.")
    parser.add_argument("--dsn", default=None, help="PostgreSQL DSN string.")
    parser.add_argument("--top_k_values", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--no_hybrid", action="store_true")
    parser.add_argument("--output", default="retrieval_eval_results.json")
    args = parser.parse_args()

    with open(args.qa_file) as f:
        qa_pairs = json.load(f)

    conn = get_connection(args.dsn)
    results = evaluate_retrieval(
        conn=conn,
        qa_pairs=qa_pairs,
        top_k_values=args.top_k_values,
        use_hybrid=not args.no_hybrid,
    )
    conn.close()

    print("\nRetrieval Evaluation Results:")
    print("=" * 40)
    for k, v in results.items():
        print(f"  {k:<20}: {v:.4f}")
    print("=" * 40)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
