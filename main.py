import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from bm25_retrieval import run_bm25
from data_utils import QA_DATASET_FILE, CONTRACT_FOLDER, ensure_output_dir, load_data
from hnsw_retrieval import run_hnsw
from hybrid_retrieval import run_hybrid
from splade_retrieval import run_splade


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multiple retrieval systems on the contract QA dataset.")
    parser.add_argument("--contracts", default=CONTRACT_FOLDER, help="Folder containing contract PDFs.")
    parser.add_argument("--dataset", default=QA_DATASET_FILE, help="Path to dataset.json.")
    parser.add_argument("--output", default="output", help="Directory for saving run artifacts.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top documents to inspect per question.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output)
    corpus_df, qa_records = load_data(args.contracts, args.dataset)
    if corpus_df.empty:
        raise RuntimeError("No contract chunks were loaded – please check the contracts directory.")
    if not qa_records:
        raise RuntimeError("QA dataset is empty – please verify dataset.json.")

    qa_df = pd.DataFrame(qa_records)
    required_cols = {"question", "context_chunk"}
    missing_cols = [col for col in required_cols if col not in qa_df.columns]
    if missing_cols:
        raise RuntimeError(f"QA dataset missing required columns: {missing_cols}")

    metrics_summary: Dict[str, Dict] = {}
    errors: Dict[str, str] = {}

    jobs = [
        ("bm25", run_bm25),
        ("hnsw", run_hnsw),
        ("splade", run_splade),
        ("hybrid", run_hybrid),
    ]

    for name, fn in jobs:
        print(f"\n=== Running {name.upper()} retrieval ===")
        try:
            metrics_summary[name] = fn(corpus_df, qa_df, output_dir, top_k=args.top_k)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            errors[name] = str(exc)
            print(f"[ERROR] {name} run failed: {exc}")

    summary_payload = {
        "metrics": metrics_summary,
        "errors": errors,
        "output_dir": str(output_dir),
    }
    summary_path = Path(output_dir) / "summary_metrics.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print("\n=== Run Summary ===")
    for name, metrics in metrics_summary.items():
        accuracy = metrics.get("accuracy", 0.0)
        mrr = metrics.get("mrr", 0.0)
        print(f"{name.upper()}: accuracy={accuracy:.3f}, mrr={mrr:.3f}")
    if errors:
        print("Errors:")
        for name, message in errors.items():
            print(f"  - {name}: {message}")
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
