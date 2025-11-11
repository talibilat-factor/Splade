import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from data_utils import FUZZY_THRESHOLD, fuzzy_match_rank, tokenize


def _build_bm25(corpus_df: pd.DataFrame) -> BM25Okapi:
    tokens = [tokenize(str(chunk)) for chunk in corpus_df["chunk"].tolist()]
    print(f"[BM25] Tokenized {len(tokens)} chunks.")
    return BM25Okapi(tokens)


def _summarize_metrics(per_question: List[Dict]) -> Dict:
    num_rows = len(per_question)
    accuracy = sum(1 for row in per_question if row["context_match"]) / num_rows if num_rows else 0.0
    mrr = 0.0
    if num_rows:
        reciprocals = [1.0 / row["first_match_rank"] for row in per_question if isinstance(row["first_match_rank"], int) and row["first_match_rank"] > 0]
        mrr = sum(reciprocals) / num_rows if reciprocals else 0.0
    rank_counts: Dict[str, int] = {}
    for row in per_question:
        key = str(row["first_match_rank"]) if row["first_match_rank"] is not None else "None"
        rank_counts[key] = rank_counts.get(key, 0) + 1
    return {
        "num_questions": num_rows,
        "accuracy": accuracy,
        "mrr": mrr,
        "rank_counts": rank_counts,
        "fuzzy_threshold": FUZZY_THRESHOLD,
    }


def run_bm25(corpus_df: pd.DataFrame, qa_df: pd.DataFrame, output_dir: Path, top_k: int = 5) -> Dict:
    if corpus_df.empty:
        raise ValueError("corpus_df is empty – cannot build BM25 index.")
    if qa_df.empty:
        raise ValueError("qa_df is empty – nothing to evaluate.")

    t0 = time.time()
    bm25 = _build_bm25(corpus_df)
    build_time = time.time() - t0
    corpus_chunks = corpus_df["chunk"].tolist()

    per_question: List[Dict] = []
    for idx, row in qa_df.iterrows():
        question_text = row.get("question", "") or ""
        tokens = tokenize(question_text)
        if not tokens:
            per_question.append({
                "question": question_text,
                "context_chunk": row.get("context_chunk", ""),
                "top_chunks": [],
                "top_scores": [],
                "best_chunk": None,
                "best_score": None,
                "context_match": False,
                "first_match_rank": None,
            })
            continue

        scores = bm25.get_scores(tokens)
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        top_chunks = [corpus_chunks[i] for i in ranked_idx]
        top_scores = [float(scores[i]) for i in ranked_idx]
        best_chunk = top_chunks[0] if top_chunks else None
        best_score = top_scores[0] if top_scores else None

        match_found, first_rank = fuzzy_match_rank(row.get("context_chunk", "") or "", top_chunks)
        per_question.append({
            "question": question_text,
            "context_chunk": row.get("context_chunk", ""),
            "top_chunks": top_chunks,
            "top_scores": top_scores,
            "best_chunk": best_chunk,
            "best_score": best_score,
            "context_match": match_found,
            "first_match_rank": first_rank,
        })

    metrics = _summarize_metrics(per_question)
    metrics["build_time_sec"] = build_time
    metrics["method"] = "BM25"

    payload = {
        "method": "BM25",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "per_question": per_question,
    }

    output_path = Path(output_dir) / "bm25_results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"[BM25] Completed evaluation for {metrics['num_questions']} questions.")
    print(f"[BM25] Accuracy: {metrics['accuracy']:.3f} | MRR: {metrics['mrr']:.3f}")
    print(f"[BM25] Results stored at {output_path}")
    return metrics
