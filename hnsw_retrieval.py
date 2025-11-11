import json
import time
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from data_utils import FUZZY_THRESHOLD, fuzzy_match_rank

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HNSW_M = 32
EF_CONSTRUCTION = 200
EF_SEARCH = 128
BATCH_SIZE = 64


def _encode(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")


def _summarize_metrics(per_question: List[Dict]) -> Dict:
    num_rows = len(per_question)
    accuracy = sum(1 for row in per_question if row["context_match"]) / num_rows if num_rows else 0.0
    reciprocals = [
        1.0 / row["first_match_rank"]
        for row in per_question
        if isinstance(row["first_match_rank"], int) and row["first_match_rank"] > 0
    ]
    mrr = sum(reciprocals) / num_rows if num_rows and reciprocals else 0.0
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


def run_hnsw(corpus_df: pd.DataFrame, qa_df: pd.DataFrame, output_dir: Path, top_k: int = 5) -> Dict:
    if corpus_df.empty:
        raise ValueError("corpus_df is empty – cannot encode embeddings.")
    if qa_df.empty:
        raise ValueError("qa_df is empty – nothing to evaluate.")

    model_t0 = time.time()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    model_load_time = time.time() - model_t0
    print(f"[HNSW] Loaded '{EMBED_MODEL_NAME}' in {model_load_time:.2f}s")

    corpus_texts = corpus_df["chunk"].astype(str).tolist()
    query_texts = qa_df["question"].fillna("").astype(str).tolist()

    corpus_encode_t0 = time.time()
    corpus_embeddings = _encode(embed_model, corpus_texts)
    corpus_encode_time = time.time() - corpus_encode_t0
    print(f"[HNSW] Encoded {len(corpus_texts)} corpus chunks.")

    query_encode_t0 = time.time()
    question_embeddings = _encode(embed_model, query_texts)
    query_encode_time = time.time() - query_encode_t0
    print(f"[HNSW] Encoded {len(query_texts)} questions.")

    dim = corpus_embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.hnsw.efSearch = EF_SEARCH
    index.add(corpus_embeddings)

    search_t0 = time.time()
    scores, indices = index.search(question_embeddings, top_k)
    search_time = time.time() - search_t0
    print(f"[HNSW] Completed ANN search in {search_time:.2f}s.")

    corpus_chunks = corpus_df["chunk"].tolist()
    per_question: List[Dict] = []
    for row_idx in range(len(query_texts)):
        idxs = indices[row_idx]
        sims = scores[row_idx]
        top_chunks = [corpus_chunks[i] if i >= 0 else None for i in idxs]
        top_scores = [float(score) for score in sims]
        match_found, first_rank = fuzzy_match_rank(qa_df.iloc[row_idx].get("context_chunk", ""), top_chunks)
        per_question.append({
            "question": query_texts[row_idx],
            "context_chunk": qa_df.iloc[row_idx].get("context_chunk", ""),
            "top_chunks": top_chunks,
            "top_scores": top_scores,
            "best_chunk": top_chunks[0] if top_chunks else None,
            "best_score": top_scores[0] if top_scores else None,
            "context_match": match_found,
            "first_match_rank": first_rank,
        })

    metrics = _summarize_metrics(per_question)
    metrics.update({
        "method": "Semantic-HNSW",
        "model_name": EMBED_MODEL_NAME,
        "model_load_time_sec": model_load_time,
        "corpus_encode_time_sec": corpus_encode_time,
        "question_encode_time_sec": query_encode_time,
        "search_time_sec": search_time,
        "index_params": {"M": HNSW_M, "efConstruction": EF_CONSTRUCTION, "efSearch": EF_SEARCH},
    })

    payload = {
        "method": "Semantic-HNSW",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "per_question": per_question,
    }

    output_path = Path(output_dir) / "hnsw_results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"[HNSW] Accuracy: {metrics['accuracy']:.3f} | MRR: {metrics['mrr']:.3f}")
    print(f"[HNSW] Results stored at {output_path}")
    return metrics
