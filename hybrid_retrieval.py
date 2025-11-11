import json
import time
from pathlib import Path
from typing import Dict, List, Set

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from data_utils import FUZZY_THRESHOLD, fuzzy_match_rank, tokenize

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HYBRID_TOP_BM25 = 25
HYBRID_TOP_HNSW = 50
HYBRID_BM25_WEIGHT = 0.55  # lexical weight vs dense weight
HYBRID_TOP_K = 5


def _build_bm25(corpus_df: pd.DataFrame) -> BM25Okapi:
    tokens = [tokenize(str(chunk)) for chunk in corpus_df["chunk"].tolist()]
    print(f"[Hybrid] BM25 tokenized {len(tokens)} chunks.")
    return BM25Okapi(tokens)


def _encode(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")


def _summarize_metrics(per_question: List[Dict]) -> Dict:
    total = len(per_question)
    accuracy = sum(1 for row in per_question if row["context_match"]) / total if total else 0.0
    reciprocals = [
        1.0 / row["first_match_rank"]
        for row in per_question
        if isinstance(row["first_match_rank"], int) and row["first_match_rank"] > 0
    ]
    mrr = sum(reciprocals) / total if total and reciprocals else 0.0
    rank_counts: Dict[str, int] = {}
    for row in per_question:
        key = str(row["first_match_rank"]) if row["first_match_rank"] is not None else "None"
        rank_counts[key] = rank_counts.get(key, 0) + 1
    return {
        "num_questions": total,
        "accuracy": accuracy,
        "mrr": mrr,
        "rank_counts": rank_counts,
        "fuzzy_threshold": FUZZY_THRESHOLD,
    }


def run_hybrid(
    corpus_df: pd.DataFrame,
    qa_df: pd.DataFrame,
    output_dir: Path,
    top_k: int = HYBRID_TOP_K,
    top_bm25: int = HYBRID_TOP_BM25,
    top_hnsw: int = HYBRID_TOP_HNSW,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
) -> Dict:
    if corpus_df.empty:
        raise ValueError("corpus_df is empty – cannot run hybrid retriever.")
    if qa_df.empty:
        raise ValueError("qa_df is empty – nothing to evaluate.")

    bm25 = _build_bm25(corpus_df)
    corpus_chunks = corpus_df["chunk"].astype(str).tolist()

    model_t0 = time.time()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    model_load_time = time.time() - model_t0
    print(f"[Hybrid] Loaded '{EMBED_MODEL_NAME}' in {model_load_time:.2f}s")

    corpus_embeddings = _encode(embed_model, corpus_chunks)
    dim = corpus_embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 128
    index.add(corpus_embeddings)

    question_texts = qa_df["question"].fillna("").astype(str).tolist()
    question_embeddings = _encode(embed_model, question_texts)

    dense_weight = 1.0 - bm25_weight
    per_question: List[Dict] = []

    for qi, question_text in enumerate(question_texts):
        tokens = tokenize(question_text)
        bm25_scores_raw = bm25.get_scores(tokens) if tokens else np.zeros(len(corpus_chunks))
        bm25_order = np.argsort(bm25_scores_raw)[::-1][:top_bm25]
        bm25_max = float(max(bm25_scores_raw[bm25_order])) if bm25_order.size else 0.0
        bm25_norm = {idx: (bm25_scores_raw[idx] / bm25_max) if bm25_max > 0 else 0.0 for idx in bm25_order}

        q_vec = question_embeddings[qi]
        sims, ann_idx = index.search(q_vec.reshape(1, -1), top_hnsw)
        ann_indices = [i for i in ann_idx[0] if i >= 0]

        candidate_ids: Set[int] = set(bm25_order.tolist())
        candidate_ids.update(ann_indices)

        if not candidate_ids:
            candidate_ids = set(bm25_order.tolist() or ann_indices or [])

        dense_scores = {idx: float(np.dot(corpus_embeddings[idx], q_vec)) for idx in candidate_ids}
        dense_norm = {idx: (score + 1.0) / 2.0 for idx, score in dense_scores.items()}  # map to [0,1]

        combined_scores = {
            idx: bm25_weight * bm25_norm.get(idx, 0.0) + dense_weight * dense_norm.get(idx, 0.0)
            for idx in candidate_ids
        }

        ranked_candidates = sorted(combined_scores, key=combined_scores.get, reverse=True)
        top_indices = ranked_candidates[:top_k]
        top_chunks = [corpus_chunks[i] for i in top_indices]
        top_scores = [combined_scores[i] for i in top_indices]
        best_chunk = top_chunks[0] if top_chunks else None
        best_score = top_scores[0] if top_scores else None

        match_found, first_rank = fuzzy_match_rank(qa_df.iloc[qi].get("context_chunk", ""), top_chunks)

        per_question.append({
            "question": question_text,
            "context_chunk": qa_df.iloc[qi].get("context_chunk", ""),
            "top_chunks": top_chunks,
            "top_scores": top_scores,
            "best_chunk": best_chunk,
            "best_score": best_score,
            "context_match": match_found,
            "first_match_rank": first_rank,
        })

    metrics = _summarize_metrics(per_question)
    metrics.update({
        "method": "Hybrid(BM25+HNSW)",
        "bm25_weight": bm25_weight,
        "dense_weight": dense_weight,
        "bm25_candidates": top_bm25,
        "hnsw_candidates": top_hnsw,
        "top_k": top_k,
        "model_name": EMBED_MODEL_NAME,
        "model_load_time_sec": model_load_time,
    })

    payload = {
        "method": "Hybrid(BM25+HNSW)",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "per_question": per_question,
    }

    output_path = Path(output_dir) / "hybrid_results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"[Hybrid] Accuracy: {metrics['accuracy']:.3f} | MRR: {metrics['mrr']:.3f}")
    print(f"[Hybrid] Results stored at {output_path}")
    return metrics
