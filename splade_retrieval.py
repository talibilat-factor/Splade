import json
import time
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from data_utils import FUZZY_THRESHOLD, fuzzy_match_rank

SPLADE_MODEL_NAME = "naver/efficient-splade-VI-BT-large-doc"
SPLADE_BATCH_SIZE = 16


def _encode_splade(
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    device: torch.device,
    texts: List[str],
    batch_size: int = SPLADE_BATCH_SIZE,
) -> np.ndarray:
    if not texts:
        vocab_size = model.config.vocab_size
        return np.zeros((0, vocab_size), dtype="float32")
    encoded_batches = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs)
            logits = outputs.logits  # (batch, seq_len, vocab_size)
            activations = torch.log1p(torch.relu(logits))
            masked = activations * inputs.attention_mask.unsqueeze(-1)
            vec = torch.max(masked, dim=1).values
            encoded_batches.append(vec.cpu().numpy())
    return np.concatenate(encoded_batches, axis=0).astype("float32")


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


def run_splade(corpus_df: pd.DataFrame, qa_df: pd.DataFrame, output_dir: Path, top_k: int = 5) -> Dict:
    if corpus_df.empty:
        raise ValueError("corpus_df is empty – cannot encode SPLADE vectors.")
    if qa_df.empty:
        raise ValueError("qa_df is empty – nothing to evaluate.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(SPLADE_MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL_NAME).to(device)
    model_load_time = time.time() - load_t0
    print(f"[SPLADE] Loaded '{SPLADE_MODEL_NAME}' on {device} in {model_load_time:.2f}s.")

    corpus_texts = corpus_df["chunk"].astype(str).tolist()
    query_texts = qa_df["question"].fillna("").astype(str).tolist()

    corpus_encode_t0 = time.time()
    corpus_embeddings = _encode_splade(tokenizer, model, device, corpus_texts)
    corpus_encode_time = time.time() - corpus_encode_t0
    print(f"[SPLADE] Encoded {len(corpus_texts)} corpus chunks.")

    query_encode_t0 = time.time()
    question_embeddings = _encode_splade(tokenizer, model, device, query_texts)
    query_encode_time = time.time() - query_encode_t0
    print(f"[SPLADE] Encoded {len(query_texts)} questions.")

    dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_embeddings)

    search_t0 = time.time()
    scores, indices = index.search(question_embeddings, top_k)
    search_time = time.time() - search_t0
    print(f"[SPLADE] Completed retrieval in {search_time:.2f}s.")

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
        "method": "SPLADE",
        "model_name": SPLADE_MODEL_NAME,
        "model_load_time_sec": model_load_time,
        "corpus_encode_time_sec": corpus_encode_time,
        "question_encode_time_sec": query_encode_time,
        "search_time_sec": search_time,
        "device": str(device),
    })

    payload = {
        "method": "SPLADE",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "per_question": per_question,
    }

    output_path = Path(output_dir) / "splade_results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"[SPLADE] Accuracy: {metrics['accuracy']:.3f} | MRR: {metrics['mrr']:.3f}")
    print(f"[SPLADE] Results stored at {output_path}")
    return metrics
