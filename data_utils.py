import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import PyPDF2

CONTRACT_FOLDER = "contracts"
QA_DATASET_FILE = "dataset.json"
FUZZY_THRESHOLD = 70


def extract_source_filename(raw: Optional[str]) -> Optional[str]:
    """Extract the PDF filename embedded in the dataset's source document field."""
    if not raw:
        return None
    parts = raw.split()
    for token in parts:
        if token.lower().endswith(".pdf"):
            return token
    return None


def load_data(
    contract_folder_path: str = CONTRACT_FOLDER,
    qa_dataset_file_path: str = QA_DATASET_FILE,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Load contract text (chunk-level) and QA evaluation rows."""
    rows = []
    if os.path.isdir(contract_folder_path):
        for filename in os.listdir(contract_folder_path):
            if not filename.lower().endswith(".pdf"):
                continue
            filepath = os.path.join(contract_folder_path, filename)
            try:
                with open(filepath, "rb") as handle:
                    reader = PyPDF2.PdfReader(handle)
                    for page in reader.pages:
                        extracted_text = page.extract_text() or ""
                        for line in extracted_text.split("\n"):
                            line = line.strip()
                            if line and len(line.split()) > 5:
                                rows.append({"chunk": line, "source_pdf": filename})
            except Exception as exc:  # pragma: no cover - best-effort parsing
                print(f"[WARN] Could not read {filepath}: {exc}")
    else:
        print(f"[WARN] Contract folder does not exist: {os.path.abspath(contract_folder_path)}")

    qa_dataset = []
    if os.path.isfile(qa_dataset_file_path):
        try:
            with open(qa_dataset_file_path, "r", encoding="utf-8") as handle:
                qa_dataset = json.load(handle)
                if isinstance(qa_dataset, dict):
                    qa_dataset = list(qa_dataset.values())
                elif not isinstance(qa_dataset, list):
                    qa_dataset = [qa_dataset]
        except json.JSONDecodeError as exc:
            print(f"[ERROR] Failed to decode {qa_dataset_file_path}: {exc}")
        except FileNotFoundError:
            print(f"[ERROR] Dataset file not found at {qa_dataset_file_path}")
    else:
        print(f"[WARN] Dataset file missing: {os.path.abspath(qa_dataset_file_path)}")

    normalized = []
    for item in qa_dataset:
        if not isinstance(item, dict):
            continue
        norm = {
            "question": item.get("question") or item.get("query") or "",
            "answer_snippet": item.get("answer_snippet") or item.get("answer") or "",
            "context_chunk": item.get("context_chunk") or item.get("context") or "",
            "source_document_raw": item.get("source document") or item.get("source_document") or "",
        }
        norm["source_document"] = extract_source_filename(norm["source_document_raw"])
        normalized.append(norm)

    corpus_df = pd.DataFrame(rows)
    print(f"[DATA] Loaded {len(normalized)} QA rows and {len(corpus_df)} contract chunks.")
    return corpus_df, normalized


def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"\w+", text.lower())


def fuzzy_match_rank(
    gold_context: str,
    candidates: Iterable[Optional[str]],
    threshold: int = FUZZY_THRESHOLD,
) -> Tuple[bool, Optional[int]]:
    """Return whether any candidate satisfies the fuzzy threshold and its 1-based rank."""
    try:
        from rapidfuzz import fuzz
    except ImportError:  # pragma: no cover - dependency managed elsewhere
        raise RuntimeError("rapidfuzz is required for evaluation scoring.")

    gold_context = gold_context or ""
    for rank_idx, candidate in enumerate(candidates, start=1):
        if not candidate:
            continue
        sim_score = fuzz.token_set_ratio(gold_context, candidate)
        if sim_score >= threshold:
            return True, rank_idx
    return False, None


def ensure_output_dir(path: str) -> Path:
    """Create the output directory if needed and return it as a Path."""
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path
