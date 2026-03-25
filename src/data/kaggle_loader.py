import glob
import os
from typing import List, Optional

import pandas as pd
import kagglehub

from ..security.anonymization import anonymize_text


def _discover_csv_files(root_path: str) -> List[str]:
    csv_files: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root_path):
        for name in filenames:
            if name.lower().endswith(".csv"):
                csv_files.append(os.path.join(dirpath, name))
    return sorted(csv_files)


def _row_to_doc(row: pd.Series, columns: List[str]) -> str:
    parts: List[str] = []
    for col in columns:
        if col not in row:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        parts.append(f"{col}: {val}")
    return "; ".join(parts)


def load_healthcare_docs_from_kaggle(
    dataset_id: str,
    max_rows: int = 200,
    text_columns: Optional[List[str]] = None,
    anonymize: bool = True,
) -> List[str]:
    """
    Download a Kaggle dataset via kagglehub and convert tabular rows into RAG text docs.
    """
    dataset_path = kagglehub.dataset_download(dataset_id)
    csv_files = _discover_csv_files(dataset_path)
    if not csv_files:
        raise RuntimeError(f"No CSV files found in downloaded dataset: {dataset_path}")

    docs: List[str] = []
    remaining = max_rows

    for csv_path in csv_files:
        if remaining <= 0:
            break

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        # Choose columns (default: first 6 columns) so the doc is short enough.
        columns = text_columns or list(df.columns[:6])
        for _idx, row in df.iterrows():
            doc = _row_to_doc(row, columns)
            if not doc.strip():
                continue
            if anonymize:
                doc, _mapping = anonymize_text(doc)
            docs.append(doc)
            remaining -= 1
            if remaining <= 0:
                break

    return docs

