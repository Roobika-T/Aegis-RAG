import os
from typing import List, Optional

import pandas as pd

from ..security.anonymization import anonymize_text


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


def load_healthcare_docs_from_local_csv(
    csv_path: str,
    max_rows: int = 200,
    text_columns: Optional[List[str]] = None,
    anonymize: bool = True,
) -> List[str]:
    """
    Load a local healthcare CSV into RAG documents.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    columns = text_columns or list(df.columns[:6])
    docs: List[str] = []
    remaining = max_rows

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

