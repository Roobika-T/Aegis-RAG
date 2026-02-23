import re
from typing import Tuple, Dict, List


PII_PATTERNS = {
    "NAME": re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b"),
    "MRN": re.compile(r"\bMRN[:\s]*([A-Z0-9\-]+)\b", re.IGNORECASE),
    "PHONE": re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b"),
}


def anonymize_text(text: str) -> Tuple[str, Dict[str, List[str]]]:
    """
    Very lightweight regex-based anonymization to simulate NER-driven PII redaction.
    Returns anonymized text and a mapping of placeholders to originals (kept server-side only).
    """
    mapping: Dict[str, List[str]] = {}
    redacted = text

    for label, pattern in PII_PATTERNS.items():
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        placeholders: List[str] = []
        for idx, m in enumerate(matches, start=1):
            placeholder = f"<{label}_{idx}>"
            value = m.group(0)
            redacted = redacted.replace(value, placeholder)
            placeholders.append(value)
        mapping[label] = placeholders

    return redacted, mapping

