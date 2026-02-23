import re
from typing import Tuple


DISALLOWED_PATTERNS = [
    re.compile(r"ignore previous instructions", re.IGNORECASE),
    re.compile(r"reveal training data", re.IGNORECASE),
    re.compile(r"show me all patient", re.IGNORECASE),
]


def sanitize_prompt(prompt: str) -> Tuple[str, bool]:
    """
    Naive prompt sanitization that detects common injection phrases and flags them.
    Returns sanitized prompt and a boolean indicating whether it was modified/flagged.
    """
    flagged = False
    sanitized = prompt
    for pattern in DISALLOWED_PATTERNS:
        if pattern.search(sanitized):
            flagged = True
            sanitized = pattern.sub("[REDACTED-UNSAFE-CONTENT]", sanitized)
    return sanitized, flagged

