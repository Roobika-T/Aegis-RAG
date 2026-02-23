from typing import Dict, Optional


class InMemoryTokenizer:
    """
    Simple tokenization vault for sensitive identifiers (e.g., account or record numbers).
    In production, back this with an HSM or dedicated tokenization service.
    """

    def __init__(self) -> None:
        self._forward: Dict[str, str] = {}
        self._reverse: Dict[str, str] = {}
        self._counter = 0

    def tokenize(self, raw: str) -> str:
        if raw in self._forward:
            return self._forward[raw]
        self._counter += 1
        token = f"TOKEN-{self._counter:08d}"
        self._forward[raw] = token
        self._reverse[token] = raw
        return token

    def detokenize(self, token: str) -> Optional[str]:
        return self._reverse.get(token)

