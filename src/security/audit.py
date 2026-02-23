from datetime import datetime
from loguru import logger
from typing import Any, Dict, Optional


def audit_event(
    user_role: Optional[str],
    action: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "role": user_role,
        "action": action,
        "details": details or {},
    }
    logger.bind(audit=True).info(payload)

