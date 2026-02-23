from typing import Literal, Optional

Role = Literal["doctor", "nurse", "auditor"]


def resolve_role_from_api_key(api_key: str, api_keys: dict) -> Optional[Role]:
    """
    Simple API-key to role mapping.
    In production, integrate with IAM/IdP instead.
    """
    return api_keys.get(api_key)  # type: ignore[return-value]


def can_access_patient_details(role: Role) -> bool:
    return role in ("doctor", "nurse")


def can_view_audit_logs(role: Role) -> bool:
    return role == "auditor"

