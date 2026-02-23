from cryptography.fernet import Fernet
import base64
import hashlib


def _derive_fernet_key(raw_key: str) -> bytes:
    """
    Derive a 32-byte key from an arbitrary string.
    This is for demo purposes only; use a proper KDF in production.
    """
    digest = hashlib.sha256(raw_key.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def build_cipher(secret: str) -> Fernet:
    key = _derive_fernet_key(secret)
    return Fernet(key)


def encrypt_str(cipher: Fernet, value: str) -> str:
    return cipher.encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_str(cipher: Fernet, token: str) -> str:
    return cipher.decrypt(token.encode("utf-8")).decode("utf-8")

