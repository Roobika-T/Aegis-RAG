import re
import time
from typing import List

import requests

from ..config import get_settings


GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MAX_RETRIES_ON_RATE_LIMIT = 2
RETRY_DELAY_SECONDS = 5


def _build_rag_prompt(query: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks) if context_chunks else "(No relevant context retrieved.)"
    return (
        "You are a helpful assistant. Answer the user's question using only the provided context. "
        "If the context does not contain enough information, say so briefly. Keep answers concise.\n\n"
        "Context:\n"
        f"{context}\n\n"
        "User question:\n"
        f"{query}\n\n"
        "Answer:"
    )


def _call_ollama(base_url: str, model: str, prompt: str) -> str:
    """Call local Ollama API; returns generated text or raises. No rate limits."""
    url = f"{base_url.rstrip('/')}/api/generate"
    resp = requests.post(
        url,
        json={"model": model, "prompt": prompt, "stream": False},
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return (data.get("response") or "").strip()


def _call_gemini_api(api_key: str, model: str, prompt: str) -> str:
    """Call Gemini REST API; returns generated text or raises. Retries on 429."""
    url = GEMINI_URL.format(model=model) + f"?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
    }
    for attempt in range(MAX_RETRIES_ON_RATE_LIMIT + 1):
        resp = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        if resp.status_code == 429 and attempt < MAX_RETRIES_ON_RATE_LIMIT:
            time.sleep(RETRY_DELAY_SECONDS)
            continue
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return "No response generated from the model."
        parts = (candidates[0].get("content") or {}).get("parts") or []
        if not parts:
            return "No response generated from the model."
        return (parts[0].get("text") or "").strip()


def _safe_error_message(exc: Exception) -> str:
    """Return a user-facing error message that never includes API keys or secrets."""
    msg = str(exc)
    if "429" in msg or "Too Many Requests" in msg:
        return "Rate limit exceeded (429). Please try again in a minute."
    if "401" in msg or "403" in msg:
        return "Invalid or unauthorized API key."
    if "404" in msg:
        return "Model or endpoint not found."
    if "key=" in msg:
        msg = re.sub(r"key=[^&\s]+", "key=[REDACTED]", msg)
    return msg or "LLM provider error."


def generate_answer(query: str, context_chunks: List[str]) -> str:
    """
    Prefer Ollama (local) if OLLAMA_BASE_URL is set; else use Gemini if GEMINI_API_KEY is set;
    otherwise return a placeholder response.
    """
    settings = get_settings()
    prompt = _build_rag_prompt(query, context_chunks)
    context_preview = "\n\n".join(context_chunks[:3]) if context_chunks else "(none)"

    # 1. Try Ollama (local, no key, no rate limits)
    if settings.ollama_base_url:
        try:
            out = _call_ollama(
                settings.ollama_base_url,
                settings.ollama_model,
                prompt,
            )
            if out:
                return out
        except Exception as e:
            safe_err = _safe_error_message(e)
            return (
                "Ollama returned an error. Using fallback.\n\n"
                f"User question:\n{query}\n\n"
                f"Top retrieved context:\n{context_preview}\n\n(Error: {safe_err})"
            )

    # 2. Try Gemini (if key is set and not the literal placeholder)
    gemini_key = (settings.gemini_api_key or "").strip()
    if gemini_key and gemini_key != "your_api_key_here":
        try:
            out = _call_gemini_api(gemini_key, GEMINI_MODEL, prompt)
            if out:
                return out
        except Exception as e:
            safe_err = _safe_error_message(e)
            return (
                "The LLM provider returned an error. Using fallback.\n\n"
                f"User question:\n{query}\n\n"
                f"Top retrieved context:\n{context_preview}\n\n(Error: {safe_err})"
            )

    # 3. Placeholder
    return (
        "This is a demo answer (no LLM configured). Set OLLAMA_BASE_URL for local Ollama or GEMINI_API_KEY for Gemini.\n\n"
        f"User question:\n{query}\n\n"
        f"Top retrieved context:\n{context_preview}"
    )
