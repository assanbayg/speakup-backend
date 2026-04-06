"""Chat service — Groq API with Ollama fallback."""

from typing import Optional, List, Dict, Any, AsyncIterator
from collections import OrderedDict
import os
import json

import httpx

from config import OLLAMA_URL, LLM_MODEL

# ---------------------------------------------------------------------------
# Groq config
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def _use_groq() -> bool:
    return bool(GROQ_API_KEY)

# ---------------------------------------------------------------------------
# In-memory conversation history
# ---------------------------------------------------------------------------
MAX_SESSIONS = 200
MAX_MESSAGES_PER_SESSION = 20

_history: OrderedDict[str, List[Dict[str, str]]] = OrderedDict()


def _get_history(session_id: str) -> List[Dict[str, str]]:
    if session_id in _history:
        _history.move_to_end(session_id)
        return _history[session_id]
    if len(_history) >= MAX_SESSIONS:
        _history.popitem(last=False)
    _history[session_id] = []
    return _history[session_id]


def _append_history(session_id: str, role: str, content: str) -> None:
    history = _get_history(session_id)
    history.append({"role": role, "content": content})
    if len(history) > MAX_MESSAGES_PER_SESSION:
        del history[: len(history) - MAX_MESSAGES_PER_SESSION]


def clear_history(session_id: str) -> None:
    _history.pop(session_id, None)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """Ты — Спичи, весёлый и любопытный друг-собеседник для ребёнка.

ПРАВИЛА (строго соблюдай):
• Отвечай ТОЛЬКО на русском языке. НИКОГДА не используй английский, даже отдельные слова. Не спрашивай про язык.
• Отвечай КРАТКО: 1–2 предложения максимум.
• Говори просто, как с ребёнком 5–8 лет. Без сложных слов, без медицинских терминов.
• Будь игривым, тёплым, ободряющим. Используй восклицания (ура, ого, вау).
• ВСЕГДА заканчивай свой ответ вопросом или предложением продолжить разговор. Примеры: «А ты?», «Расскажи ещё!», «А что было дальше?», «Хочешь поиграем в слова?»
• Если сообщение ребёнка непонятное или бессмысленное — НЕ повторяй эти слова. Вместо этого мягко переспроси: «Ой, я не расслышал! Скажи ещё разок?» или предложи тему: «Давай поговорим о животных! Какое твоё любимое?»
• Никогда не повторяй и не цитируй то, что сказал ребёнок, дословно.
• Не упоминай речевые нарушения, дизартрию, логопедию, терапию, чёткость речи.

СТИЛЬ:
Ты как мультяшный персонаж — добрый, чуть смешной, всегда рад поговорить."""


def _prepare_messages(
    messages: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    return [{"role": "system", "content": _SYSTEM_PROMPT}] + messages


# ---------------------------------------------------------------------------
# Groq API call
# ---------------------------------------------------------------------------

async def _groq_chat(messages: List[Dict[str, str]]) -> str:
    """Call Groq API (OpenAI-compatible)."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": 150,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(GROQ_URL, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def _groq_chat_stream(messages: List[Dict[str, str]]) -> AsyncIterator[bytes]:
    """Stream from Groq API."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": 150,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", GROQ_URL, headers=headers, json=body) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        # Emit in Ollama-compatible ndjson format so the
                        # Flutter streaming client doesn't need changes
                        ollama_chunk = json.dumps({
                            "message": {"role": "assistant", "content": content},
                            "done": False,
                        })
                        yield (ollama_chunk + "\n").encode()
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            # Send final done message
            yield json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}).encode()


# ---------------------------------------------------------------------------
# Ollama fallback
# ---------------------------------------------------------------------------

_OLLAMA_OPTIONS = {
    "num_predict": 80,
    "temperature": 0.7,
    "top_p": 0.9,
    "repeat_penalty": 1.3,
}


async def _ollama_chat(messages: List[Dict[str, str]]) -> str:
    body = {
        "model": LLM_MODEL,
        "stream": False,
        "messages": messages,
        "options": _OLLAMA_OPTIONS,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(f"{OLLAMA_URL}/api/chat", json=body)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")


async def _ollama_chat_stream(messages: List[Dict[str, str]]) -> AsyncIterator[bytes]:
    body = {
        "model": LLM_MODEL,
        "stream": True,
        "messages": messages,
        "options": _OLLAMA_OPTIONS,
    }
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=body) as r:
            async for chunk in r.aiter_bytes():
                yield chunk


# ---------------------------------------------------------------------------
# Public API (unchanged contract)
# ---------------------------------------------------------------------------

async def chat_stream(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[bytes]:
    """Stream chat response."""
    prepared = _prepare_messages(messages)

    if _use_groq():
        async for chunk in _groq_chat_stream(prepared):
            yield chunk
    else:
        async for chunk in _ollama_chat_stream(prepared):
            yield chunk


async def chat_sync(
    message: str,
    model: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> str:
    """Non-streaming chat with conversation memory."""
    sid = session_id or "_default"

    _append_history(sid, "user", message)
    history = _get_history(sid)
    prepared = _prepare_messages(list(history))

    if _use_groq():
        print(f"[CHAT] Using Groq ({GROQ_MODEL})")
        assistant_text = await _groq_chat(prepared)
    else:
        print(f"[CHAT] Using Ollama ({LLM_MODEL})")
        assistant_text = await _ollama_chat(prepared)

    _append_history(sid, "assistant", assistant_text)
    return assistant_text


async def check_connection() -> bool:
    """Check if LLM backend is reachable."""
    if _use_groq():
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                )
                print(f"Groq connection OK (status={r.status_code})")
                return r.status_code == 200
        except Exception as e:
            print(f"Groq connection failed: {e}")
            return False
    else:
        async with httpx.AsyncClient(timeout=5) as client:
            try:
                await client.get(f"{OLLAMA_URL}/api/tags")
                return True
            except Exception as e:
                print(f"Ollama connection failed: {e}")
                return False