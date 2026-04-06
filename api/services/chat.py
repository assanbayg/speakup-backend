"""Chat service — Ollama LLM integration."""

from typing import Optional, List, Dict, Any, AsyncIterator
from collections import OrderedDict

import httpx

from config import OLLAMA_URL, LLM_MODEL

# ---------------------------------------------------------------------------
# In-memory conversation history (keyed by session_id)
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
# System prompt — flat, no branching. 1.5B models can't handle conditionals.
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
    """Prepend system prompt to messages."""
    return [{"role": "system", "content": _SYSTEM_PROMPT}] + messages


# ---------------------------------------------------------------------------
# Ollama options
# ---------------------------------------------------------------------------

_OLLAMA_OPTIONS = {
    "num_predict": 80,
    "temperature": 0.7,
    "top_p": 0.9,
    "repeat_penalty": 1.3,
}


async def chat_stream(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[bytes]:
    """Stream chat response from Ollama."""
    prepared = _prepare_messages(messages)
    body = {
        "model": LLM_MODEL,  # always use server config
        "stream": True,
        "messages": prepared,
        "options": _OLLAMA_OPTIONS,
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=body) as r:
            async for chunk in r.aiter_bytes():
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

    body = {
        "model": LLM_MODEL,  # always use server config, ignore client
        "stream": False,
        "messages": prepared,
        "options": _OLLAMA_OPTIONS,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(f"{OLLAMA_URL}/api/chat", json=body)
        response.raise_for_status()
        data = response.json()
        assistant_text = data.get("message", {}).get("content", "")

    _append_history(sid, "assistant", assistant_text)

    return assistant_text


async def check_connection() -> bool:
    """Check if Ollama is reachable."""
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            await client.get(f"{OLLAMA_URL}/api/tags")
            return True
        except Exception as e:
            print(f"Ollama connection failed: {e}")
            return False