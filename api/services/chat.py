"""Chat service — Ollama LLM integration with adaptive speech context."""

from typing import Optional, List, Dict, Any, AsyncIterator
from collections import OrderedDict
import time

import httpx

from config import OLLAMA_URL, LLM_MODEL

# ---------------------------------------------------------------------------
# In-memory conversation history (keyed by session_id or fallback key)
# ---------------------------------------------------------------------------
# LRU-style: max 200 sessions, each stores last 20 messages.
# This is volatile (lost on restart) but good enough until DB-backed history.

MAX_SESSIONS = 200
MAX_MESSAGES_PER_SESSION = 20

_history: OrderedDict[str, List[Dict[str, str]]] = OrderedDict()


def _get_history(session_id: str) -> List[Dict[str, str]]:
    """Get or create conversation history for a session."""
    if session_id in _history:
        _history.move_to_end(session_id)
        return _history[session_id]
    if len(_history) >= MAX_SESSIONS:
        _history.popitem(last=False)  # evict oldest
    _history[session_id] = []
    return _history[session_id]


def _append_history(session_id: str, role: str, content: str) -> None:
    """Append a message and trim to max length."""
    history = _get_history(session_id)
    history.append({"role": role, "content": content})
    # Keep only the last N messages
    if len(history) > MAX_MESSAGES_PER_SESSION:
        del history[: len(history) - MAX_MESSAGES_PER_SESSION]


def clear_history(session_id: str) -> None:
    """Clear history for a session (call on session end)."""
    _history.pop(session_id, None)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """Ты — Спичи, весёлый и любопытный друг-собеседник для ребёнка.

ПРАВИЛА (строго соблюдай):
• Отвечай ТОЛЬКО на русском языке. НИКОГДА не используй английский, даже отдельные слова. Не спрашивай про язык.
• Отвечай КРАТКО: 1–2 предложения максимум.
• Говори просто, как с ребёнком 5–8 лет. Без сложных слов, без медицинских терминов.
• Будь игривым, тёплым, ободряющим. Используй восклицания, смайлики в словах (ура, ого, вау).
• ВСЕГДА заканчивай свой ответ вопросом или предложением продолжить разговор. Примеры: «А ты?», «Расскажи ещё!», «А что было дальше?», «Хочешь поиграем в слова?»
• Если сообщение ребёнка непонятное или бессмысленное — НЕ повторяй эти слова. Вместо этого мягко переспроси: «Ой, я не расслышал! Скажи ещё разок?» или предложи тему: «Давай поговорим о животных! Какое твоё любимое?»
• Никогда не повторяй и не цитируй то, что сказал ребёнок, дословно.
• Не упоминай речевые нарушения, дизартрию, логопедию, терапию.

СТИЛЬ:
Ты как мультяшный персонаж — добрый, чуть смешной, всегда рад поговорить."""


def build_system_context(metrics: Dict[str, Any]) -> str:
    """Build system prompt with adaptive hints based on speech metrics.

    The key change from the old version: we NO LONGER embed the raw
    transcribed text into the system prompt.  The user text travels in
    the normal user message — the system prompt only carries behavioural
    instructions and gentle adaptation hints.
    """
    avg_confidence = metrics.get("avg_confidence", 1.0)
    clarity_level = metrics.get("clarity_level", "high")
    wpm = metrics.get("wpm", 100)

    parts = [_SYSTEM_PROMPT]

    # Adaptive behaviour hints (no raw text!)
    if clarity_level == "low" or avg_confidence < 0.5:
        parts.append(
            "\nСейчас речь ребёнка нечёткая. Если сообщение непонятно — "
            "ласково переспроси или предложи другую тему. Не притворяйся, "
            "что понял, если не понял."
        )
    elif clarity_level == "medium" or avg_confidence < 0.75:
        parts.append(
            "\nРечь ребёнка средней чёткости. Кратко подтверди, что понял, "
            "и продолжай разговор."
        )
    else:
        parts.append(
            "\nРебёнок говорит чётко. Иногда хвали: «Как здорово сказал!» "
            "или «Отлично!»"
        )

    if wpm and wpm < 60:
        parts.append("Ребёнок говорит медленно — не торопи, подбодри.")

    return "\n".join(parts)


def _prepare_messages(
    messages: List[Dict[str, str]],
    metrics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """Prepend system prompt (with optional metric hints) to messages."""
    system_content = build_system_context(metrics) if metrics else _SYSTEM_PROMPT
    return [{"role": "system", "content": system_content}] + messages


# ---------------------------------------------------------------------------
# Ollama options — keep responses short to cut latency on CPU
# ---------------------------------------------------------------------------

_OLLAMA_OPTIONS = {
    "num_predict": 80,       # hard cap on tokens → faster on CPU
    "temperature": 0.7,      # some creativity but not wild
    "top_p": 0.9,
    "repeat_penalty": 1.3,   # discourage echoing / repetition
}


async def chat_stream(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[bytes]:
    """Stream chat response from Ollama."""
    prepared = _prepare_messages(messages, metrics)
    body = {
        "model": model or LLM_MODEL,
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
    """Non-streaming chat with optional conversation memory.

    If session_id is provided, conversation history is maintained
    across calls.  The API contract is unchanged — session_id is
    an optional new field that old clients simply won't send.
    """
    sid = session_id or "_default"

    # Append the new user message to history
    _append_history(sid, "user", message)

    # Build full message list from history
    history = _get_history(sid)
    prepared = _prepare_messages(list(history), metrics)

    body = {
        "model": model or LLM_MODEL,
        "stream": False,
        "messages": prepared,
        "options": _OLLAMA_OPTIONS,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(f"{OLLAMA_URL}/api/chat", json=body)
        response.raise_for_status()
        data = response.json()
        assistant_text = data.get("message", {}).get("content", "")

    # Store assistant reply in history
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