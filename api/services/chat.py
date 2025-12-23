from typing import Optional, List, Dict, Any, AsyncIterator

import httpx

from config import OLLAMA_URL, LLM_MODEL


def build_system_context(metrics: Dict[str, Any], text: str) -> str:
    """Build adaptive system context based on speech metrics.
    
    Args:
        metrics: Dict with avg_confidence, wpm, clarity_level
        text: The transcribed user text
    
    Returns:
        System prompt tailored to child's speech clarity
    """
    avg_confidence = metrics.get("avg_confidence", 1.0)
    wpm = metrics.get("wpm", 100)
    clarity_level = metrics.get("clarity_level", "high")
    
    # Base context - RUSSIAN ONLY
    base = """Ты дружелюбный собеседник для ребёнка с дизартрией (нарушением речи).
ВСЕГДА отвечай ТОЛЬКО на русском языке. Никогда не используй английский.
Отвечай КРАТКО (максимум 1-2 предложения), игриво и ободряюще. Никогда не используй клинические или медицинские термины."""
    
    context = f"{base}\n\nРечь ребёнка: чёткость {avg_confidence:.0%}, {wpm} слов/минуту.\n"
    
    # Adaptive instructions based on clarity
    if clarity_level == "low" or avg_confidence < 0.5:
        context += f"Ребёнок пытался сказать: '{text}'. Начни с подтверждения: 'Я услышал [{text}] - это правильно?' Подожди подтверждения."
    elif clarity_level == "medium" or avg_confidence < 0.75:
        context += f"Ребёнок сказал: '{text}'. Кратко подтверди понимание (например, 'Понял!'), затем ответь естественно."
    else:
        context += f"Ребёнок сказал: '{text}'. Отвечай естественно. Иногда хвали: 'Очень чётко!' или 'Отлично!'"
    
    # Speech rate context
    if wpm < 60:
        context += " Ребёнок говорит медленно - скажи что-то вроде 'Не торопись, я слушаю!' или похожее."
    elif wpm > 150:
        context += " Ребёнок говорит быстро - это здорово!"
    
    return context


def _prepare_messages(
    messages: List[Dict[str, str]],
    metrics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """Prepare messages with optional system context from metrics."""
    if not metrics or not messages:
        return messages
    
    user_text = messages[-1].get("content", "")
    system_context = build_system_context(metrics, user_text)
    return [{"role": "system", "content": system_context}] + messages


async def chat_stream(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[bytes]:
    """Stream chat response from Ollama.
    
    Args:
        messages: List of message dicts with role and content
        model: Model name (defaults to LLM_MODEL)
        metrics: Optional speech metrics for context
    
    Yields:
        Response chunks as bytes
    """
    prepared = _prepare_messages(messages, metrics)
    body = {
        "model": model or LLM_MODEL,
        "stream": True,
        "messages": prepared,
    }
    
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=body) as r:
            async for chunk in r.aiter_bytes():
                yield chunk


async def chat_sync(
    message: str,
    model: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Get non-streaming chat response from Ollama.
    
    Args:
        message: User message
        model: Model name (defaults to LLM_MODEL)
        metrics: Optional speech metrics for context
    
    Returns:
        Assistant response text
    """
    messages = [{"role": "user", "content": message}]
    prepared = _prepare_messages(messages, metrics)
    
    body = {
        "model": model or LLM_MODEL,
        "stream": False,
        "messages": prepared,
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(f"{OLLAMA_URL}/api/chat", json=body)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")


async def check_connection() -> bool:
    """Check if Ollama is reachable."""
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            await client.get(f"{OLLAMA_URL}/api/tags")
            return True
        except Exception as e:
            print(f"Ollama connection failed: {e}")
            return False