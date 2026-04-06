from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx

from services import chat
from config import LLM_MODEL

router = APIRouter(tags=["chat"])


class ChatSyncRequest(BaseModel):
    message: str
    model: Optional[str] = None      # accepted but ignored
    metrics: Optional[dict] = None
    character: Optional[str] = None
    session_id: Optional[str] = None  # optional, backward compatible


@router.post("/chat")
async def chat_stream_endpoint(payload: dict):
    """Streaming chat with Ollama."""
    messages = payload.get("messages", [])
    # model from client is intentionally ignored

    async def generate():
        async for chunk in chat.chat_stream(messages):
            yield chunk

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.post("/chat/sync")
async def chat_sync_endpoint(request: ChatSyncRequest):
    """Non-streaming chat for mobile clients."""
    try:
        response = await chat.chat_sync(
            message=request.message,
            metrics=request.metrics,
            session_id=request.session_id,
        )
        return {"response": response}
    except httpx.HTTPError as e:
        print(f"Ollama API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")
    except Exception as e:
        print(f"General Error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")