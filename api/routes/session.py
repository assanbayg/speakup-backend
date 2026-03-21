from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import session

router = APIRouter(tags=["session"], prefix="/session")


class StartSessionRequest(BaseModel):
    user_id: str


class EndSessionRequest(BaseModel):
    session_id: str
    message_count: int


@router.post("/start")
async def start_session_endpoint(req: StartSessionRequest):
    """Start a new practice session. Returns session_id."""
    session_id = session.start_session(req.user_id)
    if not session_id:
        raise HTTPException(status_code=503, detail="Session service unavailable")
    return {"session_id": session_id}


@router.post("/end")
async def end_session_endpoint(req: EndSessionRequest):
    """End a session with message count."""
    if req.message_count < 0:
        raise HTTPException(status_code=400, detail="message_count must be >= 0")
    ok = session.end_session(req.session_id, req.message_count)
    if not ok:
        raise HTTPException(
            status_code=404,
            detail="Session not found or already ended",
        )
    return {"ok": True}


@router.get("/progress/{user_id}")
async def get_progress_endpoint(user_id: str):
    """Get aggregated progress stats for a user."""
    return session.get_progress(user_id)