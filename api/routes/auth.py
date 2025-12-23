from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import supabase

router = APIRouter(tags=["auth"])


class DeleteUserRequest(BaseModel):
    user_id: str


@router.post("/delete-user")
async def delete_user(req: DeleteUserRequest):
    """Delete a user via Supabase Admin API."""
    client = supabase.get_supabase()
    if not client:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Supabase not configured",
        )
    
    response = client.auth.admin.delete_user(req.user_id)
    if response is None:
        raise HTTPException(status_code=500, detail="User doesn't exist")
    
    return {"ok": True}