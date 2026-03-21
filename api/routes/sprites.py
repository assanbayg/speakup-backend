from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, Response, RedirectResponse
from pydantic import BaseModel

from services.sprites import get_storage

router = APIRouter(tags=["sprites"], prefix="/sprites")


class ApproveSpriteRequest(BaseModel):
    user_id: str
    sprite_name: str


# === USER ENDPOINTS ===

@router.post("/upload-pending")
async def upload_pending_sprite(
    user_id: str = Query(..., description="User uploading the sprite"),
    file: UploadFile = File(...),
):
    """Kid uploads their drawing for review."""
    storage = get_storage()
    try:
        image_data = await file.read()
        filename = storage.save_pending(
            user_id=user_id,
            image_data=image_data,
            content_type=file.content_type,
            original_filename=file.filename,
        )
        return JSONResponse({"ok": True, "message": "Sprite uploaded for review", "filename": filename})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Pending upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/list")
async def list_user_sprites(
    user_id: str = Query(..., description="User ID"),
):
    """List approved sprites available for this user."""
    storage = get_storage()
    sprites = storage.list_approved(user_id)
    return JSONResponse({"user_id": user_id, "sprites": sprites})


@router.get("/my-pending")
async def list_my_pending_sprites(
    user_id: str = Query(..., description="User ID"),
):
    """List user's own pending sprites awaiting admin review."""
    storage = get_storage()
    pending = storage.list_pending(user_id)
    filenames = pending.get(user_id, [])
    return JSONResponse({"user_id": user_id, "pending": filenames})


@router.get("/image/{user_id}/{filename}")
async def get_sprite_image(user_id: str, filename: str):
    """Serve approved sprite image."""
    storage = get_storage()
    url = storage.get_sprite_url(user_id, filename, pending=False)
    if url:
        return RedirectResponse(url=url)
    image_bytes = storage.get_sprite_bytes(user_id, filename, pending=False)
    if not image_bytes:
        raise HTTPException(status_code=404, detail="Sprite not found")
    media_type = "image/png"
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        media_type = "image/jpeg"
    elif filename.endswith(".webp"):
        media_type = "image/webp"
    return Response(content=image_bytes, media_type=media_type)


# === ADMIN ENDPOINTS ===

@router.get("/admin/pending")
async def list_pending_sprites(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
):
    """[ADMIN] List all pending sprite uploads."""
    storage = get_storage()
    pending = storage.list_pending(user_id)
    return JSONResponse({"pending": pending})


@router.get("/admin/pending/{user_id}/{filename}")
async def get_pending_sprite_image(user_id: str, filename: str):
    """[ADMIN] View pending sprite image for review."""
    storage = get_storage()
    url = storage.get_sprite_url(user_id, filename, pending=True)
    if url:
        return RedirectResponse(url=url)
    image_bytes = storage.get_sprite_bytes(user_id, filename, pending=True)
    if not image_bytes:
        raise HTTPException(status_code=404, detail="Pending sprite not found")
    media_type = "image/png"
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        media_type = "image/jpeg"
    elif filename.endswith(".webp"):
        media_type = "image/webp"
    return Response(content=image_bytes, media_type=media_type)


@router.post("/admin/approve")
async def approve_sprite(
    user_id: str = Query(...),
    sprite_name: str = Query(...),
    file: UploadFile = File(...),
):
    """[ADMIN] Upload approved sprite for a user."""
    storage = get_storage()
    try:
        image_data = await file.read()
        filename = storage.approve_sprite(
            user_id=user_id,
            image_data=image_data,
            content_type=file.content_type,
            sprite_name=sprite_name,
        )
        return JSONResponse({"ok": True, "message": f"Sprite approved for user {user_id}", "filename": filename})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Approve sprite error: {e}")
        raise HTTPException(status_code=500, detail=f"Approval failed: {str(e)}")


@router.delete("/admin/pending/{user_id}/{filename}")
async def delete_pending_sprite(user_id: str, filename: str):
    """[ADMIN] Delete a pending sprite after review."""
    storage = get_storage()
    if storage.delete_pending(user_id, filename):
        return JSONResponse({"ok": True, "message": "Pending sprite deleted"})
    else:
        raise HTTPException(status_code=404, detail="Pending sprite not found")