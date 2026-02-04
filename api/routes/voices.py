from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from services import voice_cloning
from services import tts as tts_service
from utils import guess_audio_format
from config import MAX_AUDIO_BYTES

router = APIRouter(tags=["voices"], prefix="/voices")


class PreviewRequest(BaseModel):
    text: str
    voice_id: str
    user_id: str


# === VALIDATION ===

@router.post("/validate")
async def validate_voice_sample(
    file: UploadFile = File(...),
):
    """Validate audio file before upload.
    
    Use this to give user feedback before processing.
    Returns validation status, warnings, and recommendations.
    """
    data = await file.read()
    
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")
    
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large ({len(data) / 1024 / 1024:.1f}MB). Max: 15MB"
        )
    
    audio_format = guess_audio_format(file.content_type)
    result = voice_cloning.validate_audio(data, audio_format)
    
    return JSONResponse({
        "valid": result.valid,
        "duration": result.duration,
        "sample_rate": result.sample_rate,
        "channels": result.channels,
        "errors": result.errors,
        "warnings": result.warnings,
        "recommendations": _get_recommendations(result),
    })


def _get_recommendations(result: voice_cloning.VoiceValidationResult) -> list[str]:
    """Generate user-friendly recommendations."""
    recs = []
    
    if result.duration < 15:
        recs.append("Record at least 15-30 seconds for best voice quality")
    
    if result.sample_rate < 22050:
        recs.append("Use higher quality recording settings if possible")
    
    if not result.errors and not result.warnings:
        recs.append("Audio looks good! Ready to upload.")
    
    return recs


# === UPLOAD & PROCESSING ===

@router.post("/upload")
async def upload_voice(
    user_id: str = Query(..., description="User ID"),
    voice_name: Optional[str] = Query(None, description="Custom name for this voice"),
    file: UploadFile = File(...),
):
    """Upload and process a voice sample for cloning.
    
    The audio will be:
    1. Validated for duration and quality
    2. Preprocessed (noise reduction, normalization)
    3. Saved for use with TTS
    
    Optimal audio: 15-30 seconds of clear speech, minimal background noise.
    """
    data = await file.read()
    
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")
    
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(data) / 1024 / 1024:.1f}MB). Max: 15MB"
        )
    
    audio_format = guess_audio_format(file.content_type)
    
    try:
        result = voice_cloning.save_voice(
            user_id=user_id,
            audio_bytes=data,
            audio_format=audio_format,
            voice_name=voice_name,
        )
        
        return JSONResponse({
            "ok": True,
            "voice_id": result.voice_id,
            "duration": result.duration,
            "message": "Voice processed and saved successfully",
            "metadata": result.metadata,
        })
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Voice upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# === VOICE MANAGEMENT ===

@router.get("/list")
async def list_voices(
    user_id: str = Query(..., description="User ID"),
):
    """List all voices for a user."""
    voices = voice_cloning.get_user_voices(user_id)
    default_voice = voice_cloning.get_default_voice(user_id)
    
    return JSONResponse({
        "user_id": user_id,
        "voices": voices,
        "default_voice_id": default_voice,
        "count": len(voices),
    })


@router.post("/set-default")
async def set_default_voice(
    user_id: str = Query(..., description="User ID"),
    voice_id: str = Query(..., description="Voice ID to set as default"),
):
    """Set a voice as the user's default for TTS."""
    if voice_cloning.set_default_voice(user_id, voice_id):
        return JSONResponse({
            "ok": True,
            "message": f"Voice {voice_id} set as default",
        })
    else:
        raise HTTPException(status_code=404, detail="Voice not found")


@router.delete("/{user_id}/{voice_id}")
async def delete_voice(
    user_id: str,
    voice_id: str,
):
    """Delete a voice."""
    if voice_cloning.delete_voice(user_id, voice_id):
        return JSONResponse({"ok": True, "message": "Voice deleted"})
    else:
        raise HTTPException(status_code=404, detail="Voice not found")


# === PREVIEW ===

@router.post("/preview")
async def preview_voice(
    request: PreviewRequest,
):
    """Preview TTS with a cloned voice.
    
    Use this to let users hear their cloned voice before confirming.
    """
    filepath = voice_cloning.get_voice_filepath(request.user_id, request.voice_id)
    if not filepath:
        raise HTTPException(status_code=404, detail="Voice not found")
    
    try:
        # Use the cloned voice for TTS
        audio_bytes = tts_service.synthesize_with_reference(
            text=request.text,
            speaker_wav=filepath,
        )
        
        return Response(content=audio_bytes, media_type="audio/mpeg")
    
    except Exception as e:
        print(f"Voice preview error: {e}")
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")


@router.get("/preview/{user_id}/{voice_id}")
async def preview_voice_get(
    user_id: str,
    voice_id: str,
    text: str = Query("Привет! Это мой голос.", description="Text to synthesize"),
):
    """Preview TTS with a cloned voice (GET for easy testing)."""
    filepath = voice_cloning.get_voice_filepath(user_id, voice_id)
    if not filepath:
        raise HTTPException(status_code=404, detail="Voice not found")
    
    try:
        audio_bytes = tts_service.synthesize_with_reference(
            text=text,
            speaker_wav=filepath,
        )
        
        return Response(content=audio_bytes, media_type="audio/mpeg")
    
    except Exception as e:
        print(f"Voice preview error: {e}")
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")