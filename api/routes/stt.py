from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from services import stt
from utils import guess_audio_format
from config import MAX_AUDIO_BYTES, MAX_AUDIO_SECONDS

router = APIRouter(tags=["stt"])


@router.post("/stt")
async def stt_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = "ru",
):
    """Transcribe audio to text with speech metrics."""
    data = await file.read()
    
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio too large")
    
    audio_format = guess_audio_format(file.content_type)
    
    try:
        result = stt.transcribe(
            audio_bytes=data,
            audio_format=audio_format,
            language=language,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode audio ({file.content_type}). {e}",
        )
    except Exception as e:
        print(f"STT Error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    if result.duration > MAX_AUDIO_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Audio too long ({result.duration:.1f}s)",
        )
    
    return JSONResponse({
        "text": result.text,
        "duration": result.duration,
        "language": result.language,
        "metrics": {
            "avg_confidence": result.avg_confidence,
            "wpm": result.wpm,
            "word_count": result.word_count,
            "clarity_level": result.clarity_level,
        },
    })