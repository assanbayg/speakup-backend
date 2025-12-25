from fastapi import APIRouter
from fastapi.responses import Response

from services import tts
from config import XTTS_VOICE, XTTS_LANG, TTS_FORMAT

router = APIRouter(tags=["tts"])


@router.get("/speakers")
async def list_speakers():
    """Get available TTS speakers with default indicator."""
    try:
        speakers = tts.list_speakers()
        return {
            "speakers": speakers,
            "default": "default",
        }
    except Exception as e:
        return {"error": str(e), "speakers": [], "default": None}


@router.post("/tts")
async def tts_endpoint(payload: dict):
    """Synthesize speech from text.
    
    Args:
        text: Text to synthesize
        voice: Speaker/voice name (optional, uses default if not provided)
        lang: Language code (default: ru)
        format: Output format - wav or mp3 (default: mp3)
    """
    text = payload.get("text", "")
    if not text or not text.strip():
        return Response(content=b"", status_code=400)
    
    voice = payload.get("voice", XTTS_VOICE)
    lang = payload.get("lang", XTTS_LANG)
    fmt = payload.get("format", TTS_FORMAT)
    
    audio_bytes = tts.synthesize(
        text=text,
        voice=voice,
        lang=lang,
        output_format=fmt,
    )
    
    media_type = "audio/wav" if fmt == "wav" else "audio/mpeg"
    return Response(content=audio_bytes, media_type=media_type)