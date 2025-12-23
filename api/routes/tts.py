from fastapi import APIRouter
from fastapi.responses import Response

from services import tts
from config import XTTS_VOICE, XTTS_LANG, TTS_FORMAT

router = APIRouter(tags=["tts"])


@router.get("/speakers")
async def list_speakers():
    """Get available TTS speakers."""
    try:
        speakers = tts.list_speakers()
        return {"speakers": speakers}
    except Exception as e:
        return {"error": str(e), "speakers": []}


@router.post("/tts")
async def tts_endpoint(payload: dict):
    """Synthesize speech from text."""
    text = payload["text"]
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