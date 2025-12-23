import torch

# Robust imports: class locations changed across TTS releases
try:
    from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig
except Exception:  # fallback in case of layout differences
    from TTS.tts.configs.xtts_config import XttsConfig

    try:
        from TTS.tts.models.xtts import XttsAudioConfig  # older/newer layouts
    except Exception:
        XttsAudioConfig = type("XttsAudioConfig", (), {})  # harmless stub

import os, io, asyncio
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, Response
import httpx, soundfile as sf
from pydub import AudioSegment
from TTS.api import TTS
from transformers import pipeline
from pydantic import BaseModel
from supabase import create_client, Client

class ChatRequest(BaseModel): 
    message: str
    model: Optional[str] = None
    metrics: Optional[dict] = None
    character: Optional[str] = None
    
class DeleteUserRequest(BaseModel):
    user_id: str
    
OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1b")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "qymyz/whisper-tiny-russian-dysarthria")
TORCH_DEVICE = os.getenv("TORCH_DEVICE", "cpu")
XTTS_LANG = os.getenv("XTTS_LANG", "ru")

XTTS_VOICE = os.getenv("XTTS_VOICE", "Gracie Wise") 
TTS_FORMAT = os.getenv("TTS_FORMAT", "mp3")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

supabase: Optional[Client] = None

if SUPABASE_URL and SUPABASE_SECRET_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
else:
    print("Warning: Supabase keys not found. Auth endpoints will be disabled.")

app = FastAPI(title="SpeakUP API")

tts_model = None
stt_model = None


def get_tts():
    global tts_model
    if tts_model is None:
        tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    return tts_model


def get_stt():
    global stt_model
    if stt_model is None:
        print(f"Loading STT model: {WHISPER_MODEL_NAME}")
        stt_model = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL_NAME,
            device=0 if TORCH_DEVICE == "cuda" else -1,
        )
    return stt_model


def build_system_context(metrics: dict, text: str) -> str:
    """Build adaptive system context based on speech metrics"""
    avg_confidence = metrics.get("avg_confidence", 1.0)
    wpm = metrics.get("wpm", 100)
    clarity_level = metrics.get("clarity_level", "high")
    
    # Base context for all responses - RUSSIAN ONLY
    base = """Ты дружелюбный собеседник для ребёнка с дизартрией (нарушением речи).
ВСЕГДА отвечай ТОЛЬКО на русском языке. Никогда не используй английский.
Отвечай КРАТКО (максимум 1-2 предложения), игриво и ободряюще. Никогда не используй клинические или медицинские термины."""
    
    # Add metrics context
    context = f"{base}\n\nРечь ребёнка: чёткость {avg_confidence:.0%}, {wpm} слов/минуту.\n"
    
    # Adaptive instructions based on clarity
    if clarity_level == "low" or avg_confidence < 0.5:
        context += f"Ребёнок пытался сказать: '{text}'. Начни с подтверждения: 'Я услышал [{text}] - это правильно?' Подожди подтверждения."
    elif clarity_level == "medium" or avg_confidence < 0.75:
        context += f"Ребёнок сказал: '{text}'. Кратко подтверди понимание (например, 'Понял!'), затем ответь естественно."
    else:  # high clarity
        context += f"Ребёнок сказал: '{text}'. Отвечай естественно. Иногда хвали: 'Очень чётко!' или 'Отлично!'"
    
    # Add speech rate context
    if wpm < 60:
        context += " Ребёнок говорит медленно - скажи что-то вроде 'Не торопись, я слушаю!' или похожее."
    elif wpm > 150:
        context += " Ребёнок говорит быстро - это здорово!"
    
    return context


@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/delete-user")
async def delete_user(req: DeleteUserRequest):
    """Sends request to Supabase Admin to delete user"""
    if not supabase:
        raise HTTPException(status_code=503, detail="Service unavailable: Supabase not configured")
    response = supabase.auth.admin.delete_user(req.user_id)
    if response is None:
        raise HTTPException(status_code=500, detail="User doesn't exist")

        
@app.get("/speakers")
async def list_speakers():
    """Get available speakers for debugging"""
    try:
        tts = get_tts()
        if (
            hasattr(tts.synthesizer.tts_model, "speaker_manager")
            and tts.synthesizer.tts_model.speaker_manager
        ):
            speakers = list(tts.synthesizer.tts_model.speaker_manager.speakers.keys())
            return {"speakers": speakers}
        else:
            return {"speakers": [], "note": "No speaker manager found"}
    except Exception as e:
        return {"error": str(e), "speakers": []}


@app.post("/chat")
async def chat(payload: dict):
    model = payload.get("model", LLM_MODEL)
    messages = payload.get("messages", [])
    
    # If metrics provided, prepend system context
    if "metrics" in payload and messages:
        metrics = payload["metrics"]
        user_text = messages[-1].get("content", "")
        system_context = build_system_context(metrics, user_text)
        
        # Prepend system message
        messages = [{"role": "system", "content": system_context}] + messages
    
    body = {"model": model, "stream": True, "messages": messages}

    async def gen():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{OLLAMA}/api/chat", json=body) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/chat/sync")
async def chat_sync(request: ChatRequest):
    """Non-streaming version for mobile clients with metrics support"""
    model = request.model or LLM_MODEL
    messages = [{"role": "user", "content": request.message}]
    
    # Build system context if metrics provided
    if request.metrics:
        system_context = build_system_context(request.metrics, request.message)
        messages = [{"role": "system", "content": system_context}] + messages
    
    body = {"model": model, "stream": False, "messages": messages}
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(f"{OLLAMA}/api/chat", json=body)
            response.raise_for_status()
            data = response.json()
            return {"response": data.get("message", {}).get("content", "")}
        except httpx.HTTPError as e:
            print(f"Ollama API Error: {e}")
            raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")
        except Exception as e:
            print(f"General Error: {e}")
            raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/tts")
async def tts_endpoint(payload: dict):
    text = payload["text"]
    voice = payload.get("voice", XTTS_VOICE)
    lang = payload.get("lang", XTTS_LANG)
    fmt = payload.get("format", TTS_FORMAT)
    tts = get_tts()

    # Handle speaker parameter properly
    kwargs = {"text": text, "language": lang}

    # Only add speaker if it's not None and exists
    if voice is not None:
        # Check if speaker exists
        if (
            hasattr(tts.synthesizer.tts_model, "speaker_manager")
            and tts.synthesizer.tts_model.speaker_manager
        ):
            available_speakers = list(
                tts.synthesizer.tts_model.speaker_manager.speakers.keys()
            )
            if voice in available_speakers:
                kwargs["speaker"] = voice
            else:
                # Log available speakers for debugging
                print(
                    f"Warning: Speaker '{voice}' not found. Available speakers: {available_speakers}"
                )
                # Use first available speaker or skip speaker parameter
                if available_speakers:
                    kwargs["speaker"] = available_speakers[0]

    wav = tts.tts(**kwargs)

    SR = 24000

    buf = io.BytesIO()
    sf.write(buf, wav, samplerate=SR, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    if fmt == "wav":
        return Response(content=wav_bytes, media_type="audio/wav")

    audio = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
    audio = audio.set_frame_rate(SR)
    out = io.BytesIO()
    audio.export(out, format="mp3", parameters=["-q:a", "3", "-ar", str(SR)])
    return Response(content=out.getvalue(), media_type="audio/mpeg")


def _guess_format_from_ct(ct: Optional[str]) -> Optional[str]:
    if not ct:
        return None
    ct = ct.lower()
    if "wav" in ct:
        return "wav"
    if "mpeg" in ct or "mp3" in ct:
        return "mp3"
    if "ogg" in ct:
        return "ogg"
    if "webm" in ct:
        return "webm"
    if "aac" in ct or "mp4" in ct or "m4a" in ct:
        return "m4a"
    return None


MAX_BYTES = 15 * 1024 * 1024  # 15 MB
MAX_SECONDS = 25  # keep kid turns short


@app.post("/stt")
async def stt_endpoint(file: UploadFile = File(...), language: Optional[str] = "ru"):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Audio too large")

    src_fmt = _guess_format_from_ct(file.content_type)
    try:
        audio = (
            AudioSegment.from_file(io.BytesIO(data), format=src_fmt)
            if src_fmt
            else AudioSegment.from_file(io.BytesIO(data))
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Could not decode audio ({file.content_type}). {e}"
        )

    # normalize for Whisper
    audio = audio.set_channels(1).set_frame_rate(16000)

    # fast duration check before export
    duration_seconds = audio.duration_seconds
    if duration_seconds > MAX_SECONDS:
        raise HTTPException(
            status_code=400, detail=f"Audio too long ({duration_seconds:.1f}s)"
        )

    buf = io.BytesIO()
    audio.export(buf, format="wav")
    buf.seek(0)

    # Transcribe with transformers pipeline
    model = get_stt()
    
    try:
        # Use pipeline with return_timestamps to get confidence scores
        result = model(
            buf.read(),
            return_timestamps="word",
            generate_kwargs={"language": language}
        )
        
        text = result["text"].strip()
        
        # Extract confidence scores from chunks
        chunks = result.get("chunks", [])
        confidences = []
        word_count = 0
        
        for chunk in chunks:
            if "score" in chunk:  # transformers uses 'score' for confidence
                confidences.append(chunk["score"])
            word_count += 1
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Calculate WPM
        wpm = (word_count / duration_seconds * 60) if duration_seconds > 0 else 0
        
        # Determine clarity level
        if avg_confidence >= 0.75:
            clarity_level = "high"
        elif avg_confidence >= 0.5:
            clarity_level = "medium"
        else:
            clarity_level = "low"
        
        return JSONResponse({
            "text": text,
            "duration": duration_seconds,
            "language": language,
            "metrics": {
                "avg_confidence": round(avg_confidence, 2),
                "wpm": round(wpm, 1),
                "word_count": word_count,
                "clarity_level": clarity_level
            }
        })
        
    except Exception as e:
        print(f"STT Error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.on_event("startup")
async def warmup():
    async def _w():
        try:
            # Use the same logic as the endpoint for consistency
            tts = get_tts()
            kwargs = {"text": "Привет!", "language": XTTS_LANG}
            if XTTS_VOICE is not None:
                if (
                    hasattr(tts.synthesizer.tts_model, "speaker_manager")
                    and tts.synthesizer.tts_model.speaker_manager
                ):
                    available_speakers = list(
                        tts.synthesizer.tts_model.speaker_manager.speakers.keys()
                    )
                    if XTTS_VOICE in available_speakers:
                        kwargs["speaker"] = XTTS_VOICE
                    elif available_speakers:
                        kwargs["speaker"] = available_speakers[0]
            _ = tts.tts(**kwargs)
        except Exception as e:
            print(f"TTS warmup failed: {e}")

        try:
            _ = get_stt()
        except Exception as e:
            print(f"STT warmup failed: {e}")

        async with httpx.AsyncClient(timeout=5) as c:
            try:
                await c.get(f"{OLLAMA}/api/tags")
            except Exception as e:
                print(f"Ollama connection failed: {e}")

    asyncio.create_task(_w())