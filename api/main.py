import asyncio

from fastapi import FastAPI

from routes import auth, chat, stt, tts, sprites 
from services import supabase
from services import tts as tts_service
from services import stt as stt_service
from services import chat as chat_service

app = FastAPI(title="SpeakUP API")

# Register routes
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(stt.router)
app.include_router(tts.router)
app.include_router(sprites.router)  


@app.get("/health")
async def health():
    return {"ok": True}


@app.on_event("startup")
async def startup():
    """Warm up models and check connections on startup."""
    
    async def warmup_task():
        # Log Supabase status
        if supabase.is_configured():
            print("Supabase configured")
        else:
            print("Warning: Supabase not configured. Auth endpoints disabled.")
        
        # Warm up TTS
        tts_service.warmup()
        
        # Warm up STT
        stt_service.warmup()
        
        # Check Ollama connection
        await chat_service.check_connection()
    
    asyncio.create_task(warmup_task())