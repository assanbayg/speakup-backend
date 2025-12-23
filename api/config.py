import os

# Ollama / LLM
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1b")

# Whisper / STT
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "qymyz/whisper-tiny-russian-dysarthria")

# TTS
XTTS_LANG = os.getenv("XTTS_LANG", "ru")
XTTS_VOICE = os.getenv("XTTS_VOICE", "Gracie Wise")
TTS_FORMAT = os.getenv("TTS_FORMAT", "mp3")

# Runtime
TORCH_DEVICE = os.getenv("TORCH_DEVICE", "cpu")

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

# Limits
MAX_AUDIO_BYTES = 15 * 1024 * 1024  # 15 MB
MAX_AUDIO_SECONDS = 25  # keep kid turns short