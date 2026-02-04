import io
import os
from typing import Optional, List

import soundfile as sf
from pydub import AudioSegment
from TTS.api import TTS

from config import XTTS_LANG

_model: Optional[TTS] = None
SAMPLE_RATE = 24000

# Voice configuration
VOICES_DIR = os.path.join(os.path.dirname(__file__), "..", "voices")
USER_VOICES_DIR = os.path.join(VOICES_DIR, "users")
DEFAULT_SPEAKER = "Claribel Dervla" 

# Place wav files in api/voices/ directory
CUSTOM_VOICES = {
    "aiym": "aiym.wav", 
}

# Ensure directories exist
os.makedirs(USER_VOICES_DIR, exist_ok=True)


def get_model() -> TTS:
    """Get TTS model (lazy-loaded singleton)."""
    global _model
    if _model is None:
        _model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    return _model


def list_speakers() -> List[str]:
    """Get available voices (default + custom + user voices)."""
    voices = ["default"]
    voices.extend(CUSTOM_VOICES.keys())
    return voices


def get_user_voice_path(user_id: str, voice_id: str) -> Optional[str]:
    """Get path to a user's cloned voice file."""
    path = os.path.join(USER_VOICES_DIR, user_id, f"{voice_id}.wav")
    if os.path.exists(path):
        return path
    return None


def get_user_default_voice_path(user_id: str) -> Optional[str]:
    """Get path to user's default cloned voice."""
    default_file = os.path.join(USER_VOICES_DIR, user_id, "default.txt")
    if os.path.exists(default_file):
        with open(default_file) as f:
            voice_id = f.read().strip()
        return get_user_voice_path(user_id, voice_id)
    return None


def _wav_to_bytes(wav: list, output_format: str = "mp3") -> bytes:
    """Convert wav array to bytes in requested format."""
    buf = io.BytesIO()
    sf.write(buf, wav, samplerate=SAMPLE_RATE, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()
    
    if output_format == "wav":
        return wav_bytes
    
    # Convert to MP3
    audio = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
    audio = audio.set_frame_rate(SAMPLE_RATE)
    out = io.BytesIO()
    audio.export(out, format="mp3", parameters=["-q:a", "3", "-ar", str(SAMPLE_RATE)])
    return out.getvalue()


def synthesize_with_reference(
    text: str,
    speaker_wav: str,
    lang: str = XTTS_LANG,
    output_format: str = "mp3",
) -> bytes:
    """Synthesize speech using a reference audio file for voice cloning.
    
    Args:
        text: Text to synthesize
        speaker_wav: Path to reference audio file
        lang: Language code
        output_format: 'wav' or 'mp3'
    
    Returns:
        Audio bytes in requested format
    """
    if not os.path.exists(speaker_wav):
        raise FileNotFoundError(f"Reference audio not found: {speaker_wav}")
    
    tts = get_model()
    wav = tts.tts(text=text, language=lang, speaker_wav=speaker_wav)
    return _wav_to_bytes(wav, output_format)


def synthesize(
    text: str,
    voice: Optional[str] = None,
    lang: str = XTTS_LANG,
    output_format: str = "mp3",
    user_id: Optional[str] = None,
) -> bytes:
    """Synthesize speech from text.
    
    Args:
        text: Text to synthesize
        voice: "default", custom voice name, or voice_id (if user_id provided)
        lang: Language code
        output_format: 'wav' or 'mp3'
        user_id: Optional user ID to check for custom cloned voices
    
    Returns:
        Audio bytes in requested format
    """
    tts = get_model()
    voice = voice or "default"
    
    # Check if it's a user's custom cloned voice
    if user_id:
        # First check if voice is a specific voice_id
        user_voice_path = get_user_voice_path(user_id, voice)
        if user_voice_path:
            wav = tts.tts(text=text, language=lang, speaker_wav=user_voice_path)
            return _wav_to_bytes(wav, output_format)
        
        # Check for user's default voice if requesting "parent" or "user"
        if voice.lower() in ("parent", "user", "my_voice"):
            default_path = get_user_default_voice_path(user_id)
            if default_path:
                wav = tts.tts(text=text, language=lang, speaker_wav=default_path)
                return _wav_to_bytes(wav, output_format)
    
    # Check predefined custom voices
    if voice == "default":
        wav = tts.tts(text=text, language=lang, speaker=DEFAULT_SPEAKER)
    elif voice in CUSTOM_VOICES:
        ref_path = os.path.join(VOICES_DIR, CUSTOM_VOICES[voice])
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Voice file not found: {ref_path}")
        wav = tts.tts(text=text, language=lang, speaker_wav=ref_path)
    else:
        print(f"Unknown voice '{voice}', falling back to default")
        wav = tts.tts(text=text, language=lang, speaker=DEFAULT_SPEAKER)
    
    return _wav_to_bytes(wav, output_format)


def warmup() -> None:
    """Warm up TTS model with a test synthesis."""
    try:
        synthesize("Привет!", lang=XTTS_LANG)
        print("TTS warmup complete")
    except Exception as e:
        print(f"TTS warmup failed: {e}")