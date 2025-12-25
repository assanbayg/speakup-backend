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
DEFAULT_SPEAKER = "Claribel Dervla" 

# Place wav files in api/voices/ directory
CUSTOM_VOICES = {
    "aiym": "aiym.wav", 
}


def get_model() -> TTS:
    """Get TTS model (lazy-loaded singleton)."""
    global _model
    if _model is None:
        _model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    return _model


def list_speakers() -> List[str]:
    """Get available voices (default + custom)."""
    voices = ["default"]
    voices.extend(CUSTOM_VOICES.keys())
    return voices


def synthesize(
    text: str,
    voice: Optional[str] = None,
    lang: str = XTTS_LANG,
    output_format: str = "mp3",
) -> bytes:
    """Synthesize speech from text.
    
    Args:
        text: Text to synthesize
        voice: "default" or custom voice name
        lang: Language code
        output_format: 'wav' or 'mp3'
    
    Returns:
        Audio bytes in requested format
    """
    tts = get_model()
    voice = voice or "default"
    
    if voice == "default":
        # Use XTTS prebuilt speaker
        wav = tts.tts(text=text, language=lang, speaker=DEFAULT_SPEAKER)
    elif voice in CUSTOM_VOICES:
        # Use voice cloning with reference audio
        ref_path = os.path.join(VOICES_DIR, CUSTOM_VOICES[voice])
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Voice file not found: {ref_path}")
        wav = tts.tts(text=text, language=lang, speaker_wav=ref_path)
    else:
        print(f"Unknown voice '{voice}', falling back to default")
        wav = tts.tts(text=text, language=lang, speaker=DEFAULT_SPEAKER)
    
    # Convert to WAV bytes
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


def warmup() -> None:
    """Warm up TTS model with a test synthesis."""
    try:
        synthesize("Привет!", lang=XTTS_LANG)
        print("TTS warmup complete")
    except Exception as e:
        print(f"TTS warmup failed: {e}")