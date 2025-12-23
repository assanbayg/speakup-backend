import io
from typing import Optional, List

import soundfile as sf
from pydub import AudioSegment

# Robust imports: class locations changed across TTS releases
try:
    from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig
except Exception:
    from TTS.tts.configs.xtts_config import XttsConfig
    try:
        from TTS.tts.models.xtts import XttsAudioConfig
    except Exception:
        XttsAudioConfig = type("XttsAudioConfig", (), {})  # harmless stub

from TTS.api import TTS

from config import XTTS_LANG, XTTS_VOICE

_model: Optional[TTS] = None
SAMPLE_RATE = 24000


def get_model() -> TTS:
    """Get TTS model (lazy-loaded singleton)."""
    global _model
    if _model is None:
        _model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    return _model


def list_speakers() -> List[str]:
    """Get available speaker names."""
    tts = get_model()
    if (
        hasattr(tts.synthesizer.tts_model, "speaker_manager")
        and tts.synthesizer.tts_model.speaker_manager
    ):
        return list(tts.synthesizer.tts_model.speaker_manager.speakers.keys())
    return []


def _resolve_speaker(voice: Optional[str]) -> Optional[str]:
    """Resolve speaker name, falling back to first available if not found."""
    if voice is None:
        return None
    
    available = list_speakers()
    if not available:
        return None
    
    if voice in available:
        return voice
    
    print(f"Warning: Speaker '{voice}' not found. Available: {available}")
    return available[0] if available else None


def synthesize(
    text: str,
    voice: Optional[str] = None,
    lang: str = XTTS_LANG,
    output_format: str = "mp3",
) -> bytes:
    """Synthesize speech from text.
    
    Args:
        text: Text to synthesize
        voice: Speaker name (optional)
        lang: Language code
        output_format: 'wav' or 'mp3'
    
    Returns:
        Audio bytes in requested format
    """
    tts = get_model()
    
    kwargs = {"text": text, "language": lang}
    speaker = _resolve_speaker(voice or XTTS_VOICE)
    if speaker:
        kwargs["speaker"] = speaker
    
    wav = tts.tts(**kwargs)
    
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