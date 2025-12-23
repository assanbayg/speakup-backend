import io
from typing import Optional
from dataclasses import dataclass

from pydub import AudioSegment
from transformers import pipeline

from config import WHISPER_MODEL_NAME, TORCH_DEVICE


@dataclass
class TranscriptionResult:
    text: str
    duration: float
    language: str
    avg_confidence: float
    wpm: float
    word_count: int
    clarity_level: str  # "high", "medium", "low"


_model = None


def get_model():
    """Get STT model (lazy-loaded singleton)."""
    global _model
    if _model is None:
        print(f"Loading STT model: {WHISPER_MODEL_NAME}")
        _model = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL_NAME,
            device=0 if TORCH_DEVICE == "cuda" else -1,
        )
    return _model


def _calculate_clarity_level(confidence: float) -> str:
    """Determine clarity level from confidence score."""
    if confidence >= 0.75:
        return "high"
    elif confidence >= 0.5:
        return "medium"
    return "low"


def transcribe(
    audio_bytes: bytes,
    audio_format: Optional[str] = None,
    language: str = "ru",
) -> TranscriptionResult:
    """Transcribe audio to text with confidence metrics.
    
    Args:
        audio_bytes: Raw audio bytes
        audio_format: Audio format hint (wav, mp3, etc.)
        language: Target language code
    
    Returns:
        TranscriptionResult with text and metrics
    """
    # Decode audio
    try:
        if audio_format:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
        else:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    except Exception as e:
        raise ValueError(f"Could not decode audio: {e}")
    
    # Normalize for Whisper: mono, 16kHz
    audio = audio.set_channels(1).set_frame_rate(16000)
    duration_seconds = audio.duration_seconds
    
    # Export to WAV for pipeline
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    buf.seek(0)
    
    # Transcribe
    model = get_model()
    result = model(
        buf.read(),
        return_timestamps="word",
        generate_kwargs={"language": language},
    )
    
    text = result["text"].strip()
    chunks = result.get("chunks", [])
    
    # Extract confidence scores
    confidences = [chunk["score"] for chunk in chunks if "score" in chunk]
    word_count = len(chunks)
    
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
    wpm = (word_count / duration_seconds * 60) if duration_seconds > 0 else 0
    
    return TranscriptionResult(
        text=text,
        duration=duration_seconds,
        language=language,
        avg_confidence=round(avg_confidence, 2),
        wpm=round(wpm, 1),
        word_count=word_count,
        clarity_level=_calculate_clarity_level(avg_confidence),
    )


def warmup() -> None:
    """Warm up STT model."""
    try:
        _ = get_model()
        print("STT warmup complete")
    except Exception as e:
        print(f"STT warmup failed: {e}")