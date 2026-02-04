import io
import os
import hashlib
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import json

import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize, high_pass_filter, low_pass_filter

from config import TORCH_DEVICE


@dataclass
class VoiceValidationResult:
    valid: bool
    duration: float
    sample_rate: int
    channels: int
    errors: list[str]
    warnings: list[str]


@dataclass 
class ProcessedVoice:
    user_id: str
    voice_id: str
    filepath: str
    duration: float
    created_at: str
    metadata: Dict[str, Any]


# Voice storage config
VOICES_BASE_DIR = os.environ.get("VOICES_DIR", "/app/voices")
USER_VOICES_DIR = os.path.join(VOICES_BASE_DIR, "users")

# Validation limits
MIN_DURATION_SECONDS = 6      # XTTS needs at least 6s
MAX_DURATION_SECONDS = 60     # Cap at 1 minute
OPTIMAL_DURATION = (15, 30)   # Sweet spot for XTTS
MIN_SAMPLE_RATE = 16000       # Minimum acceptable
TARGET_SAMPLE_RATE = 22050    # XTTS optimal


def _ensure_dirs():
    """Ensure voice directories exist."""
    os.makedirs(USER_VOICES_DIR, exist_ok=True)


def _get_user_voice_dir(user_id: str) -> str:
    """Get directory for user's voice files."""
    user_dir = os.path.join(USER_VOICES_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def _generate_voice_id(user_id: str, audio_hash: str) -> str:
    """Generate unique voice ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"voice_{timestamp}_{audio_hash[:8]}"


def validate_audio(
    audio_bytes: bytes,
    audio_format: Optional[str] = None,
) -> VoiceValidationResult:
    """Validate audio file for voice cloning.
    
    Checks:
    - Duration within acceptable range
    - Sample rate sufficient
    - Audio is decodable
    """
    errors = []
    warnings = []
    
    try:
        if audio_format:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
        else:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    except Exception as e:
        return VoiceValidationResult(
            valid=False,
            duration=0,
            sample_rate=0,
            channels=0,
            errors=[f"Could not decode audio: {e}"],
            warnings=[],
        )
    
    duration = audio.duration_seconds
    sample_rate = audio.frame_rate
    channels = audio.channels
    
    # Check duration
    if duration < MIN_DURATION_SECONDS:
        errors.append(
            f"Audio too short ({duration:.1f}s). Need at least {MIN_DURATION_SECONDS}s "
            f"of clear speech for voice cloning."
        )
    elif duration > MAX_DURATION_SECONDS:
        errors.append(
            f"Audio too long ({duration:.1f}s). Maximum is {MAX_DURATION_SECONDS}s. "
            f"Optimal range is {OPTIMAL_DURATION[0]}-{OPTIMAL_DURATION[1]}s."
        )
    
    # Warn if outside optimal range
    if MIN_DURATION_SECONDS <= duration < OPTIMAL_DURATION[0]:
        warnings.append(
            f"Audio is {duration:.1f}s. For best results, use {OPTIMAL_DURATION[0]}-{OPTIMAL_DURATION[1]}s."
        )
    elif OPTIMAL_DURATION[1] < duration <= MAX_DURATION_SECONDS:
        warnings.append(
            f"Audio is {duration:.1f}s. Will be trimmed to {OPTIMAL_DURATION[1]}s for optimal cloning."
        )
    
    # Check sample rate
    if sample_rate < MIN_SAMPLE_RATE:
        errors.append(
            f"Sample rate too low ({sample_rate}Hz). Need at least {MIN_SAMPLE_RATE}Hz."
        )
    elif sample_rate < TARGET_SAMPLE_RATE:
        warnings.append(
            f"Sample rate is {sample_rate}Hz. {TARGET_SAMPLE_RATE}Hz or higher recommended."
        )
    
    # Check for very quiet audio (might indicate silence/noise)
    if audio.dBFS < -40:
        warnings.append(
            "Audio is very quiet. Recording might be mostly silence or very low volume."
        )
    
    return VoiceValidationResult(
        valid=len(errors) == 0,
        duration=duration,
        sample_rate=sample_rate,
        channels=channels,
        errors=errors,
        warnings=warnings,
    )


def preprocess_audio(
    audio_bytes: bytes,
    audio_format: Optional[str] = None,
    trim_to_optimal: bool = True,
) -> Tuple[bytes, Dict[str, Any]]:
    """Preprocess audio for optimal XTTS voice cloning.
    
    Steps:
    1. Decode and normalize format
    2. Convert to mono
    3. Resample to target rate
    4. Apply noise reduction (high-pass filter)
    5. Normalize volume
    6. Trim silence
    7. Trim to optimal duration if needed
    
    Returns:
        Tuple of (processed WAV bytes, preprocessing metadata)
    """
    # Load audio
    if audio_format:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
    else:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    
    original_duration = audio.duration_seconds
    original_sample_rate = audio.frame_rate
    original_channels = audio.channels
    
    # Step 1: Convert to mono
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Step 2: Resample to target rate
    if audio.frame_rate != TARGET_SAMPLE_RATE:
        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
    
    # Step 3: High-pass filter to remove low-frequency noise (rumble, hum)
    # 80Hz cutoff removes most environmental noise while preserving speech
    audio = high_pass_filter(audio, cutoff=80)
    
    # Step 4: Low-pass filter to remove high-frequency noise/hiss
    # 8000Hz cutoff preserves speech clarity (fundamental + harmonics)
    audio = low_pass_filter(audio, cutoff=8000)
    
    # Step 5: Normalize volume to standard level
    audio = normalize(audio, headroom=1.0)
    
    # Step 6: Trim silence from start and end
    # Use -40dB threshold and minimum silence length of 100ms
    audio = _trim_silence(audio, silence_thresh=-40, min_silence_len=100)
    
    # Step 7: Trim to optimal duration if too long
    trimmed = False
    if trim_to_optimal and audio.duration_seconds > OPTIMAL_DURATION[1]:
        # Take middle section (often clearest speech)
        target_ms = OPTIMAL_DURATION[1] * 1000
        total_ms = len(audio)
        start = (total_ms - target_ms) // 2
        audio = audio[start:start + target_ms]
        trimmed = True
    
    # Export to WAV
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    processed_bytes = buf.getvalue()
    
    metadata = {
        "original_duration": original_duration,
        "processed_duration": audio.duration_seconds,
        "original_sample_rate": original_sample_rate,
        "processed_sample_rate": TARGET_SAMPLE_RATE,
        "original_channels": original_channels,
        "was_trimmed": trimmed,
        "final_dbfs": audio.dBFS,
    }
    
    return processed_bytes, metadata


def _trim_silence(
    audio: AudioSegment,
    silence_thresh: int = -40,
    min_silence_len: int = 100,
    keep_silence: int = 50,
) -> AudioSegment:
    """Trim silence from start and end of audio.
    
    Args:
        audio: Audio segment to trim
        silence_thresh: dB threshold for silence detection
        min_silence_len: Minimum length of silence (ms)
        keep_silence: Amount of silence to keep at edges (ms)
    """
    from pydub.silence import detect_leading_silence
    
    # Detect leading silence
    start_trim = detect_leading_silence(audio, silence_threshold=silence_thresh)
    
    # Detect trailing silence (reverse audio)
    end_trim = detect_leading_silence(audio.reverse(), silence_threshold=silence_thresh)
    
    # Apply trim with some padding
    start = max(0, start_trim - keep_silence)
    end = len(audio) - max(0, end_trim - keep_silence)
    
    if end > start:
        return audio[start:end]
    return audio


def save_voice(
    user_id: str,
    audio_bytes: bytes,
    audio_format: Optional[str] = None,
    voice_name: Optional[str] = None,
) -> ProcessedVoice:
    """Process and save voice sample for user.
    
    Args:
        user_id: User ID
        audio_bytes: Raw audio bytes
        audio_format: Audio format hint
        voice_name: Optional custom name for the voice
    
    Returns:
        ProcessedVoice with file info
    
    Raises:
        ValueError: If audio validation fails
    """
    _ensure_dirs()
    
    # Validate first
    validation = validate_audio(audio_bytes, audio_format)
    if not validation.valid:
        raise ValueError("; ".join(validation.errors))
    
    # Preprocess
    processed_bytes, preprocess_meta = preprocess_audio(audio_bytes, audio_format)
    
    # Generate voice ID
    audio_hash = hashlib.md5(audio_bytes).hexdigest()
    voice_id = _generate_voice_id(user_id, audio_hash)
    
    # Save processed audio
    user_dir = _get_user_voice_dir(user_id)
    voice_filepath = os.path.join(user_dir, f"{voice_id}.wav")
    
    with open(voice_filepath, "wb") as f:
        f.write(processed_bytes)
    
    # Save metadata
    created_at = datetime.now().isoformat()
    metadata = {
        "voice_id": voice_id,
        "voice_name": voice_name or "Parent Voice",
        "user_id": user_id,
        "created_at": created_at,
        "preprocessing": preprocess_meta,
        "validation_warnings": validation.warnings,
    }
    
    meta_filepath = os.path.join(user_dir, f"{voice_id}.json")
    with open(meta_filepath, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return ProcessedVoice(
        user_id=user_id,
        voice_id=voice_id,
        filepath=voice_filepath,
        duration=preprocess_meta["processed_duration"],
        created_at=created_at,
        metadata=metadata,
    )


def get_user_voices(user_id: str) -> list[Dict[str, Any]]:
    """Get all voices for a user."""
    user_dir = os.path.join(USER_VOICES_DIR, user_id)
    if not os.path.exists(user_dir):
        return []
    
    voices = []
    for filename in os.listdir(user_dir):
        if filename.endswith(".json"):
            with open(os.path.join(user_dir, filename)) as f:
                metadata = json.load(f)
                voices.append(metadata)
    
    # Sort by created_at descending (newest first)
    voices.sort(key=lambda v: v.get("created_at", ""), reverse=True)
    return voices


def get_voice_filepath(user_id: str, voice_id: str) -> Optional[str]:
    """Get filepath for a specific voice."""
    filepath = os.path.join(USER_VOICES_DIR, user_id, f"{voice_id}.wav")
    if os.path.exists(filepath):
        return filepath
    return None


def delete_voice(user_id: str, voice_id: str) -> bool:
    """Delete a user's voice."""
    user_dir = os.path.join(USER_VOICES_DIR, user_id)
    
    wav_path = os.path.join(user_dir, f"{voice_id}.wav")
    json_path = os.path.join(user_dir, f"{voice_id}.json")
    
    deleted = False
    for path in [wav_path, json_path]:
        if os.path.exists(path):
            os.remove(path)
            deleted = True
    
    return deleted


def set_default_voice(user_id: str, voice_id: str) -> bool:
    """Set a voice as the user's default."""
    filepath = get_voice_filepath(user_id, voice_id)
    if not filepath:
        return False
    
    user_dir = os.path.join(USER_VOICES_DIR, user_id)
    default_path = os.path.join(user_dir, "default.txt")
    
    with open(default_path, "w") as f:
        f.write(voice_id)
    
    return True


def get_default_voice(user_id: str) -> Optional[str]:
    """Get user's default voice ID."""
    default_path = os.path.join(USER_VOICES_DIR, user_id, "default.txt")
    if os.path.exists(default_path):
        with open(default_path) as f:
            return f.read().strip()
    return None