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

CUSTOM_VOICES = {
    "Aiym": "aiym.wav",
}

os.makedirs(USER_VOICES_DIR, exist_ok=True)


def get_model() -> TTS:
    global _model
    if _model is None:
        _model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    return _model


def _get_valid_speakers() -> List[str]:
    try:
        model = get_model()
        if hasattr(model, "speakers") and model.speakers:
            return model.speakers
    except Exception:
        pass
    return []


def _validate_speaker(speaker: str) -> str:
    valid = _get_valid_speakers()
    if not valid:
        return speaker
    if speaker in valid:
        return speaker
    print(f"[TTS WARNING] Speaker '{speaker}' not in model. "
          f"Falling back to '{DEFAULT_SPEAKER}'")
    if DEFAULT_SPEAKER in valid:
        return DEFAULT_SPEAKER
    fallback = valid[0]
    print(f"[TTS WARNING] DEFAULT_SPEAKER also invalid, using '{fallback}'")
    return fallback


def list_speakers() -> List[str]:
    voices = ["default"]
    voices.extend(CUSTOM_VOICES.keys())
    return voices


def get_user_voice_path(user_id: str, voice_id: str) -> Optional[str]:
    path = os.path.join(USER_VOICES_DIR, user_id, f"{voice_id}.wav")
    if os.path.exists(path):
        return path
    return None


def get_user_default_voice_path(user_id: str) -> Optional[str]:
    default_file = os.path.join(USER_VOICES_DIR, user_id, "default.txt")
    if os.path.exists(default_file):
        with open(default_file) as f:
            voice_id = f.read().strip()
        return get_user_voice_path(user_id, voice_id)
    return None


def _wav_to_bytes(wav: list, output_format: str = "mp3") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, samplerate=SAMPLE_RATE, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    if not wav_bytes or len(wav_bytes) < 100:
        print("[TTS WARNING] Generated WAV is empty or suspiciously small")

    if output_format == "wav":
        return wav_bytes

    audio = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
    audio = audio.set_frame_rate(SAMPLE_RATE)

    if audio.dBFS < -50:
        print(f"[TTS WARNING] Output audio is near-silent (dBFS={audio.dBFS:.1f})")

    out = io.BytesIO()
    audio.export(out, format="mp3", parameters=["-q:a", "3", "-ar", str(SAMPLE_RATE)])
    return out.getvalue()


def synthesize_with_reference(
    text: str,
    speaker_wav: str,
    lang: str = XTTS_LANG,
    output_format: str = "mp3",
) -> bytes:
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
    tts = get_model()
    voice = voice or "default"

    if user_id:
        user_voice_path = get_user_voice_path(user_id, voice)
        if user_voice_path:
            wav = tts.tts(text=text, language=lang, speaker_wav=user_voice_path)
            return _wav_to_bytes(wav, output_format)

        if voice.lower() in ("parent", "user", "my_voice"):
            default_path = get_user_default_voice_path(user_id)
            if default_path:
                wav = tts.tts(text=text, language=lang, speaker_wav=default_path)
                return _wav_to_bytes(wav, output_format)

    if voice in CUSTOM_VOICES:
        ref_path = os.path.join(VOICES_DIR, CUSTOM_VOICES[voice])
        if not os.path.exists(ref_path):
            print(f"[TTS WARNING] Custom voice file not found: {ref_path}, "
                  f"falling back to default")
            speaker = _validate_speaker(DEFAULT_SPEAKER)
            wav = tts.tts(text=text, language=lang, speaker=speaker)
        else:
            wav = tts.tts(text=text, language=lang, speaker_wav=ref_path)
    else:
        if voice != "default":
            print(f"[TTS WARNING] Unknown voice '{voice}', using default")
        speaker = _validate_speaker(DEFAULT_SPEAKER)
        wav = tts.tts(text=text, language=lang, speaker=speaker)

    return _wav_to_bytes(wav, output_format)


def warmup() -> None:
    try:
        audio = synthesize("Привет!", lang=XTTS_LANG)
        if len(audio) < 100:
            print("[TTS WARNING] Warmup produced suspiciously small audio")
        else:
            print("TTS warmup complete")
    except Exception as e:
        print(f"TTS warmup failed: {e}")