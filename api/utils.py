from typing import Optional


def guess_audio_format(content_type: Optional[str]) -> Optional[str]:
    """Guess audio format from Content-Type header.
    
    Covers standard MIME types plus common mobile variants
    (Flutter/Android/iOS often send non-standard content types).
    """
    if not content_type:
        return None
    ct = content_type.lower().strip()

    exact_map = {
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/wave": "wav",
        "audio/vnd.wave": "wav",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/ogg": "ogg",
        "audio/opus": "ogg",       # opus in ogg container
        "audio/webm": "webm",
        "audio/aac": "m4a",
        "audio/mp4": "m4a",
        "audio/x-m4a": "m4a",
        "audio/m4a": "m4a",
        "audio/3gpp": "3gp",       # Android default recorder
        "audio/3gpp2": "3gp",
        "audio/amr": "amr",        # older Android devices
        "audio/flac": "flac",
        "audio/x-flac": "flac",
        "audio/x-caf": "caf",      # iOS Core Audio Format
        "video/webm": "webm",      # some browsers send video/webm for audio
    }

    # Check exact match (strip parameters like ;codecs=...)
    base_type = ct.split(";")[0].strip()
    if base_type in exact_map:
        return exact_map[base_type]

    # Fallback: substring matching for edge cases
    fallback_map = {
        "wav": "wav",
        "mpeg": "mp3",
        "mp3": "mp3",
        "ogg": "ogg",
        "opus": "ogg",
        "webm": "webm",
        "aac": "m4a",
        "mp4": "m4a",
        "m4a": "m4a",
        "3gp": "3gp",
        "amr": "amr",
        "flac": "flac",
        "caf": "caf",
    }

    for key, fmt in fallback_map.items():
        if key in ct:
            return fmt

    print(f"[AUDIO] Unknown content type: {ct}")
    return None