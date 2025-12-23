from typing import Optional


def guess_audio_format(content_type: Optional[str]) -> Optional[str]:
    """Guess audio format from Content-Type header."""
    if not content_type:
        return None
    ct = content_type.lower()
    
    format_map = {
        "wav": "wav",
        "mpeg": "mp3",
        "mp3": "mp3",
        "ogg": "ogg",
        "webm": "webm",
        "aac": "m4a",
        "mp4": "m4a",
        "m4a": "m4a",
    }
    
    for key, fmt in format_map.items():
        if key in ct:
            return fmt
    return None