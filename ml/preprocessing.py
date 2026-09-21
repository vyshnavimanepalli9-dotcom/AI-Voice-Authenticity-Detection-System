import numpy as np

def validate_audio_signal(y, sr):
    """
    Validates signal length, loudness, and non-empty status.
    Returns (is_valid, warning_message)
    """
    if len(y) == 0:
        return False, "Audio file contains no readable audio data."
    
    duration = len(y) / sr
    if duration < 0.2:
        return False, "Audio is too short (minimum 0.2 seconds required for acoustic feature extraction)."
    
    if duration > 300.0:
        return False, "Audio exceeds maximum duration of 5 minutes."

    rms = float(np.sqrt(np.mean(y**2)))
    if rms < 1e-7:
        return False, "Audio signal is completely silent. Please ensure microphone is unmuted."

    return True, None


def normalize_signal(y):
    """Peak normalization of audio signal vector to range [-1.0, 1.0]."""
    max_val = np.max(np.abs(y))
    if max_val > 0:
        return y / max_val
    return y

def trim_silence(y, top_db=30):
    """Trims leading and trailing silence from audio signal."""
    y_trimmed, _ = librosa_effects_trim(y, top_db=top_db)
    return y_trimmed

def librosa_effects_trim(y, top_db=30):
    import librosa
    return librosa.effects.trim(y, top_db=top_db)
