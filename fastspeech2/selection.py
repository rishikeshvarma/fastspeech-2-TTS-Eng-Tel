"""[new] Helper shared by the English and Telugu data selection.

Training hours are counted as speech: the leading and trailing silence that the
preprocessor later cuts off at the forced-alignment boundaries is excluded already at
selection time. A 30 dB energy trim matches the MFA-based trimming to within 0.4 % on
Telugu and over-estimates it by ~5 % on Svarah (measured on 400 clips each); the
remaining gap is covered by the per-language `margin` in config.yaml.
"""

import io

import librosa
import soundfile as sf


def speech_seconds(audio_bytes, top_db):
    """Duration of a clip after trimming leading/trailing silence."""
    y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    _, (start, end) = librosa.effects.trim(y.mean(axis=1), top_db=top_db)
    return (end - start) / sr
