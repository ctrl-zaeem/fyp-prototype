"""
Speech-to-text (STT) utilities using OpenAI Whisper (local).

Responsibilities:
- Record microphone audio
- Transcribe using Whisper "medium" model
- Focus on Urdu language detection and transcription
"""

import queue
import sys
from typing import TYPE_CHECKING, Tuple, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

import config

if TYPE_CHECKING:  # Only imported for type checkers; avoids runtime/lint errors if not installed yet.
    import whisper  # pragma: no cover


_whisper_model: Optional["whisper.Whisper"] = None


def _get_whisper_model() -> "whisper.Whisper":
    """Lazily load and cache the Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'openai-whisper' package is required but not installed. "
                "Install it with 'pip install openai-whisper'."
            ) from exc
        print(f"Loading Whisper model '{config.WHISPER_MODEL_NAME}' (this may take a while)...")
        _whisper_model = whisper.load_model(config.WHISPER_MODEL_NAME)
        print("Whisper model loaded.")
    return _whisper_model


def record_audio(duration: int = config.DEFAULT_RECORD_SECONDS, output_path: str = config.TEMP_INPUT_WAV_PATH) -> str:
    """
    Record audio from the default microphone for the given duration (seconds).

    Returns the path to the saved WAV file.
    """
    samplerate = config.SAMPLE_RATE
    print(f"Recording audio for {duration} seconds at {samplerate} Hz...")

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time, status):  # type: ignore[override]
        if status:
            print(f"Recording status: {status}", file=sys.stderr)
        audio_q.put(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
        frames = []
        for _ in range(int(duration * samplerate / 1024) + 1):
            frames.append(audio_q.get())

    audio_data = np.concatenate(frames, axis=0).flatten()

    # Save as 16-bit PCM WAV
    sf.write(output_path, audio_data, samplerate)
    print(f"Audio saved to: {output_path}")
    return output_path


def speech_to_text(audio_path: str) -> Tuple[str, str]:
    """
    Transcribe speech in the given audio file using Whisper.
    Focuses on Urdu language detection and transcription.

    Returns:
        (text, detected_language_code)
    where detected_language_code is 'ur' for Urdu (or detected language if not Urdu).
    """
    model = _get_whisper_model()

    print(f"Transcribing audio: {audio_path}")
    # Force Urdu language detection - Whisper will prioritize Urdu
    # If input is not Urdu, it will still detect but we prefer Urdu
    result = model.transcribe(audio_path, language="ur")

    text: str = result.get("text", "").strip()
    lang: str = result.get("language", "ur")  # Default to Urdu

    print(f"Detected language: {lang}")
    print(f"Transcription: {text}")
    return text, lang


def map_whisper_lang_to_name(lang_code: str) -> str:
    """
    Map Whisper language code to logical language name used in this project.
    Defaults to Urdu for Urdu-focused translation.

    Examples:
        'ur' -> 'urdu' (primary)
        'hi' -> 'urdu' (mapped to Urdu for translation)
        'en' -> 'english'
        'pa' -> 'urdu' (Punjabi mapped to Urdu)
        'sd' -> 'urdu' (Sindhi mapped to Urdu)
    """
    code = (lang_code or "").lower()
    # Map all Indic languages to Urdu for Urdu-focused translation
    mapping = {
        "ur": "urdu",
        "hi": "urdu",  # Hindi input -> translate to Urdu
        "pa": "urdu",  # Punjabi input -> translate to Urdu
        "sd": "urdu",  # Sindhi input -> translate to Urdu
        "en": "english",  # Keep English separate
    }
    return mapping.get(code, "urdu")  # Default to Urdu


__all__ = ["record_audio", "speech_to_text", "map_whisper_lang_to_name"]


