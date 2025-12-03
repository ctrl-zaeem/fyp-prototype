"""
Text-to-speech (TTS) utilities using Piper (local, via subprocess).

Responsibilities:
- Choose appropriate Piper voice model based on logical language name.
- Convert text to speech and save as WAV file.
"""

import os
import subprocess
from typing import Optional

import config


def _get_piper_model_path(lang: str) -> str:
    """
    Resolve the Piper model path for the given logical language name.

    Fallback rules are encoded in config.PIPER_VOICE_MODELS:
        if lang == "hindi": use hin-IN
        if lang in ("urdu", "sindhi"): use ur_PK
        if lang == "punjabi": use hin-IN (fallback)
        if lang == "english": use en_US
    """
    lang_key = (lang or "").strip().lower()
    model_filename = config.PIPER_VOICE_MODELS.get(lang_key)
    if not model_filename:
        # Last-resort fallback: use English voice
        model_filename = config.PIPER_VOICE_MODELS.get("english")
    if not model_filename:
        raise RuntimeError("No Piper voice models configured in config.PIPER_VOICE_MODELS.")

    model_path = model_filename
    if not os.path.isabs(model_path):
        model_path = os.path.join(config.PIPER_VOICES_DIR, model_filename)
    return model_path


def text_to_speech(text: str, lang: str, output_path: Optional[str] = None) -> str:
    """
    Convert text to speech using Piper and save as a WAV file.

    Args:
        text: Text to be spoken.
        lang: Logical language name (e.g., "urdu", "hindi", "english").
        output_path: Optional path to output WAV file; defaults to config.OUTPUT_WAV_PATH.

    Returns:
        Path to the generated WAV file.
    """
    if not text:
        raise ValueError("text_to_speech called with empty text.")

    output_path = output_path or config.OUTPUT_WAV_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model_path = _get_piper_model_path(lang)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Piper model not found at '{model_path}'. "
            "Download the appropriate .onnx voice model and update config.PIPER_VOICES_DIR or PIPER_VOICE_MODELS."
        )

    cmd = [
        config.PIPER_EXECUTABLE,
        "--model",
        model_path,
        "--output_file",
        output_path,
    ]

    try:
        # Piper reads text from stdin and writes WAV to output_file.
        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Could not run Piper executable '{config.PIPER_EXECUTABLE}'. "
            "Ensure Piper is installed and available on your PATH, or set PIPER_EXECUTABLE in config.py."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Piper TTS failed with exit code {e.returncode}. Stderr:\n{e.stderr.decode('utf-8', errors='ignore')}"
        ) from e

    # Optional: log Piper stdout for debugging
    if proc.stdout:
        print(proc.stdout.decode("utf-8", errors="ignore"))

    print(f"TTS audio generated at: {output_path}")
    return output_path


__all__ = ["text_to_speech"]


