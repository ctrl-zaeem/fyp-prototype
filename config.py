"""
Configuration for the multilingual speech-to-speech translation project.
Edit this file to change defaults for target language, model paths, and API keys.
"""

import os

# =========================
# Gemini API configuration
# =========================

# Prefer environment variable so secrets are not committed to disk.
# Fallback to hard-coded placeholder that the user can edit.
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "AIzaSyBRL02cas4f5WmPWLR7sRkLFUEbVGQxkgY")

# Target language for translation (logical language name, not locale code)
# Set to "urdu" for Urdu-focused translation with Urdu script (Nastaliq/Perso-Arabic) output
TARGET_LANGUAGE: str = "urdu"

# =========================
# Whisper configuration
# =========================

# Whisper model name – use "medium" as requested for better multilingual accuracy.
WHISPER_MODEL_NAME: str = "medium"

# Sample rate for recording audio (Whisper expects 16 kHz)
SAMPLE_RATE: int = 16000

# Duration (seconds) for microphone recording in menu option 1 (can be overridden)
DEFAULT_RECORD_SECONDS: int = 10

# =========================
# Piper TTS configuration
# =========================

# Path to the piper executable (adjust if piper is not on PATH)
PIPER_EXECUTABLE: str = "piper"

# Base directory where Piper voice models (*.onnx) are stored.
PIPER_VOICES_DIR: str = os.path.join(os.path.dirname(__file__), "voices")

# Mapping from logical language names to Piper voice model filenames.
# You can change these to your preferred voices.
PIPER_VOICE_MODELS = {
    # Hindi
    "hindi": "hin-IN-sharma-medium.onnx",
    # Urdu (and fallback for Sindhi)
    "urdu": "ur_PK-ameen-medium.onnx",  # example; adjust to actual filename you download
    "sindhi": "ur_PK-ameen-medium.onnx",  # fallback to Urdu voice
    # Punjabi – fallback to Hindi voice
    "punjabi": "hin-IN-sharma-medium.onnx",
    # English
    "english": "en_US-lessac-medium.onnx",
}

# Default audio output file paths
OUTPUT_WAV_PATH: str = os.path.join(os.path.dirname(__file__), "output.wav")
TEMP_INPUT_WAV_PATH: str = os.path.join(os.path.dirname(__file__), "input.wav")


