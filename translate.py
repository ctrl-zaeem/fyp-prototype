"""
Translation utilities using Gemini 2.5 Flash API.

Responsibilities:
- Configure Gemini client
- Provide translate_text(text, target_lang) that:
    * Auto-detects source language via prompt
    * Generates conversational replies in the target language (e.g., Urdu)
    * For questions, provides natural responses instead of literal translations
"""

from typing import Optional

# 🔑 The main changes are here: using Client instead of GenerativeModel
from google import genai
from google.genai import Client
from google.genai.errors import APIError

import config


# Changed from genai.GenerativeModel to genai.Client
_gemini_client: Optional[Client] = None
# Added constant for the model name
_GEMINI_MODEL_NAME = "gemini-2.5-flash"


def _get_gemini_client() -> Client:
    """Lazily configure and return the Gemini Client."""
    global _gemini_client
    if _gemini_client is None:
        if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "YOUR_KEY_HERE":
            raise RuntimeError(
                "Gemini API key is not set. Please set GEMINI_API_KEY in config.py or as an environment variable."
            )
        # 🔑 CHANGE 1: Create the Client instance, passing the API key directly
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


def _normalize_target_lang(target_lang: str) -> str:
    """
    Normalize logical language names to human-readable names for the prompt.
    """
    lang = (target_lang or "").strip().lower()
    mapping = {
        "urdu": "Urdu",
        "hindi": "Hindi",
        "punjabi": "Punjabi",
        "sindhi": "Sindhi",
        "english": "English",
    }
    return mapping.get(lang, target_lang)


def translate_text(text: str, target_lang: str) -> str:
    """
    Generate a conversational reply in the target language using Gemini.
    For questions, provides natural responses instead of literal translations.

    Args:
        text: Input text in any supported language (can be a question, statement, etc.).
        target_lang: Logical target language name (e.g., "urdu", "hindi").

    Returns:
        Conversational reply in the target language as a string.
        
    Examples:
        Input: "What is your name?" -> Output: "میرا نام زعیم ہے" (My name is Zaeem)
        Input: "How are you?" -> Output: "میں ٹھیک ہوں، شکریہ" (I am fine, thank you)
    """
    if not text:
        return ""

    # 🔑 CHANGE 2: Get the client instead of the model
    client = _get_gemini_client()
    target_lang_readable = _normalize_target_lang(target_lang)

    # Special prompt for Urdu to generate conversational replies in Urdu script
    if target_lang_readable.lower() == "urdu":
        prompt = (
            "You are a friendly conversational assistant that responds naturally in Urdu (اردو). "
            "Automatically detect the source language of the user's input. "
            "Generate a natural, conversational reply in Urdu script (Nastaliq/Perso-Arabic), NOT Hindi/Devanagari script.\n\n"
            "Rules:\n"
            "- If the input is a question, provide an appropriate answer in Urdu (e.g., 'What is your name?' → 'میرا نام زعیم ہے').\n"
            "- If the input is a greeting, respond with a greeting in Urdu.\n"
            "- If the input is a statement, provide a natural response or acknowledgment in Urdu.\n"
            "- Output MUST be in Urdu script (Nastaliq/Perso-Arabic script), NOT Hindi/Devanagari script.\n"
            "- Use proper Urdu orthography and natural conversational style.\n"
            "- Keep responses concise and natural (1-2 sentences typically).\n"
            "- Only return the Urdu reply, with no explanation, translation notes, or comments.\n"
            "- If the input is already in Urdu script, respond naturally in Urdu.\n"
            "- If the input is in Hindi/Devanagari script, respond in Urdu script.\n\n"
            f"User input:\n{text}\n\n"
            "Your Urdu reply:"
        )
    else:
        prompt = (
            f"You are a friendly conversational assistant that responds naturally in {target_lang_readable}. "
            "Automatically detect the source language of the user's input. "
            f"Generate a natural, conversational reply in {target_lang_readable}.\n\n"
            "Rules:\n"
            "- If the input is a question, provide an appropriate answer.\n"
            "- If the input is a greeting, respond with a greeting.\n"
            "- If the input is a statement, provide a natural response or acknowledgment.\n"
            "- Keep responses concise and natural (1-2 sentences typically).\n"
            "- Only return the reply, with no explanation or comments.\n\n"
            f"User input:\n{text}\n\n"
            f"Your {target_lang_readable} reply:"
        )

    try:
        # 🔑 CHANGE 3: Call generate_content on the client's models attribute,
        # and pass the model name explicitly.
        response = client.models.generate_content(
            model=_GEMINI_MODEL_NAME, 
            contents=prompt
        )
        return (response.text or "").strip()
    except APIError as e:
        print(f"Gemini API Error during translation: {e}")
        return f"[Translation Error: {e}]"


__all__ = ["translate_text"]