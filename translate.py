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


def _sanitize_translated_text(text: str) -> str:
    """
    Sanitize translated text by removing invalid surrogate characters.
    
    Translation APIs sometimes return text with invalid Unicode surrogates
    that cannot be encoded to UTF-8. This function removes them.
    
    Args:
        text: Translated text that may contain invalid Unicode characters.
        
    Returns:
        Sanitized text with invalid surrogates removed.
    """
    if not text:
        return text
    
    # Remove invalid surrogate characters (U+D800 to U+DFFF)
    # These are invalid in UTF-8 and cause encoding errors
    SURROGATE_START = 0xD800
    SURROGATE_END = 0xDFFF
    
    # First pass: remove surrogates
    sanitized = ''.join(
        char for char in text 
        if not (SURROGATE_START <= ord(char) <= SURROGATE_END)
    )
    
    # Second pass: ensure valid UTF-8 encoding
    try:
        sanitized.encode('utf-8')
    except UnicodeEncodeError:
        # If encoding still fails, use replace strategy
        sanitized = sanitized.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    
    return sanitized


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

    # Special prompt for Urdu - but output in Urdu script for Hindi TTS model compatibility
    # Since we're using a Hindi TTS model (hi_IN), we need Urdu script, not Urdu script
    if target_lang_readable.lower() == "urdu":
        prompt = (
            "You are a friendly conversational assistant that responds naturally in Urdu/Hindi. "
            "Automatically detect the source language of the user's input. "
            "Generate a natural, conversational reply in Hindi/Urdu using Urdu script. "
            "If the user demands another language, respond in that language.\n\n"
            "Rules:\n"
            "- If the input is a question, provide an appropriate answer (e.g., 'What is your name?' → 'میں گوگل جیمنی ہوں').\n"
            "- If the input is a greeting, respond with a greeting.\n"
            "- If the input is a statement, provide a natural response or acknowledgment.\n"
            "- Output MUST be in Urdu (Perso-Arabic) script, NOT Devanagari script.\n"
            "- Use proper Urdu orthography and natural conversational style.\n"
            "- Keep responses concise and natural (1-2 sentences typically).\n"
            "- Only return the reply in Urdu script, with no explanation, translation notes, or comments.\n"
            "- If the input is in Devanagari script, convert it to Urdu (Perso-Arabic) script in your response.\n"
            "- If the input is in Urdu (Perso-Arabic) script, respond naturally in Urdu.\n\n"
            "User input:\n{text}\n\n"
            "Your reply in Urdu script:"

        )
    else:
        prompt = (
            f"You are a friendly conversational assistant that responds naturally in {target_lang_readable}. "
            "Automatically detect the source language of the user's input. "
            f"Generate a natural, conversational reply in {target_lang_readable} or Roman {target_lang_readable} Script. If the user demands another language, respond in that language.\n\n"
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
        translated_text = (response.text or "").strip()
        
        # Sanitize the translated text to remove any invalid Unicode surrogates
        # that might cause encoding errors in downstream TTS processing
        translated_text = _sanitize_translated_text(translated_text)
        
        return translated_text
    except APIError as e:
        print(f"Gemini API Error during translation: {e}")
        return f"[Translation Error: {e}]"


__all__ = ["translate_text"]