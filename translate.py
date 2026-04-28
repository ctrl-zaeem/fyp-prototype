"""
Assistant response generation using local Ollama (Qwen) via /api/chat.
"""

import json
import time
from typing import Callable, Optional

import requests

import config


def _normalize_target_lang(target_lang: str) -> str:
    """Normalize logical language names for prompts."""
    lang = (target_lang or "").strip().lower()
    mapping = {
        "urdu": "Urdu",
        "hindi": "Hindi",
        "punjabi": "Punjabi",
        "sindhi": "Sindhi",
        "english": "English",
        "pashto": "Pashto",
        "balochi": "Balochi",
    }
    return mapping.get(lang, target_lang)


def _sanitize_translated_text(text: str) -> str:
    """Remove invalid surrogate characters."""
    if not text:
        return text

    SURROGATE_START = 0xD800
    SURROGATE_END = 0xDFFF

    sanitized = ''.join(
        char for char in text
        if not (SURROGATE_START <= ord(char) <= SURROGATE_END)
    )

    try:
        sanitized.encode('utf-8')
    except UnicodeEncodeError:
        sanitized = sanitized.encode('utf-8', errors='replace').decode('utf-8', errors='replace')

    return sanitized


def _strip_qwen_thinking_blocks(text: str) -> str:
    """
    Remove Qwen-style <think>...</think> blocks from a model response.
    """
    if not text:
        return text
    lower = text.lower()
    if "<think" not in lower:
        return text
    import re
    # Handle both <think>...</think> and <think ...>...</think>
    cleaned = re.sub(r"<think\b[^>]*>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    return cleaned


_LAST_TRANSLATE_LOGS: list[str] = []


def get_last_translate_logs() -> list[str]:
    return list(_LAST_TRANSLATE_LOGS)


def translate_text(
    text: str,
    target_lang: str,
    *,
    thinking: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Generate a detailed, professional agricultural expert response using local Ollama Qwen.
    """
    if not text:
        return ""

    target_lang_readable = _normalize_target_lang(target_lang)
    model_name = getattr(config, "QWEN_MODEL_NAME", "qwen3:8b")

    # === PROFESSIONAL AGRICULTURAL EXPERT SYSTEM PROMPT ===
    if target_lang_readable.lower() == "urdu":
        system_prompt = (
            "آپ ایک تجربہ کار اور پیشہ ور زرعی ماہر ہیں۔ آپ کا نام 'AI زرعی معاون' ہے۔ "
            "ہر سوال کا جواب بہت تفصیل سے، پیشہ ورانہ انداز میں، اور قدم بہ قدم دیں۔ "
            "جوابات میں شامل کریں:\n"
            "1. مسئلے کی درست تشخیص\n"
            "2. وجوہات کی وضاحت\n"
            "3. عملی اور قابل عمل حل (قدم بہ قدم گائیڈ)\n"
            "4. احتیاطی تدابیر اور بہترین طریقے\n"
            "5. اگر ضروری ہو تو کیمیکل/قدرتی علاج دونوں کا ذکر\n"
            "6. کسان کے لیے آسان زبان میں سمجھانا\n\n"
            "ہمیشہ اردو (نستعلیق) رسم الخط میں جواب دیں۔ "
            "جوابات مفید، تفصیلی، اور عملی ہوں۔ "
            "کبھی بھی اپنی سوچ کا اظہار نہ کریں، صرف مفید زرعی مشورہ دیں۔ "
            "جوابات واضح، منظم اور کسان کے لیے آسان ہوں۔"
        )
    elif target_lang_readable.lower() in ["pashto", "balochi"]:
        system_prompt = (
            f"You are a highly experienced and professional Agricultural Expert. "
            f"Your name is 'AI زرعی معاون'. Respond naturally in proper {target_lang_readable} language. "
            "For every question, give a detailed, step-by-step professional guide. "
            "Include: proper diagnosis of the problem, possible causes, practical solutions with clear steps, "
            "prevention tips, and best farming practices. "
            "Use simple and clear language that a farmer can easily understand. "
            "Always respond ONLY in the target language. Never add explanations in English."
        )
    else:
        system_prompt = (
            f"You are a highly experienced and professional Agricultural Expert named 'AI Agriculture Assistant'. "
            f"Respond in proper {target_lang_readable} language. "
            "For every farmer's question, provide a detailed, helpful, and professional response with:\n"
            "1. Clear diagnosis of the issue\n"
            "2. Explanation of possible causes\n"
            "3. Step-by-step practical solutions and guides\n"
            "4. Prevention methods and best practices\n"
            "5. Both chemical and natural remedies where applicable\n\n"
            "Keep language simple and easy for farmers to understand. "
            "Always be helpful, professional, and solution-oriented. "
            "Output ONLY the response in the target language. No English explanations."
        )

    # Build messages
    thinking_instruction = (
        "You may use a private <think>...</think> reasoning block before the final answer."
        if thinking
        else "Do not include any <think>...</think> blocks. Provide only the final answer."
    )
    messages = [
        {"role": "system", "content": system_prompt + "\n\n" + thinking_instruction},
        {"role": "user", "content": text}
    ]

    def _log(msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        _LAST_TRANSLATE_LOGS.append(line)
        if log_callback:
            try:
                log_callback(line)
            except Exception:
                pass

    _LAST_TRANSLATE_LOGS.clear()
    _log(f"Target language: {target_lang_readable} | Model: {model_name} | Thinking: {'ON' if thinking else 'OFF'}")

    def _timeout_tuple():
        connect_t = getattr(config, "OLLAMA_CONNECT_TIMEOUT_S", 5)
        read_t = getattr(config, "OLLAMA_READ_TIMEOUT_S", None)
        # requests supports (connect, read) where read can be None (wait forever)
        return (connect_t, read_t)

    def _call_ollama(extra_system: str = "") -> str:
        opts = {
            "temperature": 0.65,   # Slightly lower for more professional tone
            "top_p": 0.95,
        }
        # IMPORTANT: Do NOT use stop tokens for <think> here.
        # Some Qwen variants begin the response with <think>, and stopping on it yields an empty answer.
        payload_messages = messages
        if extra_system:
            payload_messages = [
                {"role": "system", "content": system_prompt + "\n\n" + thinking_instruction + "\n\n" + extra_system},
                {"role": "user", "content": text},
            ]

        # Prefer streaming so the UI can show progress.
        try:
            _log("Connecting to Ollama…")
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model_name,
                    "messages": payload_messages,
                    "stream": True,
                    "options": opts,
                },
                stream=True,
                timeout=_timeout_tuple(),
            )
        except Exception as e:
            _log(f"Connection attempt failed: {e}")
            raise

        if response.status_code != 200:
            _log(f"Ollama API error HTTP {response.status_code}")
            return ""

        _log("Generating response…")
        parts: list[str] = []
        for raw in response.iter_lines(decode_unicode=True):
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            chunk = ((obj.get("message", {}) or {}).get("content")) or ""
            if chunk:
                parts.append(chunk)
                if progress_callback:
                    try:
                        progress_callback("".join(parts))
                    except Exception:
                        pass
            if obj.get("done") is True:
                break

        final = "".join(parts).strip()
        _log(f"Done. Received {len(final)} characters.")
        return final

    try:
        translated_text = _call_ollama()

        if not translated_text:
            return "[کوئی جواب نہیں ملا]"

        cleaned = _strip_qwen_thinking_blocks(translated_text)
        # If the model returned ONLY a thinking block (cleaned becomes empty), retry once with stricter instruction.
        if not thinking and not cleaned and "<think" in translated_text.lower():
            translated_text = _call_ollama(
                extra_system=(
                    "CRITICAL: Return ONLY the final answer text. "
                    "Do NOT output <think> tags or any reasoning."
                )
            )
            if not translated_text:
                return "[کوئی جواب نہیں ملا]"
            cleaned = _strip_qwen_thinking_blocks(translated_text)

        # Final fallback: if still empty, return whatever we got (sanitized) rather than nothing.
        final_text = cleaned or translated_text
        return _sanitize_translated_text(final_text)

    except requests.exceptions.ConnectionError:
        return "[غلطی: Ollama چل رہا نہیں ہے۔ براہ مہربانی 'ollama serve' چلائیں]"
    except requests.exceptions.Timeout:
        return "[غلطی: Ollama سے رابطہ میں مسئلہ (timeout)]"
    except Exception as e:
        print(f"Translation error: {e}")
        return "[زرعی معاون میں تکنیکی خرابی]"



__all__ = ["translate_text", "get_last_translate_logs"]