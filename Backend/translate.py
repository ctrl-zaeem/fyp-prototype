"""
Translation and text helpers using a local Ollama model (e.g. Qwen 3.5).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

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
    """Remove Qwen/Ollama reasoning wrappers if they still appear in the text."""
    if not text:
        return text
    lower = text.lower()
    if "<redacted_thinking" not in lower and "<think" not in lower:
        return text
    import re
    cleaned = re.sub(
        r"<(?:redacted_)?think(?:ing)?\b[^>]*>[\s\S]*?</(?:redacted_)?think(?:ing)?>",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned


def _strip_hallucinations(text: str) -> str:
    """Remove Chinese characters and conversational markers the model might add."""
    if not text:
        return text
    import re
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)
    text = re.sub(r'\bHuman:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bAssistant:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'（\s*）', '', text)
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


_LAST_TRANSLATE_LOGS: list[str] = []


def get_last_translate_logs() -> list[str]:
    return list(_LAST_TRANSLATE_LOGS)


def _models_to_try(primary: str) -> list[str]:
    out = [primary]
    fb = getattr(config, "OLLAMA_MODEL_FALLBACK", "") or ""
    if fb and fb not in out:
        out.append(fb)
    return out


def _ollama_chat(
    *,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    think: bool = False,
    num_predict: Optional[int] = None,
) -> str:
    base = (getattr(config, "OLLAMA_BASE_URL", "http://127.0.0.1:11434") or "").rstrip("/")
    url = f"{base}/api/chat"
    opts: dict = {"temperature": temperature}
    if num_predict is not None:
        opts["num_predict"] = num_predict
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": opts,
    }
    timeout = getattr(config, "OLLAMA_TIMEOUT_SEC", 120.0)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"Ollama HTTP {e.code} at {url}: {detail}") from e
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot reach Ollama at {base}. Ensure `ollama serve` is running and the URL is correct. ({e.reason})"
        ) from e

    msg = raw.get("message") or {}
    return (msg.get("content") or "").strip()


def _ollama_try_models(
    *,
    messages_factory: Callable[[str], list[dict[str, str]]],
    primary_model: str,
    temperature: float,
    think: bool,
    num_predict: Optional[int],
    log: Callable[[str], None],
    progress_callback: Optional[Callable[[str], None]],
) -> str:
    last_err: Optional[BaseException] = None
    for attempt_model in _models_to_try(primary_model):
        for attempt in range(2):
            try:
                log(f"Calling Ollama {attempt_model} (attempt {attempt + 1})…")
                out = _ollama_chat(
                    messages=messages_factory(attempt_model),
                    model=attempt_model,
                    temperature=temperature,
                    think=think,
                    num_predict=num_predict,
                )
                if progress_callback:
                    try:
                        progress_callback(out)
                    except Exception:
                        pass
                log(f"Done. Received {len(out)} characters from {attempt_model}.")
                return out
            except BaseException as e:
                last_err = e
                err_s = str(e).lower()
                retryable = any(x in err_s for x in ("timeout", "connection", "refused", "temporarily"))
                log(f"{attempt_model} attempt {attempt + 1} failed: {e}")
                if retryable and attempt == 0:
                    time.sleep(2)
                    continue
                break
    if last_err:
        raise last_err
    return ""


def translate_text(
    text: str,
    target_lang: str,
    *,
    thinking: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Agricultural expert reply via local Ollama (Qwen, etc.).
    `thinking` is ignored: the server always receives ``think=false`` for short, conversational answers.
    """
    _ = thinking

    if not text:
        return ""

    target_lang_readable = _normalize_target_lang(target_lang)
    primary_model = getattr(config, "OLLAMA_MODEL", "qwen3.5")

    target_lang_lower = target_lang_readable.lower()
    if target_lang_lower == "urdu":
        system_prompt = (
            "آپ ایک زرعی ماہر (AI زرعی معاون) ہیں۔ کسانوں سے بات چیت کے انداز میں بات کریں۔ "
            "جواب بہت مختصر، سادہ اور براہ راست دیں۔ صرف 2 سے 4 لائنوں میں جواب دیں۔ "
            "طویل وضاحتوں سے گریز کریں جب تک کہ خاص طور پر نہ پوچھا جائے۔ "
            "ہمیشہ اردو (نستعلیق) رسم الخط میں جواب دیں۔ "
            "CRITICAL: Do NOT output any chat markers like 'Human:' or 'Assistant:'. "
            "Do NOT start with 'اردو میں:' or any language prefix. Provide ONLY your direct answer."
        )
    elif target_lang_lower == "english":
        system_prompt = (
            "You are an Agricultural Expert named 'AI Agriculture Assistant'. "
            "Your task is to answer the user's agricultural question in simple English. "
            "Speak in a conversational tone suitable for farmers. "
            "Keep your response short, simple, and direct—ideally 2 to 4 lines. "
            "CRITICAL: Do NOT output any chat markers like 'Human:' or 'Assistant:'. "
            "Provide ONLY your direct answer in English."
        )
    else:
        system_prompt = (
            f"You are an Agricultural Expert named 'AI زرعی معاون'. "
            f"Your task is to answer the user's agricultural question in proper {target_lang_readable} language. "
            f"CRITICAL SCRIPT RULE: You MUST write your response EXCLUSIVELY in the Pakistani Arabic/Urdu script (e.g., Shahmukhi for Punjabi, Sindhi Arabic script for Sindhi, Pashto Arabic script). "
            f"DO NOT use Gurmukhi. DO NOT use Devanagari. DO NOT use Latin/English script. If you generate even a single Gurmukhi or Devanagari character, you fail. "
            f"Even if the user's question is transcribed in Hindi/Devanagari/Gurmukhi script, you MUST understand it and respond using the Arabic/Urdu script. "
            "Keep your response short, simple, and direct—ideally 2 to 4 lines. "
            f"If you absolutely cannot answer in {target_lang_readable}, you may respond in simple Urdu instead. "
            "CRITICAL: Do NOT output any chat markers like 'Human:' or 'Assistant:'. "
            f"Do NOT start your response with '{target_lang_readable}:' or 'In {target_lang_readable}:' or 'سنڌي ۾:'. "
            "Do NOT output Chinese. Provide ONLY your direct answer in the requested language."
        )

    convo_rules = (
        "Talk like a real person: natural, brief, no meta-commentary. "
        "Do NOT use chain-of-thought, hidden reasoning, or any <think> / thinking tags. "
        "Answer immediately in the user-facing language only."
    )

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
    _log(f"Target language: {target_lang_readable} | Ollama model: {primary_model} | think=false (dialogue)")

    num_predict = getattr(config, "OLLAMA_NUM_PREDICT_CHAT", 512)

    def _messages(full_system: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": full_system},
            {"role": "user", "content": text},
        ]

    def _call_llm(extra_system: str = "") -> str:
        full_system = system_prompt + "\n\n" + convo_rules
        if extra_system:
            full_system = full_system + "\n\n" + extra_system

        return _ollama_try_models(
            messages_factory=lambda _m: _messages(full_system),
            primary_model=primary_model,
            temperature=0.45,
            think=False,
            num_predict=num_predict,
            log=_log,
            progress_callback=progress_callback,
        )

    try:
        translated_text = _call_llm()

        if not translated_text:
            return "[معذرت، مجھے آپ کی بات سمجھ نہیں آئی۔ براہ کرم دوبارہ کوشش کریں۔]"

        cleaned = _strip_qwen_thinking_blocks(translated_text)
        if not cleaned and (
            "<redacted_thinking" in translated_text.lower() or "<think" in translated_text.lower()
        ):
            translated_text = _call_llm(
                extra_system="CRITICAL: Return ONLY the final spoken answer. No reasoning tags or preambles."
            )
            if not translated_text:
                return "[معذرت، مجھے آپ کی بات سمجھ نہیں آئی۔ براہ کرم دوبارہ کوشش کریں۔]"
            cleaned = _strip_qwen_thinking_blocks(translated_text)

        final_text = cleaned or translated_text
        final_text = _strip_hallucinations(final_text)

        if not final_text:
            return "[معذرت، مجھے آپ کی بات سمجھ نہیں آئی۔ براہ کرم دوبارہ کوشش کریں۔]"

        def _has_invalid_script(t: str) -> bool:
            for char in t:
                if '\u0A00' <= char <= '\u0A7F' or '\u0900' <= char <= '\u097F':
                    return True
            return False

        if _has_invalid_script(final_text):
            _log("Detected Gurmukhi or Devanagari in output. Retrying with strict script enforcement...")
            retry_text = _call_llm(
                extra_system=(
                    "CRITICAL: Your last reply used Gurmukhi or Devanagari. "
                    "Rewrite using ONLY Pakistani Arabic script (Shahmukhi style). Output ONLY the corrected text."
                )
            )
            if retry_text:
                cleaned_retry = _strip_hallucinations(_strip_qwen_thinking_blocks(retry_text))
                final_text = cleaned_retry or retry_text

        return _sanitize_translated_text(final_text)

    except Exception as e:
        _log(f"Ollama error: {e}")
        raise


def normalize_transcript_for_display(text: str, target_lang: str) -> str:
    """
    Normalizes transcribed text into native Perso-Arabic script for the target language.
    """
    if not text:
        return text

    target_lang_readable = _normalize_target_lang(target_lang)
    primary_model = getattr(config, "OLLAMA_MODEL", "qwen3.5")
    system_prompt = (
        f"You are a transliteration engine. Your task is to accurately convert the user's spoken {target_lang_readable} text "
        f"into its proper native Arabic/Perso-Arabic script. "
        "The input might be in Latin/Roman script, Devanagari script, or an imperfect transcription. "
        "DO NOT answer any questions or add any conversational text. "
        "ONLY output the exact same meaning in the proper Arabic/Perso-Arabic script. "
        f"If the target is Sindhi and input is in Devanagari, rewrite it in proper Sindhi Arabic script. "
        "CRITICAL: Output ONLY the transliterated text. No chain-of-thought or <think> blocks."
    )
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    np_util = getattr(config, "OLLAMA_NUM_PREDICT_UTIL", 4096)

    last = ""
    for m in _models_to_try(primary_model):
        try:
            last = _ollama_chat(
                messages=msgs,
                model=m,
                temperature=0.1,
                think=False,
                num_predict=np_util,
            )
            if last:
                break
        except BaseException as e:
            _LAST_TRANSLATE_LOGS.append(f"Normalization error ({m}): {e}")
            continue
    if not last:
        return text

    cleaned = _strip_hallucinations(_strip_qwen_thinking_blocks(last))
    return cleaned if cleaned else text


def transliterate_regional_for_tts(text: str, target_lang: str) -> str:
    """
    Rewrite regional Arabic-script text into standard Urdu letters for Urdu TTS.
    """
    if not text:
        return text

    target_lang_readable = _normalize_target_lang(target_lang)
    primary_model = getattr(config, "OLLAMA_MODEL", "qwen3.5")
    system_prompt = (
        f"You are a phonetic transliteration engine. The user provides text in {target_lang_readable} (Arabic script). "
        "Rewrite using ONLY standard Urdu letters so an Urdu TTS engine can read it clearly. "
        "Map language-specific letters to close Urdu sounds. Do NOT translate meaning to Urdu. "
        "CRITICAL: Output ONLY the transliteration. No <think> or reasoning."
    )
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    np_util = getattr(config, "OLLAMA_NUM_PREDICT_UTIL", 4096)

    last = ""
    for m in _models_to_try(primary_model):
        try:
            last = _ollama_chat(
                messages=msgs,
                model=m,
                temperature=0.1,
                think=False,
                num_predict=np_util,
            )
            if last:
                break
        except BaseException as e:
            _LAST_TRANSLATE_LOGS.append(f"TTS transliteration error ({m}): {e}")
            continue
    if not last:
        return text

    cleaned = _strip_hallucinations(_strip_qwen_thinking_blocks(last))
    return cleaned if cleaned else text


__all__ = ["translate_text", "get_last_translate_logs", "normalize_transcript_for_display", "transliterate_regional_for_tts"]
