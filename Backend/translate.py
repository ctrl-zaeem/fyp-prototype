"""
Translation and text helpers using a local Ollama model (e.g. gemma3:4b).

For Sindhi, Punjabi, and Pashto a reliable pipeline is used:
  1. Interpret the user's question in English (from regional / Perso-Arabic STT text).
  2. Generate the agricultural answer in English (stable for local LLMs).
  3. Translate that English answer into the target regional language with a
     strict translation call that enforces the correct Arabic script.
Speech input uses the same path as typed regional text (never skips the
English understanding step).
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


# Languages that use the two-step pipeline (answer in Urdu → translate to target)
_TWO_STEP_LANGUAGES: frozenset[str] = frozenset({"sindhi", "punjabi", "pashto"})

# Script rules per language for the translation step
_SCRIPT_RULES: dict[str, str] = {
    "sindhi": (
        "Sindhi Arabic script (Perso-Arabic, used in Pakistan). "
        "Use characters like ڄ ڃ ٻ ڪ ڳ ڱ ڻ ٽ ٺ ڦ ڙ ڇ as needed. "
        "Do NOT use Devanagari or Gurmukhi."
    ),
    "punjabi": (
        "Shahmukhi script (Perso-Arabic Punjabi script used in Pakistan). "
        "Do NOT use Gurmukhi. Do NOT use Devanagari. "
        "Use standard Urdu/Arabic letters to phonetically represent Punjabi sounds."
    ),
    "pashto": (
        "Pashto Arabic script. "
        "Use Pashto-specific letters like ږ ښ څ ځ ډ ړ ټ ڼ ګ as needed. "
        "Do NOT use Latin script or Devanagari."
    ),
}

# Sindhi-heavy letters/tokens that indicate Punjabi->Sindhi drift.
_SINDHI_HEAVY_CHARS: frozenset[str] = frozenset("ڄڃٻڪڳڱڻٺڦڏڊڍڌ۾۽")
_SINDHI_HEAVY_TOKENS: tuple[str, ...] = ("۾", "۽", "آهي", "آھن", "آهن")
_PUNJABI_NORMALIZE_MAP: tuple[tuple[str, str], ...] = (
    (" ۾ ", " وچ "),
    ("۾", " وچ "),
    (" ۽ ", " تے "),
    ("۽", " تے "),
    ("آهي", "اے"),
    ("آھن", "نے"),
    ("آهن", "نے"),
    ("جي", "دی"),
    ("ٿ", "ت"),
    ("ڀ", "بھ"),
    ("ڇ", "چ"),
    ("ٻ", "ب"),
    ("ڪ", "ک"),
    ("ڳ", "گ"),
    ("ڱ", "نگ"),
    ("ڻ", "ن"),
    ("ڏ", "ڈ"),
    ("ڊ", "ڈ"),
    ("ڍ", "ڈ"),
    ("ڌ", "د"),
    ("ڄ", "ج"),
    ("ڃ", "ج"),
    ("ڦ", "پھ"),
)

# -----------------------------------------------------------------------
# Deterministic Unicode character maps: regional script → standard Urdu
# Used by transliterate_regional_for_tts() so the Urdu TTS voice can
# pronounce every character without an additional LLM call.
# This guarantees the spoken audio matches the displayed text exactly.
# -----------------------------------------------------------------------
_PASHTO_CHAR_MAP: dict[str, str] = {
    "ږ": "ژ",   # zh / ẑ
    "ښ": "ش",   # sh / ṣ
    "ګ": "گ",   # g
    "ځ": "ز",   # dz → z
    "څ": "س",   # ts → s
    "ډ": "ڈ",   # retroflex d
    "ړ": "ڑ",   # retroflex r
    "ټ": "ٹ",   # retroflex t
    "ڼ": "ن",   # retroflex n
    "ۍ": "ی",   # əj → i
    "ۀ": "ہ",   # schwa
    "ې": "ے",   # e
    "ؤ": "و",   # w
    "ئ": "ی",   # yi
}

_SINDHI_CHAR_MAP: dict[str, str] = {
    "ڄ": "ج",
    "ڃ": "ج",
    "ٻ": "ب",
    "ڪ": "ک",
    "ڳ": "گ",
    "ڱ": "ن",
    "ڻ": "ن",
    "ٽ": "ت",
    "ٺ": "ٹ",
    "ڦ": "ف",
    "ڙ": "ر",
    "ڇ": "چ",
    "ڍ": "ڈ",
    "ڌ": "د",
    "ڊ": "ڈ",
    "ڏ": "ڈ",
    "ھ": "ہ",
    "۾": " میں ",
    "۽": " اور ",
}

# Shahmukhi Punjabi uses standard Urdu/Arabic letters almost entirely.
# Only Gurmukhi leakage needs stripping (shouldn't appear but handle defensively).
_PUNJABI_CHAR_MAP: dict[str, str] = {
    # Defensive normalization when Sindhi letters leak into Punjabi output.
    "ڄ": "ج",
    "ڃ": "ج",
    "ٻ": "ب",
    "ڪ": "ک",
    "ڳ": "گ",
    "ڱ": "ن",
    "ڻ": "ن",
    "ٺ": "ٹ",
    "ڦ": "ف",
    "ڏ": "ڈ",
    "ڊ": "ڈ",
    "ڍ": "ڈ",
    "ڌ": "د",
    # strip any Gurmukhi that leaked through (U+0A00–U+0A7F)
}

_REGIONAL_CHAR_MAPS: dict[str, dict[str, str]] = {
    "pashto": _PASHTO_CHAR_MAP,
    "sindhi": _SINDHI_CHAR_MAP,
    "punjabi": _PUNJABI_CHAR_MAP,
}


def _apply_char_map(text: str, char_map: dict[str, str]) -> str:
    """Apply a deterministic character substitution map to text."""
    if not text or not char_map:
        return text
    for src, dst in char_map.items():
        text = text.replace(src, dst)
    # Also strip any Gurmukhi/Devanagari that might have leaked
    import re
    text = re.sub(r'[\u0A00-\u0A7F\u0900-\u097F]+', '', text)
    return text.strip()


def _normalize_punjabi_shahmukhi(text: str) -> str:
    """Best-effort cleanup when Sindhi words leak into Punjabi output."""
    if not text:
        return text
    out = text
    for src, dst in _PUNJABI_NORMALIZE_MAP:
        out = out.replace(src, dst)
    return " ".join(out.split())


def _translate_answer_to_regional(
    answer_text: str,
    target_lang: str,
    primary_model: str,
    log: Callable[[str], None],
    *,
    source_lang_name: str,
    original_question: str = "",
    english_question: str = "",
) -> str:
    """
    Translate a clean English (or Urdu) expert answer into Sindhi / Punjabi /
    Pashto with strict Arabic-script enforcement.

    `original_question` is the raw transcript or typed user text (often
    Perso-Arabic). `english_question` is the normalized English question when
    available, so the model can align tone and facts.
    Falls back to `answer_text` if the translation looks garbled.
    """
    target_lang_readable = _normalize_target_lang(target_lang)
    script_rule = _SCRIPT_RULES.get(target_lang.lower(), "Arabic/Perso-Arabic script")
    num_predict = getattr(config, "OLLAMA_NUM_PREDICT_CHAT", 512)

    context_chunks: list[str] = []
    if (original_question or "").strip():
        context_chunks.append(
            f"User question (original {target_lang_readable} script / transcript): "
            f"\"{original_question.strip()}\""
        )
    if (english_question or "").strip():
        context_chunks.append(
            f"English understanding of the question: \"{english_question.strip()}\""
        )
    context_note = ""
    if context_chunks:
        context_note = (
            "\n\nContext:\n"
            + "\n".join(context_chunks)
            + f"\nEnsure the {target_lang_readable} translation fully answers that question "
            "and preserves every fact from the source answer below."
        )

    system_prompt = (
        f"You are a professional translator from {source_lang_name} to {target_lang_readable}. "
        f"Translate the {source_lang_name} agricultural answer below into natural, conversational "
        f"{target_lang_readable} suitable for Pakistani farmers. "
        f"MANDATORY SCRIPT: Write EXCLUSIVELY in {script_rule} "
        f"Do NOT transliterate into Roman/Latin. "
        f"Preserve ALL facts, crop names, and recommendations from the {source_lang_name} text exactly. "
        f"Do NOT add any explanation, prefix, or language label. "
        f"Output ONLY the translated {target_lang_readable} text. "
        f"Do NOT output Chinese or any other language."
        f"{context_note}"
    )
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{source_lang_name} answer to translate:\n{answer_text}"},
    ]

    log(f"[2-step] Translating {source_lang_name} → {target_lang_readable}…")
    last = ""
    for m in _models_to_try(primary_model):
        try:
            last = _ollama_chat(
                messages=msgs,
                model=m,
                temperature=0.0,   # fully deterministic for translation
                think=False,
                num_predict=num_predict,
            )
            if last:
                break
        except BaseException as e:
            log(f"[2-step] Translation error ({m}): {e}")
            continue

    if not last:
        log(f"[2-step] Translation returned empty; falling back to {source_lang_name} answer.")
        return answer_text

    result = _strip_hallucinations(_strip_qwen_thinking_blocks(last))
    if not result:
        return answer_text

    # Quality gate: if the translation is drastically shorter than the
    # source (< 30 % of word count), it likely got truncated/garbled.
    src_words = len(answer_text.split())
    result_words = len(result.split())
    if src_words >= 4 and result_words < max(2, src_words * 0.30):
        log(
            f"[2-step] Quality gate failed ({result_words} words vs {src_words} source words). "
            f"Falling back to {source_lang_name} answer."
        )
        return answer_text

    # Punjabi-specific guard: if Sindhi-heavy letters dominate, force one corrective rewrite.
    if target_lang.lower() == "punjabi":
        sindhi_hits = sum(1 for ch in result if ch in _SINDHI_HEAVY_CHARS)
        sindhi_token_hits = sum(1 for token in _SINDHI_HEAVY_TOKENS if token in result)
        if sindhi_hits >= 2 or sindhi_token_hits >= 1:
            log(
                "[2-step] Punjabi guard: Sindhi-heavy characters detected. "
                "Retrying with stricter Shahmukhi Punjabi instruction."
            )
            retry_system = (
                "You are a strict translator. Rewrite the text into Punjabi in Shahmukhi script used in Pakistan. "
                "Do NOT write Sindhi. Avoid Sindhi-specific letters/tokens such as ڄ ڃ ٻ ڪ ڳ ڱ ڻ ٺ ڦ ڏ ڊ ڍ ڌ ۾ ۽ آهي. "
                "Use natural Punjabi vocabulary in Shahmukhi. Output only the final Punjabi text."
            )
            retry_msgs = [
                {"role": "system", "content": retry_system},
                {"role": "user", "content": f"Rewrite this in Punjabi Shahmukhi only:\n{result}"},
            ]
            retry = ""
            for m in _models_to_try(primary_model):
                try:
                    retry = _ollama_chat(
                        messages=retry_msgs,
                        model=m,
                        temperature=0.0,
                        think=False,
                        num_predict=num_predict,
                    )
                    if retry:
                        break
                except BaseException as e:
                    log(f"[2-step] Punjabi rewrite error ({m}): {e}")
                    continue
            cleaned_retry = _strip_hallucinations(_strip_qwen_thinking_blocks(retry))
            if cleaned_retry:
                retry_hits = sum(1 for ch in cleaned_retry if ch in _SINDHI_HEAVY_CHARS)
                retry_token_hits = sum(1 for token in _SINDHI_HEAVY_TOKENS if token in cleaned_retry)
                if (retry_hits < sindhi_hits) and (retry_token_hits <= sindhi_token_hits):
                    result = cleaned_retry
                else:
                    log("[2-step] Punjabi rewrite did not improve script quality; keeping previous output.")

    if target_lang.lower() == "punjabi":
        return _normalize_punjabi_shahmukhi(result)
    return result


def _translate_regional_to_english(
    regional_text: str,
    source_lang: str,
    primary_model: str,
    log: Callable[[str], None],
) -> str:
    """
    Translate user input in a regional language (Sindhi / Punjabi / Pashto)
    into clear English so the main reasoning LLM can understand it with
    maximum accuracy and depth.
    """
    source_lang_readable = _normalize_target_lang(source_lang)
    num_predict = getattr(config, "OLLAMA_NUM_PREDICT_CHAT", 512)

    script_hints = {
        "sindhi": "Sindhi (written in Perso-Arabic script used in Pakistan)",
        "punjabi": "Punjabi (written in Shahmukhi/Perso-Arabic script used in Pakistan)",
        "pashto": "Pashto (written in Pashto Arabic script)",
    }
    script_desc = script_hints.get(source_lang.strip().lower(), source_lang_readable)

    system_prompt = (
        f"You are a professional translator from {source_lang_readable} to English. "
        f"Translate the {script_desc} text below into standard, clear, natural English. "
        "The input may be from automatic speech recognition using Urdu-like spelling for "
        f"{source_lang_readable} sounds—infer the farmer's intended agricultural question. "
        "Keep the focus on agricultural, farming, and crop/soil context. "
        "Preserve the original meaning, intent, crop names, and specific details perfectly. "
        "Do NOT add any commentary, notes, translation labels, or conversational preambles. "
        "Output ONLY the translated English text."
    )
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": regional_text},
    ]

    src_key = source_lang.strip().lower()
    log(f"[{src_key}-to-english] Translating {source_lang_readable} query to English...")
    last = ""
    for m in _models_to_try(primary_model):
        try:
            last = _ollama_chat(
                messages=msgs,
                model=m,
                temperature=0.0,
                think=False,
                num_predict=num_predict,
            )
            if last:
                break
        except BaseException as e:
            log(f"[{src_key}-to-english] Translation error ({m}): {e}")
            continue

    if not last:
        log(f"[{src_key}-to-english] Translation returned empty; falling back to original text.")
        return regional_text

    result = _strip_hallucinations(_strip_qwen_thinking_blocks(last))
    if not result:
        return regional_text

    log(f"[{src_key}-to-english] Translated text: '{result}'")
    return result


# Keep backward-compatible alias
_translate_sindhi_to_english = _translate_regional_to_english


def translate_text(
    text: str,
    target_lang: str,
    *,
    thinking: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    extra_context: str | None = None,
    from_audio: bool = False,
) -> str:
    """
    Agricultural expert reply via local Ollama (Qwen, etc.).

    For Sindhi, Punjabi, and Pashto a three-step pipeline is used:
      Step 1 – Translate the user's question to English (including speech
               transcribed in Perso-Arabic as the regional language).
      Step 2 – Generate the expert answer in English.
      Step 3 – Translate that English answer into the target regional language
               using a dedicated strict translation call.

    `thinking` is ignored: the server always receives ``think=false``.
    """
    _ = thinking

    if not text:
        return ""

    target_lang_readable = _normalize_target_lang(target_lang)
    primary_model = getattr(config, "OLLAMA_MODEL", "gemma3:4b")
    target_lang_lower = target_lang_readable.lower()

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
    _log(
        f"Target language: {target_lang_readable} | Model: {primary_model} | "
        f"Pipeline: {'regional-via-English' if target_lang_lower in _TWO_STEP_LANGUAGES else 'direct'}"
    )

    num_predict = getattr(config, "OLLAMA_NUM_PREDICT_CHAT", 512)

    # ------------------------------------------------------------------
    # REGIONAL PIPELINE (Sindhi / Punjabi / Pashto)
    # Step 1: regional or Perso-Arabic STT text → English question
    # Step 2: generate answer in English
    # Step 3: translate English answer → target regional language
    # ------------------------------------------------------------------
    if target_lang_lower in _TWO_STEP_LANGUAGES:
        english_system = (
            "You are an Agricultural Expert named 'AI Agriculture Assistant'. "
            "Answer the user's agricultural question in simple English. "
            "Speak in a conversational tone suitable for farmers. "
            "Keep your response short, simple, and direct—ideally 2 to 4 lines. "
            "CRITICAL: Do NOT output any chat markers like 'Human:' or 'Assistant:'. "
            "Provide ONLY your direct answer in English."
        )
        convo_rules = (
            "Talk like a real person: natural, brief, no meta-commentary. "
            "Do NOT use chain-of-thought, hidden reasoning, or any <think> / thinking tags. "
            "Answer immediately in English only."
        )
        if extra_context:
            convo_rules += (
                f"\n\nCONTEXT INFO: {extra_context}\n"
                "Use this context to give specific advice (e.g. best crop for this location/month) if relevant. "
                "CRITICAL: If the user explicitly mentions a different location, region, city, or month in their question, "
                "ignore the automatically provided CONTEXT INFO's location/month and prioritize the user's requested location/month instead. "
                "Do NOT mix the two locations, and do NOT mention the context location/month (e.g. do NOT say 'Karachi' or 'May' in your response) "
                "if the user's question is about another place (such as Sahiwal, Punjab, etc.) or another time."
            )

        _log(
            f"[regional] Step 1: {target_lang_readable} (incl. audio) → English question…"
            + (" [from_audio]" if from_audio else "")
        )
        english_query = _translate_regional_to_english(text, target_lang_lower, primary_model, _log)
        if not (english_query or "").strip():
            english_query = text
            _log("[regional] English question empty; using raw input as fallback.")

        user_content = f"Answer this agricultural query in English:\n{english_query}"

        english_msgs = [
            {"role": "system", "content": english_system + "\n\n" + convo_rules},
            {"role": "user", "content": user_content},
        ]

        try:
            _log("[regional] Step 2: Generating English answer…")
            english_answer = _ollama_try_models(
                messages_factory=lambda _m: english_msgs,
                primary_model=primary_model,
                temperature=0.4,
                think=False,
                num_predict=num_predict,
                log=_log,
                progress_callback=None,  # don't surface intermediate result
            )

            english_answer = _strip_hallucinations(_strip_qwen_thinking_blocks(english_answer))

            if not english_answer:
                return "[معذرت، مجھے آپ کی بات سمجھ نہیں آئی۔ براہ کرم دوبارہ کوشش کریں۔]"

            _log(f"[regional] English answer ({len(english_answer)} chars): {english_answer[:80]}…")

            # Step 3: English → regional (strict script)
            regional_answer = _translate_answer_to_regional(
                english_answer,
                target_lang_lower,
                primary_model,
                _log,
                source_lang_name="English",
                original_question=text,
                english_question=english_query,
            )

            if progress_callback:
                try:
                    progress_callback(regional_answer)
                except Exception:
                    pass

            return _sanitize_translated_text(regional_answer)

        except Exception as e:
            _log(f"[regional] Error: {e}")
            raise

    # ------------------------------------------------------------------
    # DIRECT PIPELINE for Urdu / English / Balochi / other
    # ------------------------------------------------------------------
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
            f"CRITICAL SCRIPT RULE: You MUST write your response EXCLUSIVELY in the Pakistani Arabic/Urdu script. "
            f"DO NOT use Gurmukhi. DO NOT use Devanagari. DO NOT use Latin/English script. "
            f"Even if the user's question is in a different script, respond using only Arabic/Urdu script. "
            "Keep your response short, simple, and direct—ideally 2 to 4 lines. "
            f"If you absolutely cannot answer in {target_lang_readable}, respond in simple Urdu instead. "
            "CRITICAL: Do NOT output any chat markers like 'Human:' or 'Assistant:'. "
            f"Do NOT start your response with '{target_lang_readable}:' or any language prefix. "
            "Do NOT output Chinese. Provide ONLY your direct answer in the requested language."
        )

    convo_rules = (
        "Talk like a real person: natural, brief, no meta-commentary. "
        "Do NOT use chain-of-thought, hidden reasoning, or any <think> / thinking tags. "
        "Answer immediately in the user-facing language only."
    )
    if extra_context:
        convo_rules += (
            f"\n\nCONTEXT INFO: {extra_context}\n"
            "Use this context to give specific advice (e.g. best crop for this location/month) if relevant. "
            "CRITICAL: If the user explicitly mentions a different location, region, city, or month in their question, "
            "ignore the automatically provided CONTEXT INFO's location/month and prioritize the user's requested location/month instead. "
            "Do NOT mix the two locations, and do NOT mention the context location/month (e.g. do NOT say 'Karachi' or 'May' in your response) "
            "if the user's question is about another place (such as Sahiwal, Punjab, etc.) or another time."
        )

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
    primary_model = getattr(config, "OLLAMA_MODEL", "gemma3:4b")
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
    Convert regional Arabic-script text (Sindhi / Punjabi / Pashto) into
    standard Urdu letters so an Urdu TTS voice (ur-PK-UzmaNeural, etc.)
    can pronounce every character correctly.

    This uses a DETERMINISTIC Unicode character map — no LLM call — which
    guarantees the spoken audio matches the displayed text exactly and
    adds zero latency.
    """
    if not text:
        return text

    lang_key = (target_lang or "").strip().lower()
    char_map = _REGIONAL_CHAR_MAPS.get(lang_key)
    if char_map is None:
        # Unknown language: just strip any Gurmukhi/Devanagari that leaked
        import re
        return re.sub(r'[\u0A00-\u0A7F\u0900-\u097F]+', '', text).strip() or text

    result = _apply_char_map(text, char_map)
    _LAST_TRANSLATE_LOGS.append(
        f"[TTS char-map] {target_lang}: {len(text)} → {len(result)} chars"
    )
    return result if result.strip() else text


__all__ = ["translate_text", "get_last_translate_logs", "normalize_transcript_for_display", "transliterate_regional_for_tts"]
