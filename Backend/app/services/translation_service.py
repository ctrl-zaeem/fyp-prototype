from __future__ import annotations

from app.services.ollama_service import ollama_service

LANGUAGE_ALIASES = {
    "en": "english",
    "ur": "urdu",
    "pa": "punjabi",
    "ps": "pashto",
    "sd": "sindhi",
}


class TranslationService:
    def normalize_language(self, language: str) -> str:
        key = (language or "").strip().lower()
        return LANGUAGE_ALIASES.get(key, key or "english")

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text:
            return ""
        source = self.normalize_language(source_lang)
        target = self.normalize_language(target_lang)
        if source == target:
            return text

        system = (
            "You are a strict translator. Return only the translated text. "
            "Do not add explanations, role markers, or chain-of-thought."
        )
        prompt = f"Translate the following text from {source} to {target}:\n\n{text}"
        return ollama_service.generate(prompt=prompt, system_prompt=system, temperature=0.0, num_predict=300)


translation_service = TranslationService()

