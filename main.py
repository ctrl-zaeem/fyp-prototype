"""
Main entry point for the multilingual speech-to-speech translation project.

Menu:
    1. Speak → Translate → Listen
    2. Translate existing audio file
    3. Change target language
    4. Change TTS voice (per language)
    5. Exit
"""

import os
from typing import Optional

import audio_utils
import config
import stt
import translate
import tts


def _prompt_int(prompt: str, min_val: int, max_val: int) -> int:
    while True:
        try:
            value = int(input(prompt).strip())
            if min_val <= value <= max_val:
                return value
        except ValueError:
            pass
        print(f"Please enter a number between {min_val} and {max_val}.")


def _choose_target_language() -> None:
    print("\n=== Change Target Language ===")
    languages = ["urdu", "hindi", "english", "punjabi", "sindhi"]
    for i, lang in enumerate(languages, start=1):
        print(f"{i}. {lang}")
    choice = _prompt_int("Select target language: ", 1, len(languages))
    config.TARGET_LANGUAGE = languages[choice - 1]
    print(f"Target language set to: {config.TARGET_LANGUAGE}")


def _change_tts_voice() -> None:
    print("\n=== Change TTS Voice ===")
    print("Current voice mappings (logical language -> model file):")
    for lang, model in config.PIPER_VOICE_MODELS.items():
        print(f"  {lang}: {model}")
    lang = input("Enter logical language to change (e.g., 'urdu', 'hindi', 'english'): ").strip().lower()
    if not lang:
        print("No language entered, aborting.")
        return
    new_model = input("Enter new Piper model filename or absolute path (leave blank to cancel): ").strip()
    if not new_model:
        print("No change made.")
        return
    config.PIPER_VOICE_MODELS[lang] = new_model
    print(f"Updated TTS voice for '{lang}' to '{new_model}'.")


def _pipeline_speak_translate_listen() -> None:
    print("\n=== Speak → Translate → Listen ===")
    seconds_str = input(f"Enter recording duration in seconds (default {config.DEFAULT_RECORD_SECONDS}): ").strip()
    try:
        seconds = int(seconds_str) if seconds_str else config.DEFAULT_RECORD_SECONDS
    except ValueError:
        seconds = config.DEFAULT_RECORD_SECONDS

    audio_path = stt.record_audio(duration=seconds, output_path=config.TEMP_INPUT_WAV_PATH)
    text, detected_lang_code = stt.speech_to_text(audio_path)
    detected_lang_name = stt.map_whisper_lang_to_name(detected_lang_code)

    print(f"\nDetected language (Whisper): {detected_lang_name} ({detected_lang_code})")
    print(f"Recognized text: {text}")

    if not text:
        print("No text recognized; aborting.")
        return

    target_lang = config.TARGET_LANGUAGE
    print(f"\nTranslating to target language: {target_lang}")
    translated = translate.translate_text(text, target_lang=target_lang)
    print(f"Translated text: {translated}")

    if not translated:
        print("No translated text returned; aborting TTS.")
        return

    output_audio_path = tts.text_to_speech(translated, lang=target_lang, output_path=config.OUTPUT_WAV_PATH)
    audio_utils.play_audio(output_audio_path)


def _pipeline_translate_existing_audio() -> None:
    print("\n=== Translate Existing Audio File ===")
    path = input("Enter path to existing audio file (WAV recommended): ").strip()
    if not path:
        print("No path entered.")
        return
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return

    text, detected_lang_code = stt.speech_to_text(path)
    detected_lang_name = stt.map_whisper_lang_to_name(detected_lang_code)

    print(f"\nDetected language (Whisper): {detected_lang_name} ({detected_lang_code})")
    print(f"Recognized text: {text}")

    if not text:
        print("No text recognized; aborting.")
        return

    target_lang = config.TARGET_LANGUAGE
    print(f"\nTranslating to target language: {target_lang}")
    translated = translate.translate_text(text, target_lang=target_lang)
    print(f"Translated text: {translated}")

    if not translated:
        print("No translated text returned; aborting TTS.")
        return

    output_audio_path = tts.text_to_speech(translated, lang=target_lang, output_path=config.OUTPUT_WAV_PATH)
    audio_utils.play_audio(output_audio_path)


def main() -> None:
    while True:
        print("\n==============================")
        print(" Multilingual S2S Translator ")
        print("==============================")
        print(f"Current target language: {config.TARGET_LANGUAGE}")
        print("1. Speak → Translate → Listen")
        print("2. Translate existing audio file")
        print("3. Change target language")
        print("4. Change TTS voice")
        print("5. Exit")

        choice = _prompt_int("Select an option: ", 1, 5)

        if choice == 1:
            _pipeline_speak_translate_listen()
        elif choice == 2:
            _pipeline_translate_existing_audio()
        elif choice == 3:
            _choose_target_language()
        elif choice == 4:
            _change_tts_voice()
        elif choice == 5:
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()


