import os
import io
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import streamlit as st

import config
import stt
import translate
import tts


st.set_page_config(
    page_title="Multilingual Speech-to-Speech Translator",
    page_icon="🌍",
    layout="wide",
)


def _record_from_mic(seconds: int) -> str:
    """Record audio from system microphone, save to temp WAV, and return path."""
    samplerate = config.SAMPLE_RATE
    st.info(f"Recording for {seconds} seconds...")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()

    out_path = config.TEMP_INPUT_WAV_PATH
    sf.write(out_path, audio.flatten(), samplerate)
    return out_path


def main() -> None:
    st.title("🌍 Multilingual Speech-to-Speech Translator")
    st.markdown(
        "Local **Whisper** (STT) + **Gemini 2.5 Flash** (translation) + **Piper** (TTS)."
    )

    # Sidebar configuration
    st.sidebar.header("Settings")
    language_options = ["urdu", "hindi", "english", "punjabi", "sindhi"]
    default_index = language_options.index(config.TARGET_LANGUAGE) if config.TARGET_LANGUAGE in language_options else 0
    target_lang = st.sidebar.selectbox("Target language", language_options, index=default_index)
    record_seconds = st.sidebar.slider("Microphone record duration (seconds)", 3, 30, config.DEFAULT_RECORD_SECONDS)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Piper voices")
    st.sidebar.caption("These filenames must exist in the `voices/` folder (or be absolute paths).")
    for lang_key, model in list(config.PIPER_VOICE_MODELS.items()):
        new_model = st.sidebar.text_input(f"{lang_key} voice model", value=model, key=f"voice_{lang_key}")
        config.PIPER_VOICE_MODELS[lang_key] = new_model

    tab_mic, tab_upload = st.tabs(["🎙️ Microphone", "📁 Upload Audio File"])

    with tab_mic:
        st.subheader("Record from Microphone")
        st.caption("Use your system default microphone. Whisper will auto-detect the spoken language.")

        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Start Recording"):
                with st.spinner("Recording and processing..."):
                    audio_path = _record_from_mic(record_seconds)
                    st.success(f"Audio recorded to `{audio_path}`")

                    # Show raw audio
                    with open(audio_path, "rb") as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format="audio/wav")

                    # STT
                    text, detected_lang_code = stt.speech_to_text(audio_path)
                    detected_lang_name = stt.map_whisper_lang_to_name(detected_lang_code)

                    # Display STT results
                    st.markdown("#### Detected language")
                    st.write(f"{detected_lang_name} (`{detected_lang_code}`)")

                    st.markdown("#### Transcribed text")
                    st.text_area("Original text", value=text, height=150)

                    # Translation
                    st.markdown("#### Translated text")
                    translated = translate.translate_text(text, target_lang=target_lang)
                    st.text_area("Translation", value=translated, height=150)

                    # TTS
                    if translated:
                        out_path = tts.text_to_speech(translated, lang=target_lang, output_path=config.OUTPUT_WAV_PATH)
                        with open(out_path, "rb") as f:
                            out_bytes = f.read()
                        st.markdown("#### Output audio")
                        st.audio(out_bytes, format="audio/wav")
        with col2:
            st.info(
                "Click **Start Recording** to capture from your microphone, then the app will transcribe, "
                "translate, and synthesize speech in the selected target language."
            )

    with tab_upload:
        st.subheader("Upload Audio File")
        uploaded_file = st.file_uploader("Upload audio file (WAV recommended)", type=["wav", "mp3", "m4a"])

        if uploaded_file is not None:
            # Save uploaded audio to temp path
            temp_path = config.TEMP_INPUT_WAV_PATH
            # Read file-like into bytes then write out
            raw = uploaded_file.read()
            with open(temp_path, "wb") as f:
                f.write(raw)

            st.markdown("#### Input audio")
            st.audio(raw)

            if st.button("Process Uploaded Audio"):
                with st.spinner("Transcribing, translating, and synthesizing..."):
                    text, detected_lang_code = stt.speech_to_text(temp_path)
                    detected_lang_name = stt.map_whisper_lang_to_name(detected_lang_code)

                    st.markdown("#### Detected language")
                    st.write(f"{detected_lang_name} (`{detected_lang_code}`)")

                    st.markdown("#### Transcribed text")
                    st.text_area("Original text", value=text, height=150)

                    translated = translate.translate_text(text, target_lang=target_lang)
                    st.markdown("#### Translated text")
                    st.text_area("Translation", value=translated, height=150)

                    if translated:
                        out_path = tts.text_to_speech(translated, lang=target_lang, output_path=config.OUTPUT_WAV_PATH)
                        with open(out_path, "rb") as f:
                            out_bytes = f.read()
                        st.markdown("#### Output audio")
                        st.audio(out_bytes, format="audio/wav")


if __name__ == "__main__":
    main()


