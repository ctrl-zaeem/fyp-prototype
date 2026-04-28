import os
import io
import time
import threading
import queue

import numpy as np
import sounddevice as sd
import soundfile as sf
import streamlit as st

import config
import stt
import translate
import tts


st.set_page_config(
    page_title="AI Agriculture Assistant |  اے آئی زرعی معاون",
    page_icon="",
    layout="wide",
)

# Custom CSS for greenish theme and text display
st.markdown("""
    <style>
    /* Main background - greenish theme */
    .stApp {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 50%, #a5d6a7 100%);
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #f1f8e9;
        padding: 2rem;
        border-radius: 10px;
    }
    
    /* Urdu text styling - simple white box with black text */
    .urdu-text {
        font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', 'Al Qalam Taj Nastaleeq', 'Nafees Web Naskh', Arial, sans-serif;
        font-size: 24px !important;
        line-height: 1.8;
        direction: rtl;
        text-align: right;
        padding: 20px;
        background: #ffffff;
        color: #000000;
        border-radius: 8px;
        border: 1px solid #c8e6c9;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    
    /* LTR text styling - simple white box with black text */
    .ltr-text {
        font-family: Arial, sans-serif;
        font-size: 24px !important;
        line-height: 1.8;
        direction: ltr;
        text-align: left;
        padding: 20px;
        background: #ffffff;
        color: #000000;
        border-radius: 8px;
        border: 1px solid #c8e6c9;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', 'Al Qalam Taj Nastaleeq', 'Nafees Web Naskh', Arial, sans-serif !important;
        font-size: 22px !important;
        line-height: 1.8 !important;
        direction: rtl !important;
        text-align: right !important;
        padding: 15px !important;
        background: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Section headers */
    h3, h4 {
        color: #2e7d32;
        margin-top: 20px;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s;
        background-color: #4caf50;
        color: white;
    }
    
    .stButton > button:hover {
        background-color: #45a049;
    }
    
    /* Recording status indicator */
    .recording-status {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-weight: bold;
        text-align: center;
    }
    
    .recording-active {
        background-color: #ff4444;
        color: white;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    
    /* Success messages */
    .stSuccess {
        background-color: #c8e6c9;
    }
    </style>
""", unsafe_allow_html=True)


def _record_from_mic(seconds: int) -> str:
    """Record audio from system microphone, save to temp WAV, and return path."""
    samplerate = config.SAMPLE_RATE
    st.info(f"Recording for {seconds} seconds...")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()

    out_path = config.TEMP_INPUT_WAV_PATH
    sf.write(out_path, audio.flatten(), samplerate)
    return out_path


def _record_continuous(samplerate: int, stop_event: threading.Event, audio_queue: queue.Queue):
    """Record audio continuously until stop_event is set."""
    audio_data = []
    
    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_data.append(indata.copy())
    
    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32", callback=callback):
        while not stop_event.is_set():
            time.sleep(0.1)
    
    if audio_data:
        audio_queue.put(np.concatenate(audio_data, axis=0))


# ---------------------------------------------------------------------------
# Language labels shown in the language selection screen
# ---------------------------------------------------------------------------
LANGUAGE_OPTIONS = {
    "english":  {"label": "English",         "flag": "🇬🇧"},
    "urdu":     {"label": "اردو (Urdu)",      "flag": "🇵🇰"},
    "sindhi":   {"label": "سنڌي (Sindhi)",   "flag": "🌊"},
    "punjabi":  {"label": "ਪੰਜਾਬੀ (Punjabi)", "flag": "🌾"},
    "pashto":   {"label": "پښتو (Pashto)",   "flag": "🏔️"},
    "balochi":  {"label": "بلوچی (Balochi)",  "flag": "🌄"},
}

# Languages that use RTL Perso-Arabic script for display
RTL_LANGUAGES = {"urdu", "punjabi", "sindhi", "pashto", "balochi"}


def _show_language_selection() -> None:
    """Render a full-page language selector and block further rendering."""
    st.markdown("""
        <style>
        .lang-title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 800;
            color: #2e7d32;
            margin-bottom: 0.2rem;
        }
        .lang-subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #555;
            margin-bottom: 2rem;
        }
        .lang-card {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: #fff;
            border: 2px solid #c8e6c9;
            border-radius: 16px;
            padding: 28px 16px;
            font-size: 1.15rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 2px 8px rgba(76,175,80,0.08);
            min-height: 110px;
        }
        .lang-card:hover {
            border-color: #4caf50;
            box-shadow: 0 4px 16px rgba(76,175,80,0.25);
            transform: translateY(-2px);
        }
        .lang-flag { font-size: 2.4rem; margin-bottom: 8px; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="lang-title">🌿 AI Agriculture Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lang-subtitle">Please select your preferred language to continue<br>'
        'براہ کرم جاری رکھنے کے لیے اپنی زبان منتخب کریں</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    for idx, (lang_key, meta) in enumerate(LANGUAGE_OPTIONS.items()):
        with cols[idx % 3]:
            st.markdown(
                f'<div class="lang-card"><div class="lang-flag">{meta["flag"]}</div>{meta["label"]}</div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Select {meta['label']}", key=f"lang_btn_{lang_key}", use_container_width=True):
                st.session_state.selected_language = lang_key
                st.rerun()


def main() -> None:
    # ------------------------------------------------------------------
    # Language gate: show selector until the user picks a language
    # ------------------------------------------------------------------
    if "selected_language" not in st.session_state:
        _show_language_selection()
        st.stop()   # Nothing below renders until a language is chosen

    target_lang: str = st.session_state.selected_language

    # Allow changing language via sidebar
    st.sidebar.header("⚙️ Settings")
    lang_labels = {k: v["label"] for k, v in LANGUAGE_OPTIONS.items()}
    current_label = lang_labels[target_lang]

    # Qwen3 "thinking" toggle (reasoning on/off)
    if "qwen_thinking" not in st.session_state:
        st.session_state.qwen_thinking = False
    st.session_state.qwen_thinking = st.sidebar.toggle(
        "🧠 Qwen3 Thinking (Reasoning)",
        value=st.session_state.qwen_thinking,
        help="ON: model may use internal reasoning (then hidden). OFF: forces direct answers without thinking tags.",
    )

    new_label = st.sidebar.selectbox(
        "🌐 Language / زبان",
        list(lang_labels.values()),
        index=list(lang_labels.values()).index(current_label),
    )
    # Reverse-lookup from label → key
    new_lang_key = next(k for k, v in LANGUAGE_OPTIONS.items() if v["label"] == new_label)
    if new_lang_key != target_lang:
        st.session_state.selected_language = new_lang_key
        st.rerun()

    flag = LANGUAGE_OPTIONS[target_lang]["flag"]
    st.title(f"AI Agriculture Assistant | اے آئی زرعی معاون")
    st.markdown(
        f"Talking in: {flag} **{lang_labels[target_lang]}**  ·  "
        "Local **Whisper** (STT) + **Ollama Qwen3:8B** (AI) + **Google TTS** (TTS)."
    )

    tab_mic, tab_upload, tab_text = st.tabs(["🎙️ Microphone", "📂 Upload Audio File", "⌨️ Text Input"])

    with tab_mic:
        st.subheader("Record from Microphone")
        st.caption("Use your system default microphone. Whisper will transcribe in the selected language.")

        # Initialize session state for recording
        if 'recording' not in st.session_state:
            st.session_state.recording = False
        if 'stop_recording' not in st.session_state:
            st.session_state.stop_recording = threading.Event()
        if 'audio_queue' not in st.session_state:
            st.session_state.audio_queue = queue.Queue()
        if 'recording_thread' not in st.session_state:
            st.session_state.recording_thread = None
        if 'processed_audio_path' not in st.session_state:
            st.session_state.processed_audio_path = None

        col1, col2 = st.columns([1, 2])
        with col1:
            # Recording controls
            button_col1, button_col2 = st.columns(2)
            
            with button_col1:
                if st.button("START", type="primary", disabled=st.session_state.recording):
                    # Clear previous results and processing state
                    st.session_state.processed_audio_path = None
                    st.session_state.processing_step = None
                    if 'transcribed_text' in st.session_state:
                        del st.session_state.transcribed_text
                    if 'translated_text' in st.session_state:
                        del st.session_state.translated_text
                    if 'translated_text_display' in st.session_state:
                        del st.session_state.translated_text_display
                    if 'translated_text_audio' in st.session_state:
                        del st.session_state.translated_text_audio
                    if 'output_audio_path' in st.session_state:
                        del st.session_state.output_audio_path
                    
                    st.session_state.recording = True
                    st.session_state.stop_recording.clear()
                    st.session_state.audio_queue = queue.Queue()
                    st.session_state.recording_thread = threading.Thread(
                        target=_record_continuous,
                        args=(config.SAMPLE_RATE, st.session_state.stop_recording, st.session_state.audio_queue)
                    )
                    st.session_state.recording_thread.start()
                    st.rerun()
            
            with button_col2:
                if st.button("STOP", disabled=not st.session_state.recording):
                    st.session_state.recording = False
                    st.session_state.stop_recording.set()
                    if st.session_state.recording_thread:
                        st.session_state.recording_thread.join(timeout=3)
                    
                    # Wait a bit for audio to be queued
                    time.sleep(0.5)
                    
                    # Process the recorded audio
                    if not st.session_state.audio_queue.empty():
                        audio_data = st.session_state.audio_queue.get()
                        if audio_data is not None and len(audio_data) > 0:
                            out_path = config.TEMP_INPUT_WAV_PATH
                            sf.write(out_path, audio_data.flatten(), config.SAMPLE_RATE)
                            st.session_state.processed_audio_path = out_path
                            st.rerun()
                        else:
                            st.error("No audio data captured. Please try recording again.")
                    else:
                        st.warning("Recording stopped, but no audio was captured. Please try again.")
            
            # Recording status
            if st.session_state.recording:
                st.markdown(
                    '<div class="recording-status recording-active"> RECORDING... Click Stop when finished</div>',
                    unsafe_allow_html=True
                )
            
            # Process audio if available
            if st.session_state.processed_audio_path and os.path.exists(st.session_state.processed_audio_path):
                audio_path = st.session_state.processed_audio_path
                st.success("Audio recorded successfully")
                
                # Show raw audio
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/wav")
                
                # Initialize processing state
                if 'processing_step' not in st.session_state:
                    st.session_state.processing_step = None
                
                # Process audio step by step
                if st.button("PROCESS"):
                    st.session_state.processing_step = 'transcribing'
                    st.rerun()
                
                # Step 1: Transcribing
                if st.session_state.processing_step == 'transcribing':
                    with st.spinner(" Transcribing audio..."):
                        text, detected_lang_code = stt.speech_to_text(audio_path, target_lang=target_lang)
                        detected_lang_name = target_lang
                        
                        # Store results in session state
                        st.session_state.detected_lang = detected_lang_name
                        st.session_state.detected_lang_code = detected_lang_code
                        st.session_state.transcribed_text = text
                        st.session_state.processing_step = 'translating'
                        st.rerun()
                
                # Step 2: Translation (only if transcription is done)
                if st.session_state.processing_step == 'translating' and 'transcribed_text' in st.session_state:
                    live_box = st.empty()
                    with st.spinner("Translating (streaming from Ollama)…"):
                        def _progress(partial: str) -> None:
                            # Show partial text as it streams in
                            if target_lang in RTL_LANGUAGES:
                                live_box.markdown(
                                    f'<div class="urdu-text" dir="rtl">{partial}</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                live_box.markdown(
                                    f'<div class="ltr-text" dir="ltr">{partial}</div>',
                                    unsafe_allow_html=True,
                                )

                        translated = translate.translate_text(
                            st.session_state.transcribed_text,
                            target_lang=target_lang,
                            thinking=st.session_state.qwen_thinking,
                            progress_callback=_progress,
                        )
                        st.session_state.translated_text_display = translated
                        st.session_state.translated_text_audio = translated
                        st.session_state.processing_step = 'synthesizing'
                        st.rerun()
                
                # Step 3: TTS (only if translation is done)
                if (
                    st.session_state.processing_step == 'synthesizing'
                    and 'translated_text_audio' in st.session_state
                    and st.session_state.translated_text_audio
                ):
                    with st.spinner("Synthesizing speech..."):
                        out_path = tts.text_to_speech(st.session_state.translated_text_audio, lang=target_lang, output_path=config.OUTPUT_WAV_PATH)
                        st.session_state.output_audio_path = out_path
                        st.session_state.processing_step = 'complete'
                        st.rerun()
        
        with col2:
            st.info(
                "Click **Start Recording** to begin capturing audio from your microphone. "
                "Click **Stop Recording** when finished, then click **Process Audio** to transcribe, "
                "translate, and synthesize speech in the selected target language."
            )
            
            # Display results incrementally as they become available
            if 'transcribed_text' in st.session_state and st.session_state.transcribed_text:
                st.markdown("---")
                st.markdown("### Results")
                
                # Step 1: Show transcribed text
                st.markdown("####  Step 1: Transcribed Text")
                st.write(f"**Selected Language:** {lang_labels[target_lang]}")
                
                transcribed_display = st.session_state.transcribed_text
                # Use RTL box when the selected language uses Perso-Arabic script
                if target_lang in RTL_LANGUAGES:
                    st.markdown(
                        f'<div class="urdu-text" dir="rtl">{transcribed_display}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="ltr-text" dir="ltr">{transcribed_display}</div>',
                        unsafe_allow_html=True
                    )
                
                # Step 2: Show translated text (if available)
                if 'translated_text_display' in st.session_state and st.session_state.translated_text_display:
                    st.markdown("####  Step 2: Translated Text")
                    translated_display = st.session_state.translated_text_display
                    # RTL box for all Perso-Arabic script languages
                    if target_lang in RTL_LANGUAGES:
                        st.markdown(
                            f'<div class="urdu-text" dir="rtl">{translated_display}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="ltr-text" dir="ltr">{translated_display}</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Step 3: Show output audio (if available)
                    if 'output_audio_path' in st.session_state and os.path.exists(st.session_state.output_audio_path):
                        st.markdown("####  Step 3: Output Audio")
                        with open(st.session_state.output_audio_path, "rb") as f:
                            out_bytes = f.read()
                        st.audio(out_bytes, format="audio/wav")
                        st.caption(f"TTS used: **{tts.get_last_tts_engine_info()}**")

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

            st.markdown("#### Input Audio")
            st.audio(raw)
            
            # Initialize processing state for upload tab
            if 'upload_processing_step' not in st.session_state:
                st.session_state.upload_processing_step = None
            
            # Process audio step by step
            if st.button(" Process Uploaded Audio", type="primary"):
                # Clear previous results when starting new processing
                st.session_state.upload_processing_step = None
                if 'upload_transcribed_text' in st.session_state:
                    del st.session_state.upload_transcribed_text
                if 'upload_translated_text' in st.session_state:
                    del st.session_state.upload_translated_text
                if 'upload_translated_text_display' in st.session_state:
                    del st.session_state.upload_translated_text_display
                if 'upload_translated_text_audio' in st.session_state:
                    del st.session_state.upload_translated_text_audio
                if 'upload_output_audio_path' in st.session_state:
                    del st.session_state.upload_output_audio_path
                st.session_state.upload_processing_step = 'transcribing'
                st.rerun()
            
            # Step 1: Transcribing
            if st.session_state.upload_processing_step == 'transcribing':
                with st.spinner(" Transcribing audio..."):
                    text, detected_lang_code = stt.speech_to_text(temp_path, target_lang=target_lang)
                    detected_lang_name = target_lang
                    
                    # Store in session state for upload tab
                    st.session_state.upload_detected_lang = detected_lang_name
                    st.session_state.upload_detected_lang_code = detected_lang_code
                    st.session_state.upload_transcribed_text = text
                    st.session_state.upload_processing_step = 'translating'
                    st.rerun()
            
            # Step 2: Translation (only if transcription is done)
            if st.session_state.upload_processing_step == 'translating' and 'upload_transcribed_text' in st.session_state:
                live_box = st.empty()
                with st.spinner(" Translating (streaming from Ollama)…"):
                    def _progress(partial: str) -> None:
                        if target_lang in RTL_LANGUAGES:
                            live_box.markdown(
                                f'<div class="urdu-text" dir="rtl">{partial}</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            live_box.markdown(
                                f'<div class="ltr-text" dir="ltr">{partial}</div>',
                                unsafe_allow_html=True,
                            )

                    translated = translate.translate_text(
                        st.session_state.upload_transcribed_text,
                        target_lang=target_lang,
                        thinking=st.session_state.qwen_thinking,
                        progress_callback=_progress,
                    )
                    st.session_state.upload_translated_text_display = translated
                    st.session_state.upload_translated_text_audio = translated
                    st.session_state.upload_processing_step = 'synthesizing'
                    st.rerun()
            
            # Step 3: TTS (only if translation is done)
            if (
                st.session_state.upload_processing_step == 'synthesizing'
                and 'upload_translated_text_audio' in st.session_state
                and st.session_state.upload_translated_text_audio
            ):
                with st.spinner(" Synthesizing speech..."):
                    out_path = tts.text_to_speech(st.session_state.upload_translated_text_audio, lang=target_lang, output_path=config.OUTPUT_WAV_PATH)
                    st.session_state.upload_output_audio_path = out_path
                    st.session_state.upload_processing_step = 'complete'
                    st.rerun()
            
            # Display results incrementally if available
            if 'upload_transcribed_text' in st.session_state and st.session_state.upload_transcribed_text:
                st.markdown("---")
                st.markdown("###  Results")
                
                # Step 1: Show transcribed text
                st.markdown("####  Step 1: Transcribed Text")
                st.write(f"**Selected Language:** {lang_labels[target_lang]}")
                
                transcribed_display = st.session_state.upload_transcribed_text
                # Use RTL box when the selected language uses Perso-Arabic script
                if target_lang in RTL_LANGUAGES:
                    st.markdown(
                        f'<div class="urdu-text" dir="rtl">{transcribed_display}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="ltr-text" dir="ltr">{transcribed_display}</div>',
                        unsafe_allow_html=True
                    )
                
                # Step 2: Show translated text (if available)
                if 'upload_translated_text_display' in st.session_state and st.session_state.upload_translated_text_display:
                    st.markdown("####  Step 2: Translated Text")
                    translated_display = st.session_state.upload_translated_text_display
                    # RTL box for all Perso-Arabic script languages
                    if target_lang in RTL_LANGUAGES:
                        st.markdown(
                            f'<div class="urdu-text" dir="rtl">{translated_display}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="ltr-text" dir="ltr">{translated_display}</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Step 3: Show output audio (if available)
                    if 'upload_output_audio_path' in st.session_state and os.path.exists(st.session_state.upload_output_audio_path):
                        st.markdown("####  Step 3: Output Audio")
                        with open(st.session_state.upload_output_audio_path, "rb") as f:
                            out_bytes = f.read()
                        st.audio(out_bytes, format="audio/wav")
                        st.caption(f"TTS used: **{tts.get_last_tts_engine_info()}**")


    with tab_text:
        st.subheader("Type or Paste Text")
        st.caption("Enter any text below. It will be translated to your selected language and synthesized as speech.")

        if "text_input_value" not in st.session_state:
            st.session_state.text_input_value = ""
        if "text_processing_step" not in st.session_state:
            st.session_state.text_processing_step = None

        col_txt1, col_txt2 = st.columns([1, 2])

        with col_txt1:
            user_text = st.text_area(
                "Enter text to translate",
                value=st.session_state.text_input_value,
                height=200,
                placeholder="Type your message here...",
                key="text_area_input",
            )

            if st.button("\U0001f504 Translate and Speak", type="primary", use_container_width=True):
                if user_text.strip():
                    st.session_state.text_input_value = user_text
                    for key in ["text_translated_display", "text_translated_audio", "text_output_audio_path"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.session_state.text_processing_step = "translating"
                    st.rerun()
                else:
                    st.warning("Please enter some text first.")

            if st.button("\U0001f5d1\ufe0f Clear", use_container_width=True):
                st.session_state.text_input_value = ""
                st.session_state.text_processing_step = None
                for key in ["text_translated_display", "text_translated_audio", "text_output_audio_path"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        with col_txt2:
            st.info(
                "Type or paste any text in the box on the left, then click "
                "**Translate and Speak**. The text will be translated to your chosen "
                "language and spoken aloud."
            )

            # Translation step
            if st.session_state.text_processing_step == "translating":
                live_box = st.empty()
                with st.spinner("\U0001f504 Translating (streaming from Ollama)…"):
                    def _progress(partial: str) -> None:
                        if target_lang in RTL_LANGUAGES:
                            live_box.markdown(
                                f'<div class="urdu-text" dir="rtl">{partial}</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            live_box.markdown(
                                f'<div class="ltr-text" dir="ltr">{partial}</div>',
                                unsafe_allow_html=True,
                            )

                    translated = translate.translate_text(
                        st.session_state.text_input_value,
                        target_lang=target_lang,
                        thinking=st.session_state.qwen_thinking,
                        progress_callback=_progress,
                    )
                    st.session_state.text_translated_display = translated
                    st.session_state.text_translated_audio = translated
                    st.session_state.text_processing_step = "synthesizing"
                    st.rerun()

            # TTS step
            if (
                st.session_state.text_processing_step == "synthesizing"
                and "text_translated_audio" in st.session_state
                and st.session_state.text_translated_audio
            ):
                with st.spinner("\U0001f50a Synthesizing speech..."):
                    out_path = tts.text_to_speech(
                        st.session_state.text_translated_audio,
                        lang=target_lang,
                        output_path=config.OUTPUT_WAV_PATH,
                    )
                    st.session_state.text_output_audio_path = out_path
                    st.session_state.text_processing_step = "complete"
                    st.rerun()

            # Show results
            if "text_translated_display" in st.session_state and st.session_state.text_translated_display:
                st.markdown("---")
                st.markdown("### Results")

                st.markdown("#### \U0001f4dd Original Text")
                st.markdown(
                    f'<div class="ltr-text" dir="ltr">{st.session_state.text_input_value}</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("#### \U0001f310 Translated Text")
                translated_display = st.session_state.text_translated_display
                if target_lang in RTL_LANGUAGES:
                    st.markdown(
                        f'<div class="urdu-text" dir="rtl">{translated_display}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="ltr-text" dir="ltr">{translated_display}</div>',
                        unsafe_allow_html=True,
                    )

                if "text_output_audio_path" in st.session_state and os.path.exists(
                    st.session_state.text_output_audio_path
                ):
                    st.markdown("#### \U0001f50a Output Audio")
                    with open(st.session_state.text_output_audio_path, "rb") as f:
                        out_bytes = f.read()
                    st.audio(out_bytes, format="audio/wav")
                    st.caption(f"TTS used: **{tts.get_last_tts_engine_info()}**")


if __name__ == "__main__":
    main()


