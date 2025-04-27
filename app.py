import streamlit as st
import sounddevice as sd
import numpy as np
import wavio
import os
from datetime import datetime
import time
import threading
import gc
import torch
from conversational_rag import ConversationalRAG
from speech import WhisperSTT, HuggingFaceTTS

# Global flag for thread-safe recording control
stop_recording_flag = threading.Event()

# Page configuration
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Voice Assistant with RAG"
)

# Initialize all session state variables
def init_session_state():
    session_defaults = {
        'last_recording': None,
        'conversation_history': [],
        'recording_in_progress': False,
        'processed_recordings': set(),
        'current_recording': None,
        'recordings_list': [],
        'recording_counter': 0  # Counter to generate unique keys for buttons
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Cache heavy resources
@st.cache_resource
def load_rag():
    try:
        rag = ConversationalRAG(
            model_source="hf",
            embedding_source="hf",
            chroma_dir="chroma_store"
        )
        rag.initialize_vectorstore(data_dir="data")
        return rag
    except Exception as e:
        st.error(f"Failed to initialize RAG: {str(e)}")
        return None

@st.cache_resource
def load_stt():
    return WhisperSTT(model_size="base")

@st.cache_resource
def load_tts():
    return HuggingFaceTTS()

def record_audio(duration=5, fs=44100):
    """Improved audio recording with better state management"""
    global stop_recording_flag
    stop_recording_flag.clear()
    
    st.session_state.recording_in_progress = True
    recording_status = st.empty()
    recording_status.info("Recording... (Click 'Stop Recording' when finished)")
    
    audio_buffer = []
    
    def callback(indata, frames, time, status):
        if stop_recording_flag.is_set():
            raise sd.CallbackStop
        audio_buffer.append(indata.copy())
    
    # Create stop button
    stop_col = st.empty()
    stop_col.button("⏹️ Stop Recording", key=f"stop_{st.session_state.recording_counter}", 
                   on_click=lambda: stop_recording_flag.set())
    
    try:
        with sd.InputStream(
            samplerate=fs,
            channels=1,
            dtype='int16',
            callback=callback
        ) as stream:
            start_time = time.time()
            while (time.time() - start_time < duration) and not stop_recording_flag.is_set():
                time.sleep(0.1)
            
            stream.stop()
    except Exception as e:
        st.error(f"Recording error: {str(e)}")
    finally:
        # Clean up UI elements
        stop_col.empty()
        recording_status.empty()
        st.session_state.recording_in_progress = False
    
    if not audio_buffer:
        st.warning("No audio was recorded")
        return None
    
    # Process the recorded audio
    recording = np.concatenate(audio_buffer)
    os.makedirs("recordings", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"recordings/recording_{timestamp}.wav"
    
    try:
        wavio.write(filename, recording, fs, sampwidth=2)
        st.session_state.current_recording = filename
        if filename not in st.session_state.recordings_list:
            st.session_state.recordings_list.append(filename)
        st.session_state.last_recording = filename
        
        # Increment counter for unique button keys
        st.session_state.recording_counter += 1
        
        st.success("New recording saved successfully!")
        return filename
    except Exception as e:
        st.error(f"Error saving recording: {str(e)}")
        return None

def extract_helpful_answer(response_str):
    parts = response_str.split("Helpful Answer: ")
    if len(parts) > 1:
        return parts[-1].split("\n")[0].strip()
    return response_str

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Main UI
st.title("🎙️ Voice Assistant with RAG")

# Recording duration selector
duration = st.slider("Maximum recording duration (seconds)", min_value=3, max_value=10, value=5)

# Recording section - use fixed containers instead of empty placeholders
if not st.session_state.recording_in_progress:
    if st.button("🎤 Start New Recording", key=f"start_{st.session_state.recording_counter}"):
        filename = record_audio(duration=duration)
        st.rerun()

# Show current recording and processing
if st.session_state.current_recording and os.path.exists(st.session_state.current_recording):
    st.info(f"Current recording: {os.path.basename(st.session_state.current_recording)}")
    
    if not st.session_state.recording_in_progress:
        # Use a unique key for the process button
        if st.button("🔄 Process This Recording", key=f"process_{st.session_state.recording_counter}"):
            with st.spinner("Processing..."):
                try:
                    clear_gpu_memory()
                    
                    # STT
                    stt = load_stt()
                    user_input = stt.transcribe(st.session_state.current_recording)
                    st.subheader("You said:")
                    st.write(user_input)
                    
                    st.session_state.conversation_history.append(("user", user_input))
                    
                    # RAG
                    rag = load_rag()
                    if rag:
                        rag_response = rag.chat(user_input)
                        rag_response = extract_helpful_answer(rag_response['answer'])
                        st.subheader("Assistant response:")
                        st.write(rag_response)
                        
                        st.session_state.conversation_history.append(("assistant", rag_response))
                        
                        # TTS
                        with st.spinner("Generating speech..."):
                            tts = load_tts()
                            audio_array, sample_rate = tts.generate_speech(rag_response)
                            st.audio(audio_array, sample_rate=sample_rate)
                    
                    clear_gpu_memory()
                    st.session_state.processed_recordings.add(st.session_state.current_recording)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    clear_gpu_memory()

# Recordings list section
if st.session_state.recordings_list:
    st.subheader("Previous Recordings")
    for i, rec in enumerate(st.session_state.recordings_list):
        if os.path.exists(rec):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{i+1}. {os.path.basename(rec)}")
            with col2:
                if st.button("Load", key=f"load_{i}"):
                    st.session_state.current_recording = rec
                    st.rerun()

# Display conversation history
if st.session_state.conversation_history:
    st.subheader("Conversation History")
    for i, (speaker, text) in enumerate(st.session_state.conversation_history):
        if speaker == "user":
            st.markdown(f"**You ({i//2 + 1}):** {text}")
        else:
            st.markdown(f"**Assistant ({i//2 + 1}):** {text}")

# Clear buttons - always show these buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🧹 Clear Conversation History", key="clear_history"):
        st.session_state.conversation_history = []
        st.rerun()
with col2:
    if st.button("🔄 Reset All Recordings", key="reset_recordings"):
        st.session_state.current_recording = None
        st.session_state.processed_recordings = set()
        st.session_state.recordings_list = []
        st.rerun()

st.markdown("""
<style>
    .stButton button {
        padding: 10px;
        margin: 5px 0;
        width: 100%;
    }
    .stSlider {
        padding: 10px 0;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
</style>
""", unsafe_allow_html=True)