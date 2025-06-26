import streamlit as st
import torch
import torchaudio
import numpy as np
import json
import joblib
import os
import tempfile
import base64
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from transformers import WhisperProcessor, WhisperModel
from sklearn.neighbors import NearestNeighbors
from streamlit_audiorecorder import audiorecorder
from cree_learning_model import CreeLearningModel

# Set page config
st.set_page_config(
    page_title="Cree Language Learning App",
    page_icon="🌿",
    layout="wide"
)

# Constants for audio app
MODELS_DIR = "../models/audio"
FEATURES_PATH = os.path.join(MODELS_DIR, "features.npy")
KNN_MODEL_PATH = os.path.join(MODELS_DIR, "knn_model.pkl")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")
PATHS_PATH = os.path.join(MODELS_DIR, "paths.json")

# Cache functions for audio app
@st.cache_resource
def load_whisper_model():
    """Load and cache the Whisper model"""
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperModel.from_pretrained("openai/whisper-large-v3")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return processor, model, device

@st.cache_data
def load_dataset():
    """Load the pre-computed features and metadata"""
    try:
        features = np.load(FEATURES_PATH)
        
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)
        
        with open(PATHS_PATH, "r", encoding="utf-8") as f:
            paths = json.load(f)
        
        knn_model = joblib.load(KNN_MODEL_PATH)
        
        return features, labels, paths, knn_model
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None, None, None, None

# Cache function for text app
@st.cache_resource
def load_cree_model():
    """Load and cache the Cree learning model"""
    model = CreeLearningModel()
    model.load_model("../models/cree_learning_model.pkl")
    return model

# Audio processing functions
def load_audio(path, sample_rate=16000):
    """Load and preprocess audio file"""
    wav, sr = torchaudio.load(path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.mean(dim=0).numpy(), sample_rate  # mono

def extract_whisper_embedding(audio_path, processor, model):
    """Extract Whisper embeddings from audio file"""
    audio, sr = load_audio(audio_path)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(model.device)
    
    with torch.no_grad():
        encoder_out = model.encoder(input_features)[0]
    
    return encoder_out.mean(dim=1).cpu().numpy().squeeze()

def query_audio(audio_path, knn_model, labels, paths, processor, model, top_k=3):
    """Find similar audio files"""
    query_emb = extract_whisper_embedding(audio_path, processor, model).reshape(1, -1)
    distances, indices = knn_model.kneighbors(query_emb, n_neighbors=top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            'rank': i + 1,
            'label': labels[idx],
            'distance': distances[0][i],
            'path': paths[idx]
        })
    
    return results

def get_audio_player_html(audio_path):
    """Create HTML audio player"""
    try:
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_b64 = base64.b64encode(audio_bytes).decode()
        
        # Determine audio format
        ext = os.path.splitext(audio_path)[1].lower()
        audio_format = "wav" if ext == ".wav" else "mp3"
        
        html = f'''
        <audio controls style="width: 100%;">
            <source src="data:audio/{audio_format};base64,{audio_b64}" type="audio/{audio_format}">
            Your browser does not support the audio element.
        </audio>
        '''
        return html
    except Exception as e:
        return f"<p>Error loading audio: {str(e)}</p>"

def audio_learning_app():
    """Audio Learning/Matching App"""
    st.header("🎵 Audio Matching")
    st.markdown("Upload an audio file to find similar audio clips from the trained dataset.")
    
    # Load models and dataset
    with st.spinner("Loading models and dataset..."):
        processor, model, device = load_whisper_model()
        features, labels, paths, knn_model = load_dataset()
    
    if features is None:
        st.error("Failed to load dataset. Please check if all model files exist in the correct directory.")
        return
    
    st.success(f"✅ Models loaded successfully! Dataset contains {len(labels)} audio samples.")
    st.info(f"🔧 Using device: {device}")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Audio Settings")
        top_k = st.slider("Number of matches to return", min_value=1, max_value=10, value=3)
        
        # Display dataset info
        with st.expander("Dataset Info"):
            st.write(f"Total samples: {len(labels)}")
            st.write(f"Feature dimension: {features.shape[1]}")       
            st.write("Sample labels:")
            for i, label in enumerate(labels[:5]):
                st.write(f"• {label}")
            if len(labels) > 5:
                st.write(f"... and {len(labels) - 5} more")
    
    # Audio Input Section
    st.subheader("🎙️ Input Audio")
    method = st.radio("Choose input method", ["Upload Audio File", "Record Audio"])

    audio_bytes = None
    audio_filename = None

    if method == "Upload Audio File":
        uploaded_file = st.file_uploader("Upload WAV or MP3", type=["wav", "mp3"])
        if uploaded_file:
            audio_bytes = uploaded_file.getvalue()
            audio_filename = uploaded_file.name
            st.audio(audio_bytes, format=f"audio/{audio_filename.split('.')[-1]}")

    elif method == "Record Audio":
        st.info("Click 'Start Recording' then 'Stop Recording'. Wait a moment for preview.")
        recorded_audio = audiorecorder("Start Recording", "Stop Recording")
        if recorded_audio:
            from io import BytesIO
            buffer = BytesIO()
            recorded_audio.export(buffer, format="wav")
            audio_bytes = buffer.getvalue()
            audio_filename = "recorded.wav"
            st.audio(audio_bytes, format="audio/wav")

    if audio_bytes and audio_filename:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_filename.split('.')[-1]}") as tmp_file:
            tmp_file.write(audio_bytes)
            temp_path = tmp_file.name

        try:
            st.subheader("🔍 Processing...")
            progress_bar = st.progress(0)
            status = st.empty()

            status.text("Extracting embeddings...")
            progress_bar.progress(30)

            results = query_audio(temp_path, knn_model, labels, paths, processor, model, top_k)

            progress_bar.progress(100)
            status.text("✅ Done!")

            st.subheader("🎯 Top Matches")
            for result in results:
                with st.expander(f"#{result['rank']} - {result['label']} (Distance: {result['distance']:.3f})"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write(f"**Label:** {result['label']}")
                        st.write(f"**Distance:** {result['distance']:.3f}")
                        similarity = 1 - result['distance']
                        st.write(f"**Similarity:** {similarity:.3f}")
                        if similarity > 0.8:
                            st.success("🟢 Very Similar")
                        elif similarity > 0.6:
                            st.warning("🟡 Moderately Similar")
                        else:
                            st.error("🔴 Less Similar")
                    with col2:
                        if os.path.exists(result['path']):
                            with open(result['path'], 'rb') as f:
                                st.audio(f.read(), format="audio/wav")
                        else:
                            st.error("Audio file not found")

            st.subheader("📊 Summary")
            distances = [r["distance"] for r in results]
            col1, col2, col3 = st.columns(3)
            col1.metric("Best Distance", f"{min(distances):.3f}")
            col2.metric("Average", f"{np.mean(distances):.3f}")
            col3.metric("Best Similarity", f"{1 - min(distances):.3f}")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

def text_learning_app():
    """Text Learning App"""
    st.header("📝 Cree Language Text Learning")
    
    # Load the model
    model = load_cree_model()
    
    # Sidebar for text app
    with st.sidebar:
        st.header("Text Learning Mode")
        mode = st.selectbox("Choose Mode", [
            "Translate", "Exercise", "Dataset Explorer"])

    # --- Translate Mode ---
    if mode == "Translate":
        st.subheader("🔁 Translation")
        direction = st.radio("Translation Direction", ["Cree → English", "English → Cree"])
        input_word = st.text_input("Enter word:")
        if st.button("Translate"):
            if direction == "Cree → English":
                translations = model.find_translations(input_word)
            else:
                translations = model.find_cree_words(input_word)
            st.write("### Translations:", translations or "No match found.")

    # --- Exercise Mode ---
    elif mode == "Exercise":
        st.subheader("🧪 Take a Cree Translation Test")

        difficulty = st.selectbox("Choose difficulty:", ["mixed", "easy", "hard"])

        # --- Reset on restart or first time ---
        if "text_test_initialized" not in st.session_state or st.session_state.get("text_difficulty_mode") != difficulty:
            st.session_state.text_test_questions = model.create_learning_exercises(difficulty=difficulty)[:10]
            st.session_state.text_current_q = 0
            st.session_state.text_submitted_answers = [None] * 10
            st.session_state.text_correct_flags = [False] * 10
            st.session_state.text_feedbacks = [""] * 10
            st.session_state.text_finished = False
            st.session_state.text_difficulty_mode = difficulty
            st.session_state.text_test_initialized = True

        q_idx = st.session_state.text_current_q
        question = st.session_state.text_test_questions[q_idx]

        # --- If test is not finished ---
        if not st.session_state.text_finished:
            st.markdown(f"### Question {q_idx + 1} of 10")
            st.markdown(f"**Translate this Cree word:** `{question['cree_word']}`")

            selected = st.radio(
                "Choose your answer:",
                options=question["choices"],
                key=f"text_choice_q{q_idx}"
            )

            already_submitted = st.session_state.text_submitted_answers[q_idx] is not None
            if already_submitted:
                st.info(st.session_state.text_feedbacks[q_idx])

            col1, col2 = st.columns(2)

            with col1:
                if st.button("✅ Submit Answer", key="text_submit"):
                    if not already_submitted:
                        st.session_state.text_submitted_answers[q_idx] = selected
                        if selected in question["correct_answers"]:
                            st.session_state.text_correct_flags[q_idx] = True
                            st.session_state.text_feedbacks[q_idx] = "✅ Correct!"
                        else:
                            correct = ", ".join(question["correct_answers"])
                            st.session_state.text_feedbacks[q_idx] = f"❌ Incorrect. Correct answer: **{correct}**"
                        st.rerun()

            with col2:
                if already_submitted and st.button("➡️ Next Question", key="text_next"):
                    if q_idx < 9:
                        st.session_state.text_current_q += 1
                        st.rerun()
                    else:
                        st.warning("This is the last question.")

            st.markdown("---")
            if st.button("🏁 Evaluate Test", key="text_evaluate"):
                st.session_state.text_finished = True
                st.rerun()

        # --- When test is finished ---
        else:
            total_correct = sum(st.session_state.text_correct_flags)
            attempted = sum(ans is not None for ans in st.session_state.text_submitted_answers)

            st.success(f"✅ You answered {total_correct} out of 10 questions correctly.")
            st.write(f"🧮 You attempted {attempted} questions.")
            st.write(f"📊 Your score: **{total_correct} / 10**")
            st.markdown("---")

            for i, q in enumerate(st.session_state.text_test_questions):
                user_answer = st.session_state.text_submitted_answers[i]
                correct = ", ".join(q["correct_answers"])
                st.markdown(f"**Q{i+1}.** `{q['cree_word']}` → Your answer: `{user_answer or 'Not answered'}`")
                if user_answer is None:
                    st.markdown("🟡 Not attempted")
                elif user_answer in q["correct_answers"]:
                    st.markdown("✅ Correct!")
                else:
                    st.markdown(f"❌ Incorrect. Correct answer: **{correct}**")
                st.markdown("—")

            if st.button("🔁 Restart Test", key="text_restart"):
                for key in ["text_test_initialized", "text_current_q", "text_submitted_answers", "text_correct_flags", "text_feedbacks", "text_finished", "text_difficulty_mode"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

    # --- Dataset Explorer ---
    elif mode == "Dataset Explorer":
        st.subheader("📚 Dataset Overview")
        cree_words = list(model.cree_to_english.keys())
        word_limit = st.slider("How many Cree words to show:", 5, 1000, 10)
        data = [(w, ", ".join(model.cree_to_english[w])) for w in cree_words[:word_limit]]
        df = pd.DataFrame(data, columns=["Cree Word", "English Meanings"])
        st.dataframe(df)

        st.write("---")
        st.write(f"Total Cree words: {len(model.cree_to_english)}")
        st.write(f"Cree words with multiple meanings: {sum(1 for v in model.cree_to_english.values() if len(v) > 1)}")
        st.write(f"Average meanings per Cree word: {np.mean([len(v) for v in model.cree_to_english.values()]):.2f}")

def main():
    """Main app with navigation"""
    st.title("🌿 Cree Language Learning App")
    
    # Create navigation bar at the top
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if st.button("📝 Text", use_container_width=True, type="primary" if st.session_state.get('current_tab', 'text') == 'text' else "secondary"):
            st.session_state.current_tab = "text"
    
    with col2:
        if st.button("🎵 Audio", use_container_width=True, type="primary" if st.session_state.get('current_tab', 'text') == 'learning' else "secondary"):
            st.session_state.current_tab = "learning"
    
    # Initialize session state if not exists
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "text"
    
    # Add a horizontal line to separate navigation from content
    st.markdown("---")
    
    # Display the selected app based on the current tab
    if st.session_state.current_tab == "text":
        text_learning_app()
    elif st.session_state.current_tab == "learning":
        audio_learning_app()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Cree Language Learning System</p>
            <p>Built with Streamlit 🌿</p>
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()