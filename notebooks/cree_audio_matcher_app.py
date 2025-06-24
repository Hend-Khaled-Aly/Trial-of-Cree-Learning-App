import streamlit as st
import torch
import torchaudio
import numpy as np
import json
import joblib
import os
import tempfile
import base64
from transformers import WhisperProcessor, WhisperModel
from sklearn.neighbors import NearestNeighbors
from audiorecorder import audiorecorder

# Set page config
st.set_page_config(
    page_title="Audio Matching with Whisper + KNN",
    page_icon="🎵",
    layout="wide"
)

# Constants
MODELS_DIR = "../models/audio"
FEATURES_PATH = os.path.join(MODELS_DIR, "features.npy")
KNN_MODEL_PATH = os.path.join(MODELS_DIR, "knn_model.pkl")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")
PATHS_PATH = os.path.join(MODELS_DIR, "paths.json")

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

def main():
    st.title("🎵 Audio Matching with Whisper + KNN")
    st.markdown("Upload an audio file to find similar audio clips from the trained dataset.")
    
    # Load models and dataset
    with st.spinner("Loading models and dataset..."):
        processor, model, device = load_whisper_model()
        features, labels, paths, knn_model = load_dataset()
    
    if features is None:
        st.error("Failed to load dataset. Please check if all model files exist in the correct directory.")
        st.stop()
    
    st.success(f"✅ Models loaded successfully! Dataset contains {len(labels)} audio samples.")
    st.info(f"🔧 Using device: {device}")
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Number of matches to return", min_value=1, max_value=10, value=3)
    
    # Display dataset info
    with st.sidebar.expander("Dataset Info"):
        st.write(f"Total samples: {len(labels)}")
        st.write(f"Feature dimension: {features.shape[1]}")       
        st.write("Sample labels:")
        for i, label in enumerate(labels[:5]):
            st.write(f"• {label}")
        if len(labels) > 5:
            st.write(f"... and {len(labels) - 5} more")
    
    # Main interface
        # Audio Input Section
    st.header("🎙️ Input Audio")
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

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Audio Matching System using Whisper + KNN</p>
            <p>Built with Streamlit 🎧</p>
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()