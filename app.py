import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import tempfile
import os
from model import Net  # import your Net class

# Parameters
sr = 22050
duration = 5
img_height = 128
img_width = 256

label_names = [
    'German', 'Polish', 'French', 'Hungarian', 'Finnish', 'Romanian', 'Slovak',
    'Spanish', 'Italian', 'Estonian', 'Lithuanian', 'Croatian', 'English',
    'Scottish', 'Irish', 'NorthernIrish', 'Indian', 'Canadian', 'American', 'Dutch'
]
num_classes = len(label_names)

# Load model
device = 'mps'
model = Net(n_classes=num_classes)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Helper
def get_spectrogram(signal):
    spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=2048, hop_length=512, n_mels=img_height)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_resized = librosa.util.fix_length(spec_db, size=img_width, axis=1)
    spec_norm = (spec_resized - np.mean(spec_resized)) / (np.std(spec_resized) + 1e-5)
    return spec_norm

# UI
st.title("English Accent Classifier")

uploaded_file = st.file_uploader("Upload an audio file (.mp3)", type=["mp3"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Audio Playback
    st.subheader("Audio Playback")
    st.audio(uploaded_file, format="audio/mp3")

    # Convert mp3 to waveform
    signal, _ = librosa.load(tmp_path, sr=sr, duration=duration)

    # Get spectrogram
    spec = get_spectrogram(signal)
    input_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    # Display Prediction
    st.subheader("Prediction")
    top_idx = int(np.argmax(probs))
    top_label = label_names[top_idx]
    st.write(f"Predicted Accent: **{top_label}**")

    # Confidence Scores
    st.subheader("Confidence per class")
    st.bar_chart({label_names[i]: probs[i] for i in range(len(label_names))})

    # Cleanup
    os.remove(tmp_path)
