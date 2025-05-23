# English Accent Classification

This project provides an end-to-end pipeline for classifying English accents from audio samples using deep learning. It includes a training notebook, a PyTorch model, and a user-friendly Streamlit web app for real-time accent prediction.

---

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Details](#model-details)
- [Acknowledgements](#acknowledgements)

---

## Overview

- **Notebook:** The `Audio_Classification_Model.ipynb` notebook demonstrates data loading, preprocessing, model training, and evaluation for accent classification using a public dataset.
- **Web App:** The `app.py` file provides a Streamlit interface for uploading `.mp3` files and predicting the accent using the trained model.
- **Model:** The model architecture is defined in `model.py` and the trained weights are saved in `model.pth`.

---

## Demo

Try the live demo here:  
👉 **[https://english-accent-classification.streamlit.app/]**

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/english-accent-classification.git
cd english-accent-classification
```

### 2. Install dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

Then open the provided local URL in your browser.

---

## Project Structure

```
.
├── app.py                      # Streamlit web app for inference
├── model.py                    # PyTorch model definition
├── model.pth                   # Trained model weights
├── requirements.txt            # Python dependencies
├── Audio_Classification_Model.ipynb  # Training and experimentation notebook
└── README.md                   # Project documentation
```

---

## Usage

- **Web App:**  
  1. Open the Streamlit app.
  2. Upload an `.mp3` audio file (5 seconds recommended).
  3. View the predicted accent and confidence scores.

- **Notebook:**  
  Use `Audio_Classification_Model.ipynb` to explore data loading, training, and evaluation. You can adapt it for further experimentation or retraining.

---

## Model Details

- **Architecture:**  
  The model is a simple CNN for spectrogram classification, defined in `model.py`.
- **Input:**  
  Mel-spectrograms of audio clips, resized to 128x256.
- **Classes:**  
  The model predicts among 20+ English accents (see `label_names` in `app.py`).

---

## Acknowledgements

- Dataset: [English Accent DataSet on HuggingFace](https://huggingface.co/datasets/westbrook/English_Accent_DataSet)
- Streamlit for the web interface
- PyTorch and Librosa for model and audio processing
