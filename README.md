# 🎧 Audio Emotion & Speech Detector

An interactive AI-powered application that:
- Transcribes audio to text using OpenAI Whisper
- Detects emotional tone from transcribed text using a Hugging Face model
- Visualizes emotion scores in a bar chart
- Provides a web-based UI via Gradio

---

## 🚀 Workshop Setup in Google Colab

This guide walks you through building the full project in **Google Colab** in just 4 steps.

---

## ✅ Learning Outcomes

By completing this workshop, you'll gain hands-on experience in:

1. 🧑‍🎨 **Frontend Development using Gradio**  
   Build a clean, interactive web UI using just Python.

2. 💻 **Terminal + System Environment in Google Colab**  
   Learn how to install, manage, and run Python libraries and files in a cloud-based notebook.

3. 🐍 **Building a Full Python Application**  
   Integrate speech recognition, NLP, and data visualization in one end-to-end solution.

---

## 📦 Libraries We Used — And Why

| Library                  | Purpose                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| `openai-whisper`         | Transcribes speech to text (AI model by OpenAI)                         |
| `transformers`           | Loads pretrained emotion detection models from Hugging Face             |
| `gradio`                 | Builds an interactive frontend interface                               |
| `torchaudio`             | Handles audio preprocessing and compatibility                          |
| `matplotlib`             | Visualizes emotion confidence as bar charts                            |
| `sentencepiece`          | Helps with tokenizer compatibility for transformer models              |
| `ffmpeg-python`          | Ensures audio file handling and format conversion                      |

---

## 🧑‍💻 Step-by-Step Setup (in Google Colab)

### 📌 Step 1: Install Dependencies

Paste and run this in the **first code cell** of your Colab notebook:

```python
!pip install -q openai-whisper transformers gradio torchaudio matplotlib sentencepiece ffmpeg-python
```
### 📌 Step 2: Import Required Libraries

Paste and run this in the **second code cell** of your Colab notebook:

```python
import whisper
import gradio as gr
import torch
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
```
### 📌 Step 3: Load Models & Define Core Logic

Paste and run this in the **third code cell** of your Colab notebook:

```python
# Load Whisper model
whisper_model = whisper.load_model("base")

# Load emotion detection model
emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
emotion_classifier = pipeline("text-classification", model=emotion_model, tokenizer=tokenizer, return_all_scores=True)

# Plot emotions
def plot_emotions(emotion_scores):
    labels = [e['label'] for e in emotion_scores]
    scores = [e['score'] for e in emotion_scores]
    fig, ax = plt.subplots()
    ax.barh(labels, scores, color='skyblue')
    ax.set_xlabel("Confidence")
    ax.set_title("Emotion Scores")
    plt.tight_layout()
    plot_path = "/tmp/emotion_plot.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# Analyze audio
def analyze_audio(file_path):
    result = whisper_model.transcribe(file_path)
    text = result["text"].strip()
    lang_code = result.get("language", "unknown")
    lang_name = whisper.tokenizer.LANGUAGES.get(lang_code, "Unknown").capitalize()

    emotion_scores = emotion_classifier(text)[0]
    sorted_scores = sorted(emotion_scores, key=lambda x: x['score'], reverse=True)
    top_emotion = sorted_scores[0]
    emotion_text = "\n".join([f"{e['label']}: {e['score']:.2f}" for e in sorted_scores])
    plot_path = plot_emotions(sorted_scores)

    final_output = (
        f"🌐 Detected Language: {lang_name}\n\n"
        f"📝 Transcription (What they said):\n{text}\n\n"
        f"🎭 Emotion Scores:\n{emotion_text}\n\n"
        f"🔎 Final Detected Emotion: **{top_emotion['label'].upper()}** (Confidence: {top_emotion['score']:.2f})"
    )
    return final_output, plot_path
```
### 📌 Step 4: Launch the Gradio Interface

Paste and run this in the **fourth code cell** of your Colab notebook:

```python
iface = gr.Interface(
    fn=analyze_audio,
    inputs=gr.Audio(type="filepath", label="🎵 Upload an Audio File"),
    outputs=["text", gr.Image(type="filepath")],
    title="🎧 Audio Emotion & Speech Detector",
    description="Upload any voice/song. Get transcription, all emotion scores, visuals, and final emotion conclusion."
)

iface.launch(share=True)
```
---
### 📬 Contact
For workshop resources or support:

📧 workshop.darion@gmail.com

🌐 www.darion.in
---
### 📊 Expected Output Screenshots

**first code cell**

![Alt text]((https://drive.google.com/file/d/1NDOhkGHLsrOehoZlpDmi6NUiRALRX6c4/view?usp=drive_link))

