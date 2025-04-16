# 📰 AI News Speaker-Diarization and Transcription App

This project allows you to upload **audio/video news content**, automatically transcribe it with **speaker diarization**, and download a clean, speaker-tagged transcript. It supports **code-mixed Tamil and English**, and uses the **Groq API** for lightning-fast LLM-based transcription and summarization.

---

## 🚀 Features

- 🎤 **Speaker Diarization** using PyAnnote (who said what)
- ✍️ **Speech-to-Text** using Groq's Whisper model
- 📄 **Downloadable Transcripts** with timestamps and speaker labels
- 🌐 **Streamlit Web Interface** (for easy upload and interaction)
- 🧠 Optional **Groq LLM-based summarization** (coming soon!)

---

## 🛠️ Tech Stack

- `Python 3.10+`
- `Streamlit` for UI
- `Librosa` + `Soundfile` for audio processing
- `PyAnnote.audio` for speaker diarization
- `Groq API` for Whisper-based transcription
- `FFmpeg` for media handling (must be installed)

---

## ⚙️ Installation

### 1. Clone this repository

```bash
git clone https://github.com/your-username/news-ai-transcriber.git
cd news-ai-transcriber

conda create -n final python=3.10
conda activate final

streamlit run app.py


kavinass004@gmail.com
https://www.linkedin.com/in/kavinass123
