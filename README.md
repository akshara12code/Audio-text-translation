# Voice Translation using Gen AI

A Python-based application that translates audio files from one language to another using Facebook's M2M100 translation model.

## 📋 Description

This project provides an audio translation pipeline that:
- Converts audio files to text (speech-to-text)
- Translates the text to a target language using M2M100
- Supports multiple language pairs

  <img width="1858" height="976" alt="Screenshot 2025-12-10 144154" src="https://github.com/user-attachments/assets/ad73aa97-374a-4803-b5c4-a500aac97221" />

  <img width="1103" height="962" alt="Screenshot 2025-12-10 144301" src="https://github.com/user-attachments/assets/1fcfe3b6-3acd-400a-b076-be05e40490b5" />
  <img width="1577" height="969" alt="image" src="https://github.com/user-attachments/assets/52e7c7e6-60a4-49df-8e85-5f1482a13153" />



## 🚀 Features

- Multi-language support with M2M100 model
- High-quality translation using pre-trained AI models
- Easy-to-use Python interface
- Supports various audio formats

## 📦 Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Internet connection (for first-time model download)

## 🔧 Installation

1. Clone or download this repository:
```bash
git clone <your-repository-url>
cd ML_PROJECT
```

2. Install required dependencies:
```bash
pip install transformers torch torchaudio
pip install SpeechRecognition
pip install pydub
```

3. (Optional) If working with audio files, you may need FFmpeg:
```bash
# Windows (using Chocolatey)
choco install ffmpeg

# macOS (using Homebrew)
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg
```

## 📁 Project Structure

```
ML_PROJECT/
├── main.py                          # Main application file
├── static/                          # Static files (CSS, JS, images)
├── templates/                       # HTML templates
│   └── index.html                  # Web interface
├── Voice-Translation-using-...     # Additional documentation
└── README.md                        # This file
```

## 🎯 Usage

### Basic Usage

1. Activate your Python environment:
```bash
# If using conda
conda activate base

# If using venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Run the application:
```bash
python main.py
```

3. The application will:
   - Load the M2M100 tokenizer
   - Load the translation model (may take time on first run)
   - Process your audio input
   - Output translated text

### First Run

On the first run, the model will be downloaded automatically (approximately 2-3 GB). This is a one-time process.

## 🌐 Su
