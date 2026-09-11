# Transaura

**Local video & audio transcription tool powered by Faster-Whisper.**

Transaura is a lightweight Python CLI tool for converting video and audio into text locally. It supports local media files as well as videos from popular platforms such as YouTube, TikTok, Facebook, Instagram, and X/Twitter.

> **Developed By Sharif Ansari**

---

## ✨ Features

- 🎙️ Local video & audio transcription
- 🤖 Powered by **Faster-Whisper**
- 🌐 YouTube, TikTok, Facebook, Instagram & X/Twitter support
- 📁 Local media file support
- 📝 TXT, SRT & VTT subtitle output
- ⚡ CPU & NVIDIA CUDA support
- 🧠 Multiple Whisper model sizes
- 🌍 Language selection
- 🔄 Transcription & English translation modes
- 📊 Download and processing progress
- 🧹 Automatic cleanup of temporary downloads
- 📂 Custom output directory
- 🛡️ Overwrite protection
- 🧹 Safe filename cleaning
- 🔧 Configurable beam size
- 💻 Fully local — no paid API required

---

## 🧠 Supported Models

Transaura supports the following Faster-Whisper models:

| Model | Speed | Accuracy | Resource Usage |
|---|---|---|---|
| `tiny` | ⚡⚡⚡⚡⚡ | ⭐⭐ | Very Low |
| `base` | ⚡⚡⚡⚡ | ⭐⭐⭐ | Low |
| `small` | ⚡⚡⚡ | ⭐⭐⭐⭐ | Medium |
| `medium` | ⚡⚡ | ⭐⭐⭐⭐⭐ | High |
| `large-v3` | ⚡ | ⭐⭐⭐⭐⭐ | Very High |

For better accuracy, use `medium` or `large-v3` if your hardware can handle it.

---

## 📋 Requirements

- Python 3.9+
- FFmpeg
- pip
- Internet connection for downloading models and online videos
- NVIDIA GPU is optional

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/transaura.git
cd transaura
```

### 2. Install Python dependencies

```bash
pip3 install -r requirements.txt
```

### 3. Install FFmpeg

#### Ubuntu / Debian / Kali Linux

```bash
sudo apt update
sudo apt install ffmpeg
```

Check the installation:

```bash
ffmpeg -version
```

---

## ▶️ Usage

### Transcribe a local video

```bash
python3 transcribe_v5.py -f video.mp4
```

### Transcribe an Urdu video

```bash
python3 transcribe_v5.py -f video.mp4 --language ur
```

### Transcribe an English video

```bash
python3 transcribe_v5.py -f video.mp4 --language en
```

### Use a specific model

```bash
python3 transcribe_v5.py -f video.mp4 --model medium
```

### Use NVIDIA GPU

```bash
python3 transcribe_v5.py -f video.mp4 --device cuda
```

### Force CPU

```bash
python3 transcribe_v5.py -f video.mp4 --device cpu
```

### Generate SRT subtitles

```bash
python3 transcribe_v5.py -f video.mp4 --format srt
```

### Generate TXT and SRT

```bash
python3 transcribe_v5.py -f video.mp4 --format both
```

### Save output to a custom directory

```bash
python3 transcribe_v5.py -f video.mp4 --output-dir transcripts
```

### Specify an output filename

```bash
python3 transcribe_v5.py -f video.mp4 --output transcripts/my_video
```

### Download and transcribe an online video

```bash
python3 transcribe_v5.py -u "https://www.youtube.com/watch?v=VIDEO_ID"
```

The same option can be used with supported platforms such as TikTok, Facebook, Instagram and X/Twitter.

---

## 🌍 Language Options

To see supported language codes:

```bash
python3 transcribe_v5.py --languages
```

Examples:

```bash
--language en
--language ur
--language hi
--language ar
```

If the language is known, specifying it manually can improve transcription reliability.

---

## 🔄 Transcribe vs Translate

### Transcribe

Keeps the original spoken language.

```bash
python3 transcribe_v5.py -f video.mp4 --language en --task transcribe
```

For example:

**English audio → English text**

### Translate

Whisper's built-in translation mode translates supported speech **into English**.

```bash
python3 transcribe_v5.py -f video.mp4 --language ur --task translate
```

For example:

**Urdu audio → English text**

> Note: Whisper's built-in `translate` task does not produce Urdu translations. It is designed to translate speech into English.

---

## 📤 Output Formats

Transaura supports:

### TXT

Plain text transcript.

```bash
--format txt
```

### SRT

Subtitle file with timestamps.

```bash
--format srt
```

### VTT

Web Video Text Tracks subtitle format.

```bash
--format vtt
```

### Both

Creates TXT and SRT files.

```bash
--format both
```

---

## 📁 File Structure

A typical project structure:

```text
transaura/
│
├── transcribe_v5.py
├── requirements.txt
├── README.md
│
├── .downloads/
│   └── temporary media files
│
└── transcripts/
    └── generated transcripts
```

The `.downloads` folder is used only for temporary online media downloads. Temporary files are automatically removed after processing.

Whisper models are stored separately in the Hugging Face cache.

---

## ⚙️ Command Options

```text
-f, --file              Local video/audio file
-u, --url               Online video/audio URL
--model                 Whisper model
--device                auto / cpu / cuda
--language              Source language
--task                  transcribe / translate
--format                txt / srt / vtt / both
--output                Output filename/path
--output-dir            Output directory
--clean-title           Clean output filename
--force                 Overwrite existing files
--beam-size             Whisper beam size
--languages              Show supported languages
-h, --help              Show help
```

---

## ⚡ Performance

For the best balance between speed and accuracy:

```bash
python3 transcribe_v5.py -f video.mp4 --model small
```

For higher accuracy:

```bash
python3 transcribe_v5.py -f video.mp4 --model medium
```

For maximum accuracy, if your hardware has enough resources:

```bash
python3 transcribe_v5.py -f video.mp4 --model large-v3
```

An NVIDIA GPU can significantly improve processing speed.

---

## 🛠️ Troubleshooting

### FFmpeg not found

Install FFmpeg:

```bash
sudo apt install ffmpeg
```

Then verify:

```bash
ffmpeg -version
```

### Online video cannot be downloaded

Some videos may be:

- Private
- Deleted
- Region restricted
- Login required
- Unsupported by the current `yt-dlp` extractor

Make sure the URL is publicly accessible.

### CUDA error

If CUDA is unavailable, use CPU mode:

```bash
python3 transcribe_v5.py -f video.mp4 --device cpu
```

Or let Transaura automatically select the available device:

```bash
python3 transcribe_v5.py -f video.mp4 --device auto
```

---

## 🔒 Privacy

Transaura performs transcription locally using Faster-Whisper.

Your local media files and generated transcripts are not sent to a paid transcription API.

For online videos, the media must first be downloaded from the source platform before transcription.

---

## 👨‍💻 Developer

**Sharif Ansari**

Developed with Python, Faster-Whisper and yt-dlp.

---

## ⭐ Support

If you find Transaura useful, consider giving the repository a ⭐ on GitHub.

**Transaura — Turn speech into text.**
