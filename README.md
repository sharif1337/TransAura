# Transaura

**Local video & audio transcription tool powered by Faster-Whisper.**

Transaura is a Python CLI tool for converting local video/audio and supported online media into text. It supports YouTube, TikTok, Facebook, Instagram, X/Twitter and direct media URLs.

> **Developed By Sharif Ansari**

---

## ✨ Features

- 🎙️ Local video & audio transcription
- 🤖 Faster-Whisper powered
- 🌐 YouTube, TikTok, Facebook, Instagram & X/Twitter support
- 📝 TXT, SRT, VTT & JSON output
- ⚡ CPU & NVIDIA CUDA support
- 🧠 Multiple Whisper models
- 🌍 Language selection
- 🔄 Transcription & English translation
- 🔤 Optional Roman Urdu output with `uromanizer`
- 📊 Download & transcription progress
- 🧹 Automatic temporary-file cleanup
- 📂 Custom output directory and filename
- 🛡️ Overwrite protection
- 🔧 Configurable beam size and decoding candidates
- 💻 Local transcription — no paid transcription API required

---

## 🧠 Supported Models

| Model | Speed | Accuracy | Resource Usage |
|---|---|---|---|
| `tiny` | ⚡⚡⚡⚡⚡ | ⭐⭐ | Very Low |
| `base` | ⚡⚡⚡⚡ | ⭐⭐⭐ | Low |
| `small` | ⚡⚡⚡ | ⭐⭐⭐⭐ | Medium |
| `medium` | ⚡⚡ | ⭐⭐⭐⭐⭐ | High |
| `large-v3` | ⚡ | ⭐⭐⭐⭐⭐ | Very High |

Larger models generally require more system resources.

---

## 📋 Requirements

- Python 3.9+
- FFmpeg
- `faster-whisper`
- `yt-dlp` for online media
- `uromanizer` for `--roman`
- NVIDIA GPU is optional

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/sharif1337/TransAura.git
cd TransAura
```

### 2. Install dependencies

```bash
pip3 install -r requirements.txt
```

### 3. Install FFmpeg

#### Ubuntu / Debian / Kali Linux

```bash
sudo apt update
sudo apt install ffmpeg
```

## ▶️ Usage

### Local video/audio

```bash
python3 transaura.py -f video.mp4
```

### Urdu transcription

```bash
python3 transaura.py -f video.mp4 --language ur
```

### Specific model

```bash
python3 transaura.py -f video.mp4 --model medium
```

### NVIDIA GPU

```bash
python3 transaura.py -f video.mp4 --device cuda
```

### CPU

```bash
python3 transaura.py -f video.mp4 --device cpu
```

### Online video

```bash
python3 transaura.py -u "https://www.youtube.com/watch?v=VIDEO_ID"
```

The same option supports the online platforms handled by `yt-dlp` in Transaura, as well as direct media URLs.

### Roman Urdu

```bash
python3 transaura.py -f video.mp4 --language ur --roman
```

Example:

```bash
python3 transaura.py -f test2.webm --model large-v3 --force --format txt --language ur --roman
```

---

## 🌍 Language & Translation

Show supported language codes:

```bash
python3 transaura.py --languages
```

Transcribe the original language:

```bash
python3 transaura.py -f video.mp4 --language en --task transcribe
```

Translate supported speech into English:

```bash
python3 transaura.py -f video.mp4 --language ur --task translate
```

> Whisper's `translate` task translates speech **into English**; it does not translate speech into Urdu.

---

## 📤 Output Formats

Supported formats:

- `txt` — plain transcript
- `srt` — subtitles with timestamps
- `vtt` — WebVTT subtitles
- `json` — segment and transcription data
- `both` — TXT + SRT
- `all` — TXT + SRT + VTT + JSON

Examples:

```bash
python3 transaura.py -f video.mp4 --format srt
python3 transaura.py -f video.mp4 --format both
python3 transaura.py -f video.mp4 --format all
```

---

## ⚙️ Command Options

```text
-f, --file              Local video/audio file
-u, --url               Online or direct media URL
--model                 tiny / base / small / medium / large-v3
--device                auto / cpu / cuda
--language              Source language code
--task                  transcribe / translate
--roman                 Convert Urdu transcript to Roman Urdu
--format                txt / srt / vtt / json / both / all
--keep-video            Download full video for platform URLs
--output                Custom output filename
--output-dir            Custom output directory
--clean-title           Clean output filename
--force                 Overwrite existing output files
--beam-size             Beam size (default: 5)
--best-of               Candidate count for fallback (default: 5)
--languages             Show supported language codes
-h, --help              Show help
```

---

### Online video cannot be downloaded

The source may be private, deleted, region restricted, login protected, or unsupported by the current `yt-dlp` extractor.

### CUDA error

Use CPU mode:

```bash
python3 transaura.py -f video.mp4 --device cpu
```

Or use automatic device selection:

```bash
python3 transaura.py -f video.mp4 --device auto
```

### Model download

The first run of a model may take time because the model needs to be downloaded and cached.

---

## 🔒 Privacy

Transaura performs transcription locally using Faster-Whisper. Local media and generated transcripts are not sent to a paid transcription API.

For online sources, the media must first be downloaded before local transcription.

---

## 👨‍💻 Developer

**Sharif Ansari**

Developed with Python, Faster-Whisper and yt-dlp.

---

## ⭐ Support

If you find Transaura useful, consider giving the repository a ⭐ on GitHub.

**Transaura — Turn speech into text.**
