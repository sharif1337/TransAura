import argparse
import json
import os
import re
import subprocess
import sys
import time
import threading
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

# ---------------------------------------------------------
# Keep Hugging Face output clean before importing HF libs
# ---------------------------------------------------------

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

from uromanizer import Romanizer

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError:
    HfApi = None
    snapshot_download = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


# =========================================================
# TRANSAURA
# Local Multilingual Video / Audio Transcription Tool
# Developed By Sharif Ansari
# =========================================================


MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}


MEDIA_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
}


# Files created during the current run that are safe to remove on Ctrl+C.
_ACTIVE_TEMP_FILES = set()


def register_temp_file(path):
    if path:
        _ACTIVE_TEMP_FILES.add(Path(path))


def cleanup_cancelled_files():
    for path in list(_ACTIVE_TEMP_FILES):
        try:
            path = Path(path)
            if path.exists() and path.is_file():
                path.unlink()
        except Exception:
            pass
    _ACTIVE_TEMP_FILES.clear()


LANGUAGES = {
    "en": "English",
    "ur": "Urdu",
    "hi": "Hindi",
    "ar": "Arabic",
    "fa": "Persian",
    "pa": "Punjabi",
    "bn": "Bengali",
    "tr": "Turkish",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}


# =========================================================
# DISPLAY
# =========================================================

def format_size(size):
    if size is None:
        return "0 B"

    size = float(size)

    if size < 1024:
        return f"{size:.0f} B"

    if size < 1024 ** 2:
        return f"{size / 1024:.1f} KB"

    if size < 1024 ** 3:
        return f"{size / (1024 ** 2):.1f} MB"

    return f"{size / (1024 ** 3):.2f} GB"


def format_timestamp(seconds, srt=False):
    milliseconds = max(
        0,
        int(round(float(seconds) * 1000))
    )

    hours, milliseconds = divmod(
        milliseconds,
        3_600_000
    )

    minutes, milliseconds = divmod(
        milliseconds,
        60_000
    )

    secs, milliseconds = divmod(
        milliseconds,
        1000
    )

    if srt:
        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{secs:02d},"
            f"{milliseconds:03d}"
        )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


def clean_filename(name):
    name = str(name).strip()

    name = re.sub(
        r'[<>:"/\\|?*\x00-\x1f]',
        "",
        name
    )

    name = re.sub(
        r"\s+",
        " ",
        name
    )

    name = name.strip(". ")

    if not name:
        name = "transcript"

    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }

    if name.upper() in reserved:
        name = "_" + name

    return name[:180]


# =========================================================
# PROGRESS
# =========================================================

def create_progress_bar(desc):
    return tqdm(
        total=100,
        desc=desc,
        unit="%",
        ncols=90,
        bar_format=(
            "{l_bar}"
            "{bar}"
            " | {n:3.0f}%"
            " | {postfix}"
        ),
        mininterval=0.05,
        leave=True,
    )


def advance_percent(
    bar,
    state,
    target,
    delay=0.002,
    postfix=None
):
    target = max(
        0,
        min(100, int(target))
    )

    while state["shown"] < target:
        state["shown"] += 1
        bar.n = state["shown"]

        if postfix is not None:
            bar.set_postfix_str(postfix, refresh=False)

        bar.refresh()

        if delay:
            time.sleep(delay)


# =========================================================
# SYSTEM / HARDWARE
# =========================================================

def run_quiet_command(command):
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=(
            subprocess.CREATE_NO_WINDOW
            if os.name == "nt"
            else 0
        ),
    )


def check_ffmpeg():
    try:
        result = run_quiet_command(
            ["ffmpeg", "-version"]
        )

        return result.returncode == 0

    except FileNotFoundError:
        return False

    except Exception:
        return False


def check_ffprobe():
    try:
        result = run_quiet_command(
            ["ffprobe", "-version"]
        )

        return result.returncode == 0

    except FileNotFoundError:
        return False

    except Exception:
        return False


def detect_device():
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"

    except Exception:
        pass

    return "cpu", "int8"


def get_device(mode):
    auto_device, auto_compute = detect_device()

    if mode == "auto":
        return auto_device, auto_compute

    if mode == "cpu":
        return "cpu", "int8"

    if mode == "cuda":
        try:
            import ctranslate2

            count = (
                ctranslate2.get_cuda_device_count()
            )

            if count <= 0:
                raise RuntimeError(
                    "No CUDA GPU detected."
                )

            return "cuda", "float16"

        except Exception as exc:
            print()
            print(
                "ERROR: CUDA was requested "
                "but is unavailable."
            )
            print(
                f"Reason: {exc}"
            )
            print()
            print(
                "Use --device cpu "
                "or --device auto."
            )

            sys.exit(1)

    return "cpu", "int8"


# =========================================================
# URL
# =========================================================

def get_hostname(url):
    try:
        hostname = urlparse(url).hostname

        if not hostname:
            return ""

        return hostname.lower().strip(".")

    except Exception:
        return ""


def is_supported_media_url(url):
    clean_url = (
        url.lower()
        .split("?")[0]
        .split("#")[0]
    )

    return any(
        clean_url.endswith(ext)
        for ext in MEDIA_EXTENSIONS
    )


def is_ytdlp_url(url):
    hostname = get_hostname(url)

    supported_domains = {
        "youtube.com",
        "www.youtube.com",
        "youtu.be",
        "www.youtu.be",
        "youtube-nocookie.com",
        "www.youtube-nocookie.com",

        "tiktok.com",
        "www.tiktok.com",
        "vm.tiktok.com",

        "facebook.com",
        "www.facebook.com",
        "fb.watch",

        "instagram.com",
        "www.instagram.com",

        "twitter.com",
        "www.twitter.com",

        "x.com",
        "www.x.com",
    }

    return hostname in supported_domains


# =========================================================
# AUDIO
# =========================================================

def has_audio_stream(path):
    if not check_ffprobe():
        return True

    try:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]

        result = run_quiet_command(command)

        return (
            result.returncode == 0
            and "audio" in result.stdout.lower()
        )

    except Exception:
        return True


# =========================================================
# DIRECT DOWNLOAD
# =========================================================

def download_direct_url(url, output_path):
    print()
    print("Downloading direct media URL...")
    print()

    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/153.0 Safari/537.36"
                )
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=60
        ) as response:

            total_header = response.headers.get(
                "Content-Length"
            )

            try:
                total = (
                    int(total_header)
                    if total_header
                    else 0
                )
            except ValueError:
                total = 0

            downloaded = 0
            state = {"shown": 0}

            with create_progress_bar(
                "Download"
            ) as bar:

                with open(
                    output_path,
                    "wb"
                ) as file:

                    while True:
                        chunk = response.read(
                            1024 * 1024
                        )

                        if not chunk:
                            break

                        file.write(chunk)
                        downloaded += len(chunk)

                        if total:
                            target = min(
                                100,
                                downloaded * 100 // total
                            )

                            postfix = (
                                f"{format_size(downloaded)}"
                                f" / "
                                f"{format_size(total)}"
                            )

                            advance_percent(
                                bar,
                                state,
                                target,
                                postfix=postfix
                            )

                        else:
                            bar.set_postfix_str(format_size(downloaded), refresh=True)

                if total:
                    advance_percent(
                        bar,
                        state,
                        100,
                        postfix=(
                            f"{format_size(downloaded)}"
                            f" / "
                            f"{format_size(total)}"
                        )
                    )

        if (
            not output_path.exists()
            or output_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "Downloaded file is empty."
            )

        print()
        print("Download complete.")

        return True

    except Exception as exc:
        print()
        print("DIRECT DOWNLOAD ERROR")
        print(
            f"{type(exc).__name__}: {exc}"
        )

        try:
            if output_path.exists():
                output_path.unlink()
        except Exception:
            pass

        return False


# =========================================================
# YT-DLP
# =========================================================

def download_with_ytdlp(
    url,
    output_dir,
    keep_video=False
):
    try:
        import yt_dlp

    except ImportError:
        print()
        print(
            "ERROR: yt-dlp is not installed."
        )
        print()
        print(
            "Install with:"
        )
        print(
            "python -m pip install -U yt-dlp"
        )

        sys.exit(1)

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("Detected supported platform.")
    print("Using yt-dlp...")
    print()

    output_template = str(
        output_dir
        / "%(title).180s [%(id)s].%(ext)s"
    )

    progress_bar = {
        "value": None
    }

    progress_state = {
        "shown": 0
    }

    def get_bar():
        if progress_bar["value"] is None:
            progress_bar["value"] = (
                create_progress_bar(
                    "Download"
                )
            )

        return progress_bar["value"]

    def progress_hook(data):
        status = data.get("status")

        if status == "downloading":

            downloaded = data.get(
                "downloaded_bytes",
                0
            )

            total = (
                data.get("total_bytes")
                or data.get("total_bytes_estimate")
                or 0
            )

            speed = data.get("speed")

            bar = get_bar()

            if total:

                target = min(
                    100,
                    int(
                        downloaded
                        * 100
                        / total
                    )
                )

                postfix = (
                    f"{format_size(downloaded)}"
                    f" / "
                    f"{format_size(total)}"
                )

                if speed:
                    postfix += (
                        f" | "
                        f"{format_size(speed)}/s"
                    )

                advance_percent(
                    bar,
                    progress_state,
                    target,
                    postfix=postfix
                )

            else:

                postfix = (
                    f"Downloaded: "
                    f"{format_size(downloaded)}"
                )

                if speed:
                    postfix += (
                        f" | "
                        f"{format_size(speed)}/s"
                    )

                bar.set_postfix_str(postfix, refresh=True)

        elif status == "finished":

            bar = progress_bar["value"]

            if bar is not None:
                advance_percent(
                    bar,
                    progress_state,
                    100
                )

    ydl_opts = {
        "outtmpl": output_template,

        "format": (
            "bv*+ba/b"
            if keep_video
            else "bestaudio/best"
        ),

        "merge_output_format": (
            "mp4"
            if keep_video
            else "mp3"
        ),

        "noplaylist": True,

        "quiet": True,

        "no_warnings": True,

        "progress_hooks": [
            progress_hook
        ],

        "retries": 5,

        "fragment_retries": 5,

        "concurrent_fragment_downloads": 4,

        "socket_timeout": 30,

        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/153.0 Safari/537.36"
            )
        },
    }

    try:

        with yt_dlp.YoutubeDL(
            ydl_opts
        ) as ydl:

            info = ydl.extract_info(
                url,
                download=True
            )

            bar = progress_bar["value"]

            if bar is not None:
                advance_percent(
                    bar,
                    progress_state,
                    100
                )
                bar.close()

            requested = (
                info.get(
                    "requested_downloads"
                )
                or []
            )

            for item in requested:

                filepath = item.get(
                    "filepath"
                )

                if (
                    filepath
                    and Path(filepath).exists()
                ):
                    return Path(filepath)

            prepared = Path(
                ydl.prepare_filename(info)
            )

            if prepared.exists():
                return prepared

            files = [
                p
                for p in output_dir.iterdir()
                if (
                    p.is_file()
                    and p.suffix.lower()
                    in MEDIA_EXTENSIONS
                )
            ]

            if files:
                return max(
                    files,
                    key=lambda p:
                    p.stat().st_mtime
                )

            raise RuntimeError(
                "yt-dlp completed but the "
                "downloaded media could not "
                "be located."
            )

    except Exception as exc:

        bar = progress_bar["value"]

        if bar is not None:
            try:
                bar.close()
            except Exception:
                pass

        print()
        print("YT-DLP ERROR")
        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()
        print("Possible reasons:")
        print("• Video is private/deleted")
        print("• URL is invalid")
        print("• Login/cookies are required")
        print("• yt-dlp needs updating")

        print()
        print("Try:")
        print(
            "python -m pip install -U yt-dlp"
        )

        return None


# =========================================================
# MODEL
# =========================================================

def get_local_model_path(model_name):
    try:
        return snapshot_download(
            repo_id=MODEL_REPOS[model_name],
            local_files_only=True,
        )

    except Exception:
        return None


def get_model_file_sizes(repo_id):
    try:
        info = HfApi().model_info(
            repo_id,
            files_metadata=True
        )

    except Exception:
        return {}

    result = {}

    for sibling in getattr(
        info,
        "siblings",
        []
    ):

        filename = getattr(
            sibling,
            "rfilename",
            None
        )

        size = getattr(
            sibling,
            "size",
            None
        )

        if (
            not filename
            or size is None
        ):
            continue

        if filename.startswith("."):
            continue

        if filename.lower().endswith(
            (
                ".md",
                ".txt",
                ".gitattributes",
            )
        ):
            continue

        result[filename] = int(size)

    return result


def get_hf_cache_path(repo_id):
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / (
            "models--"
            + repo_id.replace("/", "--")
        )
    )


def calculate_model_downloaded_size(
    repo_cache,
    file_sizes
):
    if not repo_cache.exists():
        return 0

    downloaded = 0

    for filename, remote_size in file_sizes.items():

        target_name = Path(
            filename
        ).name

        try:
            matches = repo_cache.rglob(
                target_name
            )
        except Exception:
            continue

        best = 0

        for match in matches:

            try:

                if match.is_file():

                    size = match.stat().st_size

                    best = max(
                        best,
                        min(
                            size,
                            remote_size
                        )
                    )

            except Exception:
                pass

        downloaded += best

    return downloaded


def download_model(model_name):
    repo_id = MODEL_REPOS[model_name]

    print()
    print(f"Whisper model '{model_name}' is not downloaded.")
    print("Downloading model...")
    print()

    # Use Hugging Face's native progress display for model downloads.
    # Unlike the application's media/transcription progress bars, the
    # model repository may contain multiple files, so HF's native
    # progress is the most accurate representation of the download.
    try:
        result = snapshot_download(repo_id=repo_id)
        print()
        print("Model download complete.")
        return result
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print()
        print("MODEL DOWNLOAD ERROR")
        print(f"{type(exc).__name__}: {exc}")
        return None

def load_model(
    model_name,
    device,
    compute_type
):
    model_path = get_local_model_path(
        model_name
    )

    if WhisperModel is None:
        raise RuntimeError("faster-whisper is not installed. Install with: python -m pip install -U faster-whisper")

    if model_path:

        print()
        print(
            "Model found. Loading..."
        )

    else:

        model_path = download_model(
            model_name
        )

    if not model_path:

        print()
        print(
            "ERROR: Could not locate "
            "Whisper model."
        )

        sys.exit(1)

    try:

        model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
        )

        print(
            "Model loaded successfully."
        )

        return model

    except Exception as exc:

        print()
        print(
            "MODEL LOADING ERROR"
        )
        print(
            f"{type(exc).__name__}: {exc}"
        )

        if device == "cuda":

            print()
            print(
                "CUDA model loading failed."
            )

            print(
                "Try: --device cpu"
            )

        sys.exit(1)


# =========================================================
# ROMANIZATION
# =========================================================

def load_romanizer():
    try:
        return Romanizer(
            language="ur",
            preserve_latin=True
        )

    except Exception as exc:

        print()
        print(
            "ROMANIZER ERROR"
        )
        print(
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)


def romanize_text(
    text,
    romanizer,
    language_code
):
    if not text or romanizer is None:
        return text

    if (
        language_code or ""
    ).lower() != "ur":
        return text

    try:

        result = romanizer.convert(
            text
        )

    except Exception:
        return text

    result = re.sub(
        r"\s+",
        " ",
        result
    ).strip()

    return result


# =========================================================
# TEXT CLEANING
# =========================================================

def normalize_transcript_text(text):
    if not text:
        return ""

    text = text.replace(
        "\u200b",
        ""
    )

    text = text.replace(
        "\ufeff",
        ""
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def is_bad_segment(text):
    if not text:
        return True

    normalized = (
        normalize_transcript_text(text)
    )

    if not normalized:
        return True

    # Ignore obvious one-character noise
    # but preserve legitimate words/numbers.
    if len(normalized) == 1:
        if not normalized.isalnum():
            return True

    return False


def transform_segment_text(
    text,
    args,
    romanizer,
    language_code
):
    text = normalize_transcript_text(
        text
    )

    if not text:
        return ""

    if args.roman:

        text = romanize_text(
            text,
            romanizer,
            language_code
        )

    return normalize_transcript_text(
        text
    )


# =========================================================
# OUTPUT
# =========================================================

def write_txt(
    path,
    segments,
    transform=None
):
    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        for segment in segments:

            text = (
                transform(segment.text)
                if transform
                else normalize_transcript_text(
                    segment.text
                )
            )

            if is_bad_segment(text):
                continue

            file.write(
                f"[{format_timestamp(segment.start)}] "
                f"{text}\n"
            )


def write_srt(
    path,
    segments,
    transform=None
):
    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        index = 1

        for segment in segments:

            text = (
                transform(segment.text)
                if transform
                else normalize_transcript_text(
                    segment.text
                )
            )

            if is_bad_segment(text):
                continue

            start = max(
                0,
                float(segment.start)
            )

            end = max(
                start,
                float(segment.end)
            )

            file.write(
                f"{index}\n"
            )

            file.write(
                f"{format_timestamp(start, True)}"
                f" --> "
                f"{format_timestamp(end, True)}\n"
            )

            file.write(
                f"{text}\n\n"
            )

            index += 1


def write_vtt(
    path,
    segments,
    transform=None
):
    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "WEBVTT\n\n"
        )

        for segment in segments:

            text = (
                transform(segment.text)
                if transform
                else normalize_transcript_text(
                    segment.text
                )
            )

            if is_bad_segment(text):
                continue

            start = (
                format_timestamp(
                    segment.start,
                    True
                )
                .replace(",", ".")
            )

            end = (
                format_timestamp(
                    segment.end,
                    True
                )
                .replace(",", ".")
            )

            file.write(
                f"{start} --> {end}\n"
            )

            file.write(
                f"{text}\n\n"
            )


def write_json(
    path,
    segments,
    info,
    transform=None
):
    data = {
        "language": getattr(
            info,
            "language",
            None
        ),

        "language_probability": getattr(
            info,
            "language_probability",
            None
        ),

        "duration": getattr(
            info,
            "duration",
            None
        ),

        "segments": [],
    }

    for index, segment in enumerate(
        segments,
        start=1
    ):

        text = (
            transform(segment.text)
            if transform
            else normalize_transcript_text(
                segment.text
            )
        )

        if is_bad_segment(text):
            continue

        data["segments"].append({
            "id": index,
            "start": float(
                segment.start
            ),
            "end": float(
                segment.end
            ),
            "text": text,
        })

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


# =========================================================
# OUTPUT PATH
# =========================================================

def get_output_base(
    args,
    input_path
):
    if args.output:
        output = Path(args.output).expanduser()

        if output.suffix.lower() in {
            ".txt", ".srt", ".vtt", ".json"
        }:
            output = output.with_suffix("")

        if args.clean_title:
            output = output.parent / clean_filename(output.name)

        if args.output_dir:
            return Path(args.output_dir).expanduser() / output
        return output

    name = input_path.stem
    if args.clean_title:
        name = clean_filename(name)
    base = Path(name)

    if args.output_dir:
        return Path(args.output_dir).expanduser() / base
    return base


def output_extensions(
    output_format
):
    if output_format == "txt":
        return [".txt"]

    if output_format == "srt":
        return [".srt"]

    if output_format == "vtt":
        return [".vtt"]

    if output_format == "json":
        return [".json"]

    if output_format == "both":
        return [
            ".txt",
            ".srt",
        ]

    return [
        ".txt",
        ".srt",
        ".vtt",
        ".json",
    ]


def check_overwrite(
    output_base,
    output_format,
    force
):
    if force:
        return

    existing = []

    for ext in output_extensions(
        output_format
    ):

        path = output_base.with_suffix(
            ext
        )

        if path.exists():
            existing.append(path)

    if not existing:
        return

    print()
    print(
        "ERROR: Output file already exists:"
    )

    for path in existing:
        print(
            f"  • {path}"
        )

    print()
    print(
        "Use --force to overwrite."
    )

    sys.exit(1)


# =========================================================
# LANGUAGE
# =========================================================

def show_languages():
    print()
    print(
        "Supported language codes:"
    )

    print()

    for code, name in LANGUAGES.items():
        print(
            f"  {code:<4} {name}"
        )

    print()

    print(
        "Leave --language empty for "
        "automatic detection."
    )


# =========================================================
# TRANSCRIPTION
# =========================================================

def transcribe_with_progress(
    model,
    input_path,
    options
):
    generator, info = model.transcribe(
        str(input_path),
        **options
    )

    duration = (
        getattr(
            info,
            "duration",
            0
        )
        or 0
    )

    segments = []

    with create_progress_bar(
        "Transcribe"
    ) as bar:

        state = {
            "shown": 0
        }

        last_end = 0.0

        # faster-whisper returns a lazy generator. The first `next()` can
        # take a noticeable amount of time while audio is decoded and the
        # first inference pass is prepared. Keep the UI alive during that
        # period without inventing progress percentage.
        heartbeat_stop = threading.Event()
        heartbeat_started = time.monotonic()

        def _heartbeat():
            while not heartbeat_stop.wait(0.5):
                elapsed = int(time.monotonic() - heartbeat_started)
                bar.set_postfix_str(
                    f"Starting inference... {elapsed}s",
                    refresh=True,
                )

        heartbeat = threading.Thread(
            target=_heartbeat,
            name="transaura-transcription-progress",
            daemon=True,
        )
        heartbeat.start()

        try:
            for segment in generator:
                # The first yielded segment means actual transcription
                # progress is now available; stop the startup heartbeat.
                heartbeat_stop.set()
                if heartbeat.is_alive():
                    heartbeat.join(timeout=1.0)

                text = normalize_transcript_text(
                    segment.text
                )

                if not is_bad_segment(text):
                    segments.append(segment)

                last_end = max(
                    last_end,
                    float(segment.end)
                )

                if duration > 0:
                    target = min(
                        100,
                        int(
                            last_end
                            * 100
                            / duration
                        )
                    )

                    postfix = (
                        f"{format_timestamp(last_end)}"
                        f" / "
                        f"{format_timestamp(duration)}"
                    )

                    advance_percent(
                        bar,
                        state,
                        target,
                        delay=0.02,
                        postfix=postfix
                    )
                else:
                    bar.set_postfix_str(
                        format_timestamp(last_end),
                        refresh=True
                    )

        finally:
            heartbeat_stop.set()
            if heartbeat.is_alive():
                heartbeat.join(timeout=1.0)

        advance_percent(
            bar,
            state,
            100,
            delay=0.02,
            postfix=(
                f"{format_timestamp(last_end)}"
                + (
                    f" / "
                    f"{format_timestamp(duration)}"
                    if duration
                    else ""
                )
            )
        )

    return segments, info


# =========================================================
# MAIN
# =========================================================

def _main():

    parser = argparse.ArgumentParser(
        description=(
            "TRANSAURA - Local "
            "Video/Audio Transcription "
            "Tool using Faster-Whisper"
        ),
        formatter_class=(
            argparse.RawTextHelpFormatter
        )
    )

    # -----------------------------------------------------
    # INPUT
    # -----------------------------------------------------

    input_group = (
        parser
        .add_mutually_exclusive_group()
    )

    input_group.add_argument(
        "-f",
        "--file",
        help="Local video/audio file"
    )

    input_group.add_argument(
        "-u",
        "--url",
        help=(
            "YouTube, TikTok, Facebook, "
            "Instagram, X/Twitter or direct "
            "media URL"
        )
    )

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    parser.add_argument(
        "--model",
        default="small",
        choices=[
            "tiny",
            "base",
            "small",
            "medium",
            "large-v3",
        ],
        help=(
            "Whisper model "
            "(default: small)"
        )
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        help=(
            "Processing device "
            "(default: auto)"
        )
    )

    # -----------------------------------------------------
    # LANGUAGE / TASK
    # -----------------------------------------------------

    parser.add_argument(
        "--language",
        default=None,
        help=(
            "Spoken language code, "
            "e.g. ur, en, hi"
        )
    )

    parser.add_argument(
        "--task",
        default="transcribe",
        choices=[
            "transcribe",
            "translate",
        ],
        help=(
            "Transcribe original language "
            "or translate to English"
        )
    )

    # -----------------------------------------------------
    # ROMAN
    # -----------------------------------------------------

    parser.add_argument(
        "--roman",
        action="store_true",
        help=(
            "Romanize Urdu transcript "
            "using Romanizer"
        )
    )

    # -----------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------

    parser.add_argument(
        "--format",
        default="both",
        choices=[
            "txt",
            "srt",
            "vtt",
            "json",
            "both",
            "all",
        ],
        help=(
            "Output format "
            "(default: both)"
        )
    )

    parser.add_argument(
        "--keep-video",
        action="store_true",
        help=(
            "Download full video instead "
            "of audio only for platform URLs"
        )
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Custom output filename "
            "without extension"
        )
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Custom output folder"
    )

    parser.add_argument(
        "--clean-title",
        action="store_true",
        help=(
            "Clean output filename "
            "for filesystem compatibility"
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite existing output files"
        )
    )

    # -----------------------------------------------------
    # QUALITY
    # -----------------------------------------------------

    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help=(
            "Beam size "
            "(default: 5)"
        )
    )

    parser.add_argument(
        "--best-of",
        type=int,
        default=5,
        help=(
            "Candidate count for "
            "temperature fallback "
            "(default: 5)"
        )
    )

    # -----------------------------------------------------
    # HELP
    # -----------------------------------------------------

    parser.add_argument(
        "--languages",
        action="store_true",
        help="Show supported language codes"
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # LANGUAGE LIST
    # -----------------------------------------------------

    if args.languages:
        show_languages()
        return

    # -----------------------------------------------------
    # INPUT
    # -----------------------------------------------------

    if not args.file and not args.url:
        parser.error(
            "one of -f/--file or -u/--url "
            "is required"
        )

    # -----------------------------------------------------
    # QUALITY VALIDATION
    # -----------------------------------------------------

    if args.beam_size < 1:
        parser.error(
            "--beam-size must be at least 1"
        )

    if args.best_of < 1:
        parser.error(
            "--best-of must be at least 1"
        )

    # -----------------------------------------------------
    # LANGUAGE VALIDATION
    # -----------------------------------------------------

    if args.language:

        args.language = (
            args.language
            .lower()
            .strip()
        )

        if args.language not in LANGUAGES:

            print()
            print(
                f"ERROR: Unsupported language "
                f"code: {args.language}"
            )

            show_languages()
            sys.exit(1)

    # -----------------------------------------------------
    # ROMANIZER
    # -----------------------------------------------------

    romanizer = None

    if args.roman:
        romanizer = load_romanizer()

    # -----------------------------------------------------
    # DEVICE
    # -----------------------------------------------------

    device, compute_type = get_device(
        args.device
    )

    # -----------------------------------------------------
    # FFMPEG CHECK
    # -----------------------------------------------------

    if not check_ffmpeg():

        print()
        print(
            "WARNING: FFmpeg was not found "
            "in PATH."
        )

        print(
            "Some media formats and "
            "platform downloads may fail."
        )

        print()

    # -----------------------------------------------------
    # INPUT
    # -----------------------------------------------------

    if args.file:

        input_path = Path(
            args.file
        ).expanduser()

        if not input_path.exists():

            print(
                f"ERROR: File not found:\n"
                f"{input_path}"
            )

            sys.exit(1)

        if not input_path.is_file():

            print(
                "ERROR: Input path is not "
                "a file."
            )

            sys.exit(1)

    else:

        url = args.url.strip()

        if not url:
            print(
                "ERROR: Empty URL."
            )
            sys.exit(1)

        # -------------------------------------------------
        # Downloads stay separate from transcripts.
        # -------------------------------------------------

        download_dir = Path(
            ".downloads"
        )

        download_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # -------------------------------------------------
        # PLATFORM URL
        # -------------------------------------------------

        if is_ytdlp_url(url):

            input_path = (
                download_with_ytdlp(
                    url,
                    download_dir,
                    keep_video=args.keep_video
                )
            )

            if input_path is not None:
                register_temp_file(input_path)

            if input_path is None:
                sys.exit(1)

        # -------------------------------------------------
        # DIRECT MEDIA URL
        # -------------------------------------------------

        elif is_supported_media_url(url):

            clean_url = (
                url.lower()
                .split("?")[0]
                .split("#")[0]
            )

            extension = ".mp4"

            for ext in MEDIA_EXTENSIONS:

                if clean_url.endswith(ext):

                    extension = ext
                    break

            parsed = urlparse(url)

            raw_name = Path(
                parsed.path
            ).name

            stem = (
                Path(raw_name).stem
                if raw_name
                else "direct_download"
            )

            stem = (
                clean_filename(stem)
                or "direct_download"
            )

            input_path = (
                download_dir
                / f"{stem}{extension}"
            )

            register_temp_file(input_path)

            if not download_direct_url(
                url,
                input_path
            ):
                sys.exit(1)

        else:

            print()
            print(
                "ERROR: Unsupported URL."
            )

            print()
            print("Supported:")
            print("• YouTube")
            print("• TikTok")
            print("• Facebook")
            print("• Instagram")
            print("• X / Twitter")
            print("• Direct MP4/MP3/etc.")

            sys.exit(1)

    # -----------------------------------------------------
    # AUDIO CHECK
    # -----------------------------------------------------

    if not has_audio_stream(
        input_path
    ):

        print()
        print(
            "ERROR: No audio stream found "
            "in this media file."
        )

        print()
        print(
            "Transcription requires media "
            "containing sound."
        )

        sys.exit(1)

    # -----------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------

    output_base = get_output_base(
        args,
        input_path
    )

    output_base.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    check_overwrite(
        output_base,
        args.format,
        args.force
    )

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    width = 70

    print()
    print("=" * width)
    print(
        "TRANSAURA".center(width)
    )
    print(
        "Developed By Sharif Ansari".center(width)
    )
    print("=" * width)

    print(
        f"Input       : {input_path}"
    )

    print(
        f"Model       : {args.model}"
    )

    print(
        f"Device      : {device}"
    )

    print(
        f"Compute     : {compute_type}"
    )

    print(
        f"Language    : "
        f"{args.language or 'auto-detect'}"
    )

    print(
        f"Task        : {args.task}"
    )

    print(
        f"Roman       : "
        f"{'on' if args.roman else 'off'}"
    )

    print(
        f"Beam size   : {args.beam_size}"
    )

    print(
        f"Best-of     : {args.best_of}"
    )

    print(
        f"Format      : {args.format}"
    )

    print(
        f"Output      : {output_base}"
    )

    print("=" * width)

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = load_model(
        args.model,
        device,
        compute_type
    )

    # -----------------------------------------------------
    # TRANSCRIPTION
    # -----------------------------------------------------

    print()
    print(
        "Transcribing..."
    )
    print()

    start_time = time.time()

    transcription_options = {

        "language": args.language,

        "task": args.task,

        "beam_size": args.beam_size,

        "best_of": args.best_of,

        "temperature": (
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
        ),

        "vad_filter": True,

        "vad_parameters": {
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 400,
        },

        # Helps reduce hallucinations on
        # silence/repeated audio.
        "condition_on_previous_text": False,

        "compression_ratio_threshold": 2.4,

        "log_prob_threshold": -1.0,

        "no_speech_threshold": 0.6,
    }

    # -----------------------------------------------------
    # LANGUAGE PROMPTS
    # -----------------------------------------------------

    if args.language in {
        "ur",
        "hi",
        "pa",
        "ar",
        "fa",
        "bn",
    }:

        transcription_options[
            "initial_prompt"
        ] = (
            "یہ قدرتی گفتگو ہے۔ "
            "مقامی الفاظ، نام، جگہوں، "
            "برانڈز، نمبرز اور English "
            "الفاظ کو درست لکھیں۔ "
            "الفاظ کا مطلب تبدیل نہ کریں۔"
        )

    elif args.language == "en":

        transcription_options[
            "initial_prompt"
        ] = (
            "This is natural spoken English. "
            "Preserve names, brands, technical "
            "terms, numbers and common words accurately."
        )

    try:

        segments, info = (
            transcribe_with_progress(
                model,
                input_path,
                transcription_options
            )
        )

    except KeyboardInterrupt:
        raise

    except Exception as exc:

        print()
        print(
            "TRANSCRIPTION ERROR"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)

    # -----------------------------------------------------
    # LANGUAGE INFO
    # -----------------------------------------------------

    detected_language = getattr(
        info,
        "language",
        None
    )

    probability = getattr(
        info,
        "language_probability",
        None
    )

    print()
    print()

    if detected_language:

        language_name = (
            LANGUAGES.get(
                detected_language,
                detected_language
            )
        )

        if probability is not None:

            print(
                f"Detected language: "
                f"{language_name} "
                f"({probability:.1%})"
            )

        else:

            print(
                f"Detected language: "
                f"{language_name}"
            )

    print(
        f"Segments: {len(segments)}"
    )

    final_language = (
        args.language
        or detected_language
    )

    # -----------------------------------------------------
    # TRANSFORM
    # -----------------------------------------------------

    def transform_text(text):

        return transform_segment_text(
            text,
            args,
            romanizer,
            final_language
        )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    generated_files = []

    try:

        if args.format in {
            "txt",
            "both",
            "all",
        }:

            path = (
                output_base
                .with_suffix(".txt")
            )

            write_txt(
                path,
                segments,
                transform_text
            )

            generated_files.append(path)

        if args.format in {
            "srt",
            "both",
            "all",
        }:

            path = (
                output_base
                .with_suffix(".srt")
            )

            write_srt(
                path,
                segments,
                transform_text
            )

            generated_files.append(path)

        if args.format in {
            "vtt",
            "all",
        }:

            path = (
                output_base
                .with_suffix(".vtt")
            )

            write_vtt(
                path,
                segments,
                transform_text
            )

            generated_files.append(path)

        if args.format in {
            "json",
            "all",
        }:

            path = (
                output_base
                .with_suffix(".json")
            )

            write_json(
                path,
                segments,
                info,
                transform_text
            )

            generated_files.append(path)

    except Exception as exc:

        print()
        print(
            "OUTPUT ERROR"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)

    # -----------------------------------------------------
    # COMPLETE
    # -----------------------------------------------------

    total_time = (
        time.time()
        - start_time
    )

    print()
    print("=" * width)

    print(
        "TRANSCRIPTION COMPLETE".center(
            width
        )
    )

    print("=" * width)

    print()

    for path in generated_files:

        print(
            f"✓ {path}"
        )

    print()

    print(
        f"Time taken: "
        f"{total_time:.1f} seconds"
    )

    print(
        f"Device used: "
        f"{device}"
    )

    print(
        f"Model used: "
        f"{args.model}"
    )

    print("=" * width)


def main():
    """Run the CLI with clean cancellation and dependency errors."""
    try:
        return _main()
    except KeyboardInterrupt:
        print()
        print()
        print("Cancelled by user (Ctrl+C).")
        print("Cleaning up temporary files...")
        cleanup_cancelled_files()
        print("Cancelled cleanly.")
        return 130
    except RuntimeError as exc:
        print()
        print(f"ERROR: {exc}")
        cleanup_cancelled_files()
        return 2


if __name__ == "__main__":
    sys.exit(main() or 0)
