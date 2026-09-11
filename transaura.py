import argparse
import os
import re
import shutil
import sys
import time
import urllib.request
import warnings
from pathlib import Path

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download


# =========================================================
# CONFIGURATION
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
# DISPLAY HELPERS
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
    milliseconds = int(round(seconds * 1000))

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    secs = milliseconds // 1000
    milliseconds %= 1000

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


def progress_bar(current, total, width=32):
    if not total or total <= 0:
        return "[?]"

    percent = min(
        100,
        max(0, current / total * 100)
    )

    filled = int(
        width * percent / 100
    )

    bar = (
        "█" * filled
        + "░" * (width - filled)
    )

    return (
        f"[{bar}] "
        f"{percent:6.2f}%"
    )


def clean_filename(name):
    """
    Makes a safe filename.
    """

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

    name = name.strip(
        ". "
    )

    if not name:
        name = "transcript"

    # Windows reserved names
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
        name = f"_{name}"

    return name[:180]


# =========================================================
# HARDWARE
# =========================================================

def detect_device():

    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"

    except Exception:
        pass

    return "cpu", "int8"


def get_device(mode):

    auto_device, auto_compute = (
        detect_device()
    )

    if mode == "auto":
        return (
            auto_device,
            auto_compute
        )

    if mode == "cuda":

        try:
            import ctranslate2

            count = (
                ctranslate2
                .get_cuda_device_count()
            )

            if count <= 0:
                raise RuntimeError(
                    "CUDA GPU not detected."
                )

            return (
                "cuda",
                "float16"
            )

        except Exception as e:

            print()
            print(
                "ERROR: CUDA device requested "
                "but CUDA is unavailable."
            )

            print(
                f"Reason: {e}"
            )

            print()
            print(
                "Use --device cpu "
                "or --device auto."
            )

            sys.exit(1)

    return (
        "cpu",
        "int8"
    )


# =========================================================
# URL DETECTION
# =========================================================

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

    domains = [
        "youtube.com",
        "youtu.be",
        "youtube-nocookie.com",
        "tiktok.com",
        "vm.tiktok.com",
        "facebook.com",
        "fb.watch",
        "instagram.com",
        "twitter.com",
        "x.com",
    ]

    url_lower = url.lower()

    return any(
        domain in url_lower
        for domain in domains
    )


# =========================================================
# DIRECT MEDIA DOWNLOAD
# =========================================================

def download_direct_url(
    url,
    output_path
):

    print()
    print(
        "Downloading direct media URL..."
    )

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
            }
        )

        response = urllib.request.urlopen(
            request,
            timeout=60
        )

        total = response.headers.get(
            "Content-Length"
        )

        total = (
            int(total)
            if total
            else 0
        )

        downloaded = 0
        chunk_size = 1024 * 1024

        with open(
            output_path,
            "wb"
        ) as file:

            while True:

                chunk = response.read(
                    chunk_size
                )

                if not chunk:
                    break

                file.write(chunk)

                downloaded += len(
                    chunk
                )

                if total:

                    print(
                        f"\r"
                        f"{progress_bar(downloaded, total)} "
                        f"| {format_size(downloaded)} / "
                        f"{format_size(total)}",
                        end="",
                        flush=True
                    )

                else:

                    print(
                        f"\rDownloaded: "
                        f"{format_size(downloaded)}",
                        end="",
                        flush=True
                    )

        print()

        if (
            not output_path.exists()
            or output_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "Downloaded file is empty."
            )

        print(
            "Download complete."
        )

        return True

    except Exception as e:

        print()
        print(
            "DIRECT DOWNLOAD ERROR"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        try:
            if output_path.exists():
                output_path.unlink()
        except Exception:
            pass

        return False


# =========================================================
# YT-DLP DOWNLOAD
# =========================================================

def download_with_ytdlp(
    url,
    output_dir
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
            "Install:"
        )

        print(
            "pip install -U yt-dlp"
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
    print(
        "Detected supported platform."
    )

    print(
        "Using yt-dlp..."
    )

    print()

    output_template = str(
        output_dir
        / "%(id)s.%(ext)s"
    )

    def progress_hook(data):

        status = data.get(
            "status"
        )

        if status == "downloading":

            downloaded = data.get(
                "downloaded_bytes",
                0
            )

            total = (
                data.get(
                    "total_bytes"
                )
                or data.get(
                    "total_bytes_estimate"
                )
                or 0
            )

            speed = data.get(
                "speed"
            )

            if total:

                text = (
                    f"\r"
                    f"{progress_bar(downloaded, total)} "
                    f"| {format_size(downloaded)} / "
                    f"{format_size(total)}"
                )

            else:

                text = (
                    f"\rDownloaded: "
                    f"{format_size(downloaded)}"
                )

            if speed:
                text += (
                    f" | {format_size(speed)}/s"
                )

            print(
                text,
                end="",
                flush=True
            )

        elif status == "finished":

            print()
            print(
                "Download finished."
            )

    ydl_opts = {

        "outtmpl": output_template,

        "format": (
            "bv*+ba/"
            "b"
        ),

        "merge_output_format": "mp4",

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

            # Try requested files first
            requested = info.get(
                "requested_downloads"
            )

            if requested:

                for item in requested:

                    filepath = item.get(
                        "filepath"
                    )

                    if filepath:

                        path = Path(
                            filepath
                        )

                        if path.exists():
                            return path

            # Normal prepared filename
            prepared = (
                ydl.prepare_filename(
                    info
                )
            )

            prepared_path = Path(
                prepared
            )

            if prepared_path.exists():
                return prepared_path

            # Search downloaded media
            files = [
                f
                for f in output_dir.iterdir()
                if (
                    f.is_file()
                    and f.suffix.lower()
                    in MEDIA_EXTENSIONS
                )
            ]

            if files:

                return max(
                    files,
                    key=lambda f:
                    f.stat().st_mtime
                )

            raise RuntimeError(
                "yt-dlp completed but media "
                "file could not be found."
            )

    except Exception as e:

        print()
        print(
            "YT-DLP ERROR"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        print()
        print(
            "Possible reasons:"
        )

        print(
            "• Video is private/deleted"
        )

        print(
            "• URL is invalid"
        )

        print(
            "• Platform changed its protection"
        )

        print(
            "• Login/cookies are required"
        )

        print(
            "• yt-dlp needs updating"
        )

        print()
        print(
            "Try:"
        )

        print(
            "pip install -U yt-dlp"
        )

        return None


# =========================================================
# MODEL DOWNLOAD
# =========================================================

def model_exists(model_name):

    repo_id = MODEL_REPOS[
        model_name
    ]

    try:

        snapshot_download(
            repo_id=repo_id,
            local_files_only=True
        )

        return True

    except Exception:
        return False


def download_model(
    model_name
):

    repo_id = MODEL_REPOS[
        model_name
    ]

    print()
    print(
        f"Whisper model '{model_name}' "
        f"is not downloaded."
    )

    print(
        "Downloading model..."
    )

    print()

    warnings.filterwarnings(
        "ignore"
    )

    old_verbosity = os.environ.get(
        "HF_HUB_VERBOSITY"
    )

    os.environ[
        "HF_HUB_VERBOSITY"
    ] = "error"

    try:

        from tqdm.auto import tqdm

        class ModelProgress(tqdm):

            def display(
                self,
                msg=None,
                pos=None
            ):

                if self.total:

                    percent = (
                        self.n
                        / self.total
                        * 100
                    )

                    print(
                        f"\rModel download: "
                        f"{percent:6.2f}% | "
                        f"{format_size(self.n)} / "
                        f"{format_size(self.total)}",
                        end="",
                        flush=True
                    )

        try:

            path = snapshot_download(
                repo_id=repo_id,
                tqdm_class=ModelProgress
            )

        except TypeError:

            print(
                "Progress display is not "
                "supported by this "
                "huggingface_hub version."
            )

            print(
                "Downloading normally..."
            )

            path = snapshot_download(
                repo_id=repo_id
            )

        print()
        print(
            "Model download complete."
        )

        return path

    except Exception as e:

        print()
        print(
            "MODEL DOWNLOAD ERROR"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        print()
        print(
            "Check your internet connection "
            "and try again."
        )

        sys.exit(1)

    finally:

        if old_verbosity is None:

            os.environ.pop(
                "HF_HUB_VERBOSITY",
                None
            )

        else:

            os.environ[
                "HF_HUB_VERBOSITY"
            ] = old_verbosity


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(
    model_name,
    device,
    compute_type
):

    print()
    print(
        f"Checking model: {model_name}"
    )

    if not model_exists(
        model_name
    ):

        model_path = download_model(
            model_name
        )

    else:

        print(
            "Model already downloaded."
        )

        model_path = model_name

    print(
        "Loading Whisper model..."
    )

    try:

        model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type
        )

        print(
            "Model loaded successfully."
        )

        return model

    except Exception as e:

        print()
        print(
            "MODEL LOADING ERROR"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        if device == "cuda":

            print()
            print(
                "CUDA model loading failed."
            )

            print(
                "Try running with:"
            )

            print(
                "--device cpu"
            )

        sys.exit(1)


# =========================================================
# OUTPUT
# =========================================================

def write_txt(
    path,
    segments
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        for segment in segments:

            text = (
                segment.text
                .strip()
            )

            if text:

                timestamp = (
                    format_timestamp(
                        segment.start
                    )
                )

                file.write(
                    f"[{timestamp}] "
                    f"{text}\n"
                )


def write_srt(
    path,
    segments
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        index = 1

        for segment in segments:

            text = (
                segment.text
                .strip()
            )

            if not text:
                continue

            start = format_timestamp(
                segment.start,
                srt=True
            )

            end = format_timestamp(
                segment.end,
                srt=True
            )

            file.write(
                f"{index}\n"
            )

            file.write(
                f"{start} --> {end}\n"
            )

            file.write(
                f"{text}\n\n"
            )

            index += 1


def write_vtt(
    path,
    segments
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
                segment.text
                .strip()
            )

            if not text:
                continue

            start = (
                format_timestamp(
                    segment.start,
                    srt=True
                )
                .replace(",", ".")
            )

            end = (
                format_timestamp(
                    segment.end,
                    srt=True
                )
                .replace(",", ".")
            )

            file.write(
                f"{start} --> {end}\n"
            )

            file.write(
                f"{text}\n\n"
            )


# =========================================================
# OUTPUT PATH
# =========================================================

def get_output_base(
    args,
    input_path
):

    if args.output:

        output = Path(
            args.output
        )

        if args.output_dir:

            output = (
                Path(args.output_dir)
                / output
            )

        return output

    if args.output_dir:

        return (
            Path(args.output_dir)
            / clean_filename(
                input_path.stem
            )
        )

    return Path(
        clean_filename(
            input_path.stem
        )
    )


def check_overwrite(
    output_base,
    output_format,
    force
):

    if force:
        return

    extensions = []

    if output_format in (
        "txt",
        "both"
    ):
        extensions.append(".txt")

    if output_format in (
        "srt",
        "both"
    ):
        extensions.append(".srt")

    if output_format == "vtt":
        extensions.append(".vtt")

    existing = []

    for ext in extensions:

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
# LANGUAGE HELP
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
# CLEANUP
# =========================================================

def cleanup_temp_file(
    path
):

    if not path:
        return

    try:

        path = Path(path)

        if path.exists():

            path.unlink()

            print(
                "Temporary media deleted."
            )

    except Exception as e:

        print(
            f"Warning: Could not delete "
            f"temporary file: {e}"
        )


def cleanup_download_dir():

    download_dir = Path(
        ".downloads"
    )

    if not download_dir.exists():
        return

    try:

        if not any(
            download_dir.iterdir()
        ):

            download_dir.rmdir()

    except Exception:
        pass


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Local Video/Audio Transcription "
            "Tool using Faster-Whisper"
        ),

        formatter_class=(
            argparse.RawTextHelpFormatter
        )
    )

    # -----------------------------------------------------
    # Input
    # -----------------------------------------------------

    input_group = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
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
    # Model
    # -----------------------------------------------------

    parser.add_argument(
        "--model",
        default="small",
        choices=[
            "tiny",
            "base",
            "small",
            "medium",
            "large-v3"
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
            "cuda"
        ],
        help=(
            "Processing device: "
            "auto/cpu/cuda "
            "(default: auto)"
        )
    )

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
            "translate"
        ],
        help=(
            "transcribe original language "
            "or translate to English"
        )
    )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    parser.add_argument(
        "--format",
        default="both",
        choices=[
            "txt",
            "srt",
            "vtt",
            "both"
        ],
        help=(
            "Output format "
            "(default: both)"
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
        help=(
            "Custom output folder"
        )
    )

    parser.add_argument(
        "--clean-title",
        action="store_true",
        help=(
            "Use a cleaned/safe filename"
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
    # Quality
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

    # -----------------------------------------------------
    # Help
    # -----------------------------------------------------

    parser.add_argument(
        "--languages",
        action="store_true",
        help=(
            "Show language codes"
        )
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Language list
    # -----------------------------------------------------

    if args.languages:

        show_languages()

        sys.exit(0)

    # -----------------------------------------------------
    # Validate beam
    # -----------------------------------------------------

    if args.beam_size < 1:

        print(
            "ERROR: beam-size must be "
            "at least 1."
        )

        sys.exit(1)

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device, compute_type = (
        get_device(
            args.device
        )
    )

    # -----------------------------------------------------
    # Input
    # -----------------------------------------------------

    temp_media = None
    input_path = None

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
        # yt-dlp platforms
        # -------------------------------------------------

        if is_ytdlp_url(
            url
        ):

            download_dir = Path(
                ".downloads"
            )

            input_path = (
                download_with_ytdlp(
                    url,
                    download_dir
                )
            )

            if input_path is None:
                sys.exit(1)

            temp_media = input_path

        # -------------------------------------------------
        # Direct media
        # -------------------------------------------------

        elif is_supported_media_url(
            url
        ):

            clean_url = (
                url.lower()
                .split("?")[0]
                .split("#")[0]
            )

            extension = ".mp4"

            for ext in MEDIA_EXTENSIONS:

                if clean_url.endswith(
                    ext
                ):

                    extension = ext
                    break

            download_dir = Path(
                ".downloads"
            )

            download_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            filename = (
                f"direct_download"
                f"{extension}"
            )

            input_path = (
                download_dir
                / filename
            )

            success = (
                download_direct_url(
                    url,
                    input_path
                )
            )

            if not success:
                sys.exit(1)

            temp_media = input_path

        else:

            print()
            print(
                "ERROR: Unsupported URL."
            )

            print()
            print(
                "Supported:"
            )

            print(
                "• YouTube"
            )

            print(
                "• TikTok"
            )

            print(
                "• Facebook"
            )

            print(
                "• Instagram"
            )

            print(
                "• X / Twitter"
            )

            print(
                "• Direct MP4/MP3/etc."
            )

            sys.exit(1)

    # -----------------------------------------------------
    # Output path
    # -----------------------------------------------------

    output_base = get_output_base(
        args,
        input_path
    )

    if args.clean_title:

        output_base = (
            output_base.parent
            / clean_filename(
                output_base.name
            )
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
    # Header
    # -----------------------------------------------------

    print()
    width = 60

    print("=" * width)
    print("TRANSAURA".center(width))
    print("Developed By Sharif Ansari".center(width))
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
        f"Format      : {args.format}"
    )

    print(
        f"Output      : {output_base}"
    )

    print(
        "=" * 70
    )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model = load_model(
        args.model,
        device,
        compute_type
    )

    # -----------------------------------------------------
    # Transcription
    # -----------------------------------------------------

    print()
    print(
        "Transcribing..."
    )

    start_time = time.time()

    segments = []

    try:

        segments_generator, info = (
            model.transcribe(
                str(input_path),
                language=args.language,
                task=args.task,
                beam_size=args.beam_size,
                vad_filter=True,
                condition_on_previous_text=True
            )
        )

        duration = (
            getattr(
                info,
                "duration",
                0
            )
            or 0
        )

        for segment in (
            segments_generator
        ):

            segments.append(
                segment
            )

            current = (
                segment.end
            )

            elapsed = (
                time.time()
                - start_time
            )

            if duration:

                progress = (
                    progress_bar(
                        current,
                        duration
                    )
                )

                print(
                    f"\r{progress} "
                    f"| {format_timestamp(current)} / "
                    f"{format_timestamp(duration)} "
                    f"| {elapsed:.1f}s",
                    end="",
                    flush=True
                )

            else:

                print(
                    f"\rProcessed: "
                    f"{format_timestamp(current)} "
                    f"| {elapsed:.1f}s",
                    end="",
                    flush=True
                )

    except KeyboardInterrupt:

        print()
        print()
        print(
            "Transcription cancelled."
        )

        cleanup_temp_file(
            temp_media
        )

        cleanup_download_dir()

        sys.exit(130)

    except Exception as e:

        print()
        print()
        print(
            "TRANSCRIPTION ERROR"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        cleanup_temp_file(
            temp_media
        )

        cleanup_download_dir()

        sys.exit(1)

    print()
    print()

    # -----------------------------------------------------
    # Language info
    # -----------------------------------------------------

    detected_language = getattr(
        info,
        "language",
        None
    )

    language_probability = getattr(
        info,
        "language_probability",
        None
    )

    if detected_language:

        language_name = (
            LANGUAGES.get(
                detected_language,
                detected_language
            )
        )

        if language_probability is not None:

            print(
                f"Detected language: "
                f"{language_name} "
                f"({language_probability:.1%})"
            )

        else:

            print(
                f"Detected language: "
                f"{language_name}"
            )

    print(
        f"Segments: {len(segments)}"
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    generated_files = []

    try:

        if args.format in (
            "txt",
            "both"
        ):

            txt_path = (
                output_base
                .with_suffix(".txt")
            )

            write_txt(
                txt_path,
                segments
            )

            generated_files.append(
                txt_path
            )

        if args.format in (
            "srt",
            "both"
        ):

            srt_path = (
                output_base
                .with_suffix(".srt")
            )

            write_srt(
                srt_path,
                segments
            )

            generated_files.append(
                srt_path
            )

        if args.format == "vtt":

            vtt_path = (
                output_base
                .with_suffix(".vtt")
            )

            write_vtt(
                vtt_path,
                segments
            )

            generated_files.append(
                vtt_path
            )

    except Exception as e:

        print()
        print(
            "OUTPUT ERROR"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        cleanup_temp_file(
            temp_media
        )

        cleanup_download_dir()

        sys.exit(1)

    # -----------------------------------------------------
    # Cleanup
    # -----------------------------------------------------

    cleanup_temp_file(
        temp_media
    )

    cleanup_download_dir()

    # -----------------------------------------------------
    # Complete
    # -----------------------------------------------------

    total_time = (
        time.time()
        - start_time
    )

    print()
    print(
        "=" * 70
    )

    print(
        "              TRANSCRIPTION COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    for file_path in generated_files:

        print(
            f"✓ {file_path}"
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
        "=" * 70
    )


if __name__ == "__main__":
    main()