"""Check for required system dependencies."""

import shutil
import subprocess
from typing import Tuple


def check_ffmpeg() -> Tuple[bool, str]:
    """
    Check if FFmpeg and ffprobe are available.

    Returns:
        Tuple of (is_available, error_message)
    """
    ffmpeg_available = shutil.which("ffmpeg") is not None
    ffprobe_available = shutil.which("ffprobe") is not None

    if not ffmpeg_available or not ffprobe_available:
        missing = []
        if not ffmpeg_available:
            missing.append("ffmpeg")
        if not ffprobe_available:
            missing.append("ffprobe")

        error = (
            f"Missing required tools: {', '.join(missing)}\n"
            "Please install FFmpeg from https://ffmpeg.org/download.html\n"
            "Or use: choco install ffmpeg (if Chocolatey is installed)\n"
            "Make sure FFmpeg bin directory is in your PATH environment variable."
        )
        return False, error

    # Verify they work
    try:
        subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            timeout=5,
            check=True,
        )
    except Exception as e:
        return False, f"FFmpeg/ffprobe check failed: {str(e)}"

    return True, ""


def check_dependencies() -> Tuple[bool, str]:
    """
    Check all required system dependencies.

    Returns:
        Tuple of (all_ok, error_message)
    """
    ffmpeg_ok, ffmpeg_error = check_ffmpeg()

    if not ffmpeg_ok:
        return False, ffmpeg_error

    return True, ""
