"""Audio format utilities and helpers."""

import asyncio
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)

# Supported audio formats (by extension)
SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac",
    ".wma", ".opus", ".webm", ".mp4", ".avi", ".mkv",
    ".mov", ".3gp", ".amr", ".oga"
}

# MIME types for audio/video
SUPPORTED_MIME_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/wave", "audio/x-wav",
    "audio/ogg", "audio/x-ogg", "audio/mp4", "audio/m4a", "audio/x-m4a",
    "audio/flac", "audio/x-flac", "audio/aac", "audio/x-aac",
    "audio/x-ms-wma", "audio/opus", "audio/webm",
    "video/mp4", "video/webm", "video/x-matroska", "video/quicktime",
    "video/3gpp", "video/avi", "video/x-msvideo"
}

# Target format for processing
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1  # Mono


@dataclass
class AudioInfo:
    """Information about an audio file."""

    path: Path
    duration_seconds: float
    sample_rate: int
    channels: int
    codec: str
    format_name: str
    bit_rate: Optional[int]
    file_size_bytes: int
    is_valid: bool
    error: Optional[str] = None


async def get_audio_info(file_path: Path) -> AudioInfo:
    """
    Get audio file information using ffprobe.

    Args:
        file_path: Path to audio file

    Returns:
        AudioInfo with file details
    """
    if not file_path.exists():
        return AudioInfo(
            path=file_path,
            duration_seconds=0,
            sample_rate=0,
            channels=0,
            codec="",
            format_name="",
            bit_rate=None,
            file_size_bytes=0,
            is_valid=False,
            error="File not found",
        )

    file_size = file_path.stat().st_size

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=30.0
        )

        if process.returncode != 0:
            return AudioInfo(
                path=file_path,
                duration_seconds=0,
                sample_rate=0,
                channels=0,
                codec="",
                format_name="",
                bit_rate=None,
                file_size_bytes=file_size,
                is_valid=False,
                error=f"ffprobe failed: {stderr.decode()[:200]}",
            )

        import json
        data = json.loads(stdout.decode())

        # Extract format info
        format_info = data.get("format", {})
        duration = float(format_info.get("duration", 0))
        format_name = format_info.get("format_name", "")
        bit_rate = int(format_info.get("bit_rate", 0)) if format_info.get("bit_rate") else None

        # Find audio stream
        streams = data.get("streams", [])
        audio_stream = next(
            (s for s in streams if s.get("codec_type") == "audio"),
            {}
        )

        sample_rate = int(audio_stream.get("sample_rate", 0))
        channels = int(audio_stream.get("channels", 0))
        codec = audio_stream.get("codec_name", "")

        return AudioInfo(
            path=file_path,
            duration_seconds=duration,
            sample_rate=sample_rate,
            channels=channels,
            codec=codec,
            format_name=format_name,
            bit_rate=bit_rate,
            file_size_bytes=file_size,
            is_valid=True,
        )

    except asyncio.TimeoutError:
        return AudioInfo(
            path=file_path,
            duration_seconds=0,
            sample_rate=0,
            channels=0,
            codec="",
            format_name="",
            bit_rate=None,
            file_size_bytes=file_size,
            is_valid=False,
            error="ffprobe timeout",
        )
    except Exception as e:
        return AudioInfo(
            path=file_path,
            duration_seconds=0,
            sample_rate=0,
            channels=0,
            codec="",
            format_name="",
            bit_rate=None,
            file_size_bytes=file_size,
            is_valid=False,
            error=str(e),
        )


async def convert_to_wav(
    input_path: Path,
    output_path: Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
    timeout_seconds: float = 300.0,
) -> Tuple[bool, Optional[str]]:
    """
    Convert audio file to WAV format suitable for ASR.

    Args:
        input_path: Source audio file
        output_path: Destination WAV file
        sample_rate: Target sample rate (default 16kHz)
        channels: Target channels (default mono)
        timeout_seconds: Maximum conversion time

    Returns:
        Tuple of (success, error_message)
    """
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", str(input_path),
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-f", "wav",
        str(output_path),
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        _, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds
        )

        if process.returncode != 0:
            error_msg = stderr.decode()[:500] if stderr else "Unknown error"
            logger.error("audio_conversion_failed", error=error_msg)
            return False, f"Conversion failed: {error_msg}"

        if not output_path.exists():
            return False, "Output file not created"

        logger.debug(
            "audio_converted",
            input=str(input_path),
            output=str(output_path),
            sample_rate=sample_rate,
        )
        return True, None

    except asyncio.TimeoutError:
        logger.error("audio_conversion_timeout", timeout=timeout_seconds)
        return False, f"Conversion timeout after {timeout_seconds}s"
    except Exception as e:
        logger.error("audio_conversion_error", error=str(e))
        return False, str(e)


async def extract_audio_segment(
    input_path: Path,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> Tuple[bool, Optional[str]]:
    """
    Extract a segment from an audio file.

    Args:
        input_path: Source audio file
        output_path: Destination file
        start_seconds: Start time in seconds
        duration_seconds: Duration to extract
        sample_rate: Target sample rate

    Returns:
        Tuple of (success, error_message)
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_seconds),
        "-i", str(input_path),
        "-t", str(duration_seconds),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-f", "wav",
        str(output_path),
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        _, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=120.0
        )

        if process.returncode != 0:
            error_msg = stderr.decode()[:500] if stderr else "Unknown error"
            return False, f"Segment extraction failed: {error_msg}"

        return True, None

    except asyncio.TimeoutError:
        return False, "Segment extraction timeout"
    except Exception as e:
        return False, str(e)


def is_supported_format(file_path: Path) -> bool:
    """Check if file extension is supported."""
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_supported_mime_type(mime_type: str) -> bool:
    """Check if MIME type is supported."""
    return mime_type.lower() in SUPPORTED_MIME_TYPES


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def estimate_processing_time(duration_seconds: float) -> float:
    """
    Estimate processing time based on audio duration.

    Based on performance targets:
    - < 5 min audio: < 1 min processing
    - 5-30 min audio: < 5 min processing
    - 60 min audio: < 15 min processing

    Returns estimated seconds.
    """
    if duration_seconds <= 300:  # < 5 min
        # ~5x realtime or faster
        return duration_seconds / 5
    elif duration_seconds <= 1800:  # 5-30 min
        # ~6x realtime
        return duration_seconds / 6
    else:
        # ~4x realtime for longer audio (more overhead)
        return duration_seconds / 4


async def check_ffmpeg_available() -> Tuple[bool, str]:
    """
    Check if FFmpeg is available in PATH.

    Returns:
        Tuple of (available, version_string)
    """
    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(
            process.communicate(),
            timeout=10.0
        )

        if process.returncode == 0:
            version_line = stdout.decode().split("\n")[0]
            return True, version_line

        return False, "FFmpeg not working"

    except FileNotFoundError:
        return False, "FFmpeg not found in PATH"
    except Exception as e:
        return False, str(e)


def calculate_chunks(
    duration_seconds: float,
    chunk_duration_seconds: float = 600,  # 10 minutes
    overlap_seconds: float = 30,
) -> list[Tuple[float, float]]:
    """
    Calculate chunk boundaries for long audio processing.

    Args:
        duration_seconds: Total audio duration
        chunk_duration_seconds: Duration of each chunk
        overlap_seconds: Overlap between chunks

    Returns:
        List of (start, end) tuples in seconds
    """
    if duration_seconds <= chunk_duration_seconds:
        return [(0, duration_seconds)]

    chunks = []
    current_start = 0

    while current_start < duration_seconds:
        chunk_end = min(current_start + chunk_duration_seconds, duration_seconds)
        chunks.append((current_start, chunk_end))

        # Move to next chunk with overlap
        current_start = chunk_end - overlap_seconds

        # Avoid tiny final chunks
        if duration_seconds - current_start < overlap_seconds * 2:
            break

    return chunks
