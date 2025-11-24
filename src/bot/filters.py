"""Custom filters for Telegram bot."""

from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

from aiogram.filters import BaseFilter
from aiogram.types import Message

from src.utils.audio import SUPPORTED_EXTENSIONS, SUPPORTED_MIME_TYPES

# File size limits
DEFAULT_MAX_FILE_SIZE_MB = 500
DEFAULT_MAX_FILE_SIZE_BYTES = DEFAULT_MAX_FILE_SIZE_MB * 1024 * 1024


class AudioFileFilter(BaseFilter):
    """Filter for valid audio files."""

    def __init__(
        self,
        max_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
        allowed_extensions: Optional[Set[str]] = None,
        allowed_mime_types: Optional[Set[str]] = None,
    ):
        """
        Initialize filter.

        Args:
            max_size_bytes: Maximum file size in bytes
            allowed_extensions: Set of allowed extensions (default: SUPPORTED_EXTENSIONS)
            allowed_mime_types: Set of allowed MIME types (default: SUPPORTED_MIME_TYPES)
        """
        self.max_size_bytes = max_size_bytes
        self.allowed_extensions = allowed_extensions or SUPPORTED_EXTENSIONS
        self.allowed_mime_types = allowed_mime_types or SUPPORTED_MIME_TYPES

    async def __call__(self, message: Message) -> Union[bool, Dict[str, Any]]:
        """
        Check if message contains a valid audio file.

        Returns:
            False if invalid, or dict with file info if valid
        """
        file_info = self._extract_file_info(message)
        if not file_info:
            return False

        # Validate file
        validation = self._validate_file(file_info)
        if not validation["is_valid"]:
            return False

        return {"file_info": file_info, "validation": validation}

    def _extract_file_info(self, message: Message) -> Optional[Dict[str, Any]]:
        """Extract file information from message."""
        if message.audio:
            return {
                "file_id": message.audio.file_id,
                "file_unique_id": message.audio.file_unique_id,
                "file_size": message.audio.file_size,
                "file_name": message.audio.file_name or "audio.mp3",
                "mime_type": message.audio.mime_type,
                "duration": message.audio.duration,
                "type": "audio",
            }
        elif message.voice:
            return {
                "file_id": message.voice.file_id,
                "file_unique_id": message.voice.file_unique_id,
                "file_size": message.voice.file_size,
                "file_name": "voice.ogg",
                "mime_type": message.voice.mime_type or "audio/ogg",
                "duration": message.voice.duration,
                "type": "voice",
            }
        elif message.video_note:
            return {
                "file_id": message.video_note.file_id,
                "file_unique_id": message.video_note.file_unique_id,
                "file_size": message.video_note.file_size,
                "file_name": "video_note.mp4",
                "mime_type": "video/mp4",
                "duration": message.video_note.duration,
                "type": "video_note",
            }
        elif message.document:
            # Check if document is audio/video
            mime_type = message.document.mime_type or ""
            file_name = message.document.file_name or "file"

            # Check by extension or mime type
            ext = Path(file_name).suffix.lower()
            is_audio_video = (
                ext in self.allowed_extensions
                or mime_type.lower() in self.allowed_mime_types
                or mime_type.startswith("audio/")
                or mime_type.startswith("video/")
            )

            if not is_audio_video:
                return None

            return {
                "file_id": message.document.file_id,
                "file_unique_id": message.document.file_unique_id,
                "file_size": message.document.file_size,
                "file_name": file_name,
                "mime_type": mime_type,
                "duration": None,
                "type": "document",
            }

        return None

    def _validate_file(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate file against limits."""
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
        }

        # Check file size
        file_size = file_info.get("file_size") or 0
        if file_size > self.max_size_bytes:
            result["is_valid"] = False
            max_mb = self.max_size_bytes / (1024 * 1024)
            actual_mb = file_size / (1024 * 1024)
            result["errors"].append(
                f"File too large: {actual_mb:.1f}MB (max: {max_mb:.0f}MB)"
            )

        # Check extension
        file_name = file_info.get("file_name", "")
        ext = Path(file_name).suffix.lower()
        if ext and ext not in self.allowed_extensions:
            # Not an error, might still work via mime type
            result["warnings"].append(f"Unusual extension: {ext}")

        # Check mime type
        mime_type = file_info.get("mime_type", "")
        if mime_type and mime_type.lower() not in self.allowed_mime_types:
            if not mime_type.startswith(("audio/", "video/")):
                result["warnings"].append(f"Unusual MIME type: {mime_type}")

        return result


class FileSizeFilter(BaseFilter):
    """Simple filter for file size limits."""

    def __init__(self, max_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES):
        """Initialize with max file size."""
        self.max_size_bytes = max_size_bytes

    async def __call__(self, message: Message) -> bool:
        """Check if file is within size limit."""
        file_size = 0

        if message.audio:
            file_size = message.audio.file_size or 0
        elif message.voice:
            file_size = message.voice.file_size or 0
        elif message.document:
            file_size = message.document.file_size or 0
        elif message.video_note:
            file_size = message.video_note.file_size or 0

        return file_size <= self.max_size_bytes


class UserRateLimitFilter(BaseFilter):
    """Filter for per-user rate limiting."""

    def __init__(
        self,
        rate_limiter: Any,  # Will be RateLimiter from rate_limiter module
    ):
        """Initialize with rate limiter."""
        self.rate_limiter = rate_limiter

    async def __call__(self, message: Message) -> Union[bool, Dict[str, Any]]:
        """Check if user is within rate limits."""
        user_id = message.from_user.id if message.from_user else 0

        if not self.rate_limiter:
            return True

        is_allowed, info = await self.rate_limiter.check_user(user_id)

        if not is_allowed:
            return False

        return {"rate_limit_info": info}


class AdminFilter(BaseFilter):
    """Filter for admin-only commands."""

    def __init__(self, admin_ids: Set[int]):
        """Initialize with admin user IDs."""
        self.admin_ids = admin_ids

    async def __call__(self, message: Message) -> bool:
        """Check if user is admin."""
        if not message.from_user:
            return False
        return message.from_user.id in self.admin_ids


def get_file_info_from_message(message: Message) -> Optional[Dict[str, Any]]:
    """
    Extract file information from a message.

    Returns:
        Dict with file_id, file_size, file_name, mime_type, duration, type
        or None if no file found
    """
    if message.audio:
        return {
            "file_id": message.audio.file_id,
            "file_unique_id": message.audio.file_unique_id,
            "file_size": message.audio.file_size or 0,
            "file_name": message.audio.file_name or "audio.mp3",
            "mime_type": message.audio.mime_type or "audio/mpeg",
            "duration": message.audio.duration,
            "type": "audio",
        }
    elif message.voice:
        return {
            "file_id": message.voice.file_id,
            "file_unique_id": message.voice.file_unique_id,
            "file_size": message.voice.file_size or 0,
            "file_name": "voice.ogg",
            "mime_type": message.voice.mime_type or "audio/ogg",
            "duration": message.voice.duration,
            "type": "voice",
        }
    elif message.video_note:
        return {
            "file_id": message.video_note.file_id,
            "file_unique_id": message.video_note.file_unique_id,
            "file_size": message.video_note.file_size or 0,
            "file_name": "video_note.mp4",
            "mime_type": "video/mp4",
            "duration": message.video_note.duration,
            "type": "video_note",
        }
    elif message.document:
        mime_type = message.document.mime_type or ""
        if mime_type.startswith(("audio/", "video/")) or Path(
            message.document.file_name or ""
        ).suffix.lower() in SUPPORTED_EXTENSIONS:
            return {
                "file_id": message.document.file_id,
                "file_unique_id": message.document.file_unique_id,
                "file_size": message.document.file_size or 0,
                "file_name": message.document.file_name or "file",
                "mime_type": mime_type,
                "duration": None,
                "type": "document",
            }

    return None
