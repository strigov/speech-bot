"""Comprehensive tests for bot filters."""

import pytest
from unittest.mock import MagicMock
from pathlib import Path

from aiogram.types import Message, User, Chat, Audio, Voice, Document, VideoNote

from src.bot.filters import (
    AudioFileFilter,
    FileSizeFilter,
    UserRateLimitFilter,
    AdminFilter,
    get_file_info_from_message,
)


@pytest.fixture
def mock_user():
    """Create a mock Telegram user."""
    user = MagicMock(spec=User)
    user.id = 12345
    user.first_name = "Test"
    return user


@pytest.fixture
def mock_chat():
    """Create a mock Telegram chat."""
    chat = MagicMock(spec=Chat)
    chat.id = 12345
    chat.type = "private"
    return chat


@pytest.fixture
def mock_message(mock_user, mock_chat):
    """Create a mock Telegram message."""
    message = MagicMock(spec=Message)
    message.from_user = mock_user
    message.chat = mock_chat
    message.audio = None
    message.voice = None
    message.document = None
    message.video_note = None
    return message


# --- AudioFileFilter Tests ---

@pytest.mark.asyncio
async def test_audio_file_filter_valid_audio(mock_message):
    """Test AudioFileFilter accepts valid audio files."""
    audio = MagicMock(spec=Audio)
    audio.file_id = "test_id"
    audio.file_unique_id = "unique_id"
    audio.file_size = 1000
    audio.file_name = "test.mp3"
    audio.mime_type = "audio/mpeg"
    audio.duration = 60

    mock_message.audio = audio

    filter_instance = AudioFileFilter()
    result = await filter_instance(mock_message)

    assert result is not False
    assert isinstance(result, dict)
    assert "file_info" in result
    assert result["file_info"]["file_name"] == "test.mp3"


@pytest.mark.asyncio
async def test_audio_file_filter_file_too_large(mock_message):
    """Test AudioFileFilter rejects files that are too large."""
    audio = MagicMock(spec=Audio)
    audio.file_id = "test_id"
    audio.file_unique_id = "unique_id"
    audio.file_size = 600 * 1024 * 1024  # 600 MB
    audio.file_name = "huge.mp3"
    audio.mime_type = "audio/mpeg"
    audio.duration = 3600

    mock_message.audio = audio

    filter_instance = AudioFileFilter(max_size_bytes=500 * 1024 * 1024)
    result = await filter_instance(mock_message)

    assert result is False


@pytest.mark.asyncio
async def test_audio_file_filter_voice_message(mock_message):
    """Test AudioFileFilter accepts voice messages."""
    voice = MagicMock(spec=Voice)
    voice.file_id = "voice_id"
    voice.file_unique_id = "unique_id"
    voice.file_size = 5000
    voice.mime_type = "audio/ogg"
    voice.duration = 10

    mock_message.voice = voice

    filter_instance = AudioFileFilter()
    result = await filter_instance(mock_message)

    assert result is not False
    assert result["file_info"]["file_name"] == "voice.ogg"


@pytest.mark.asyncio
async def test_audio_file_filter_video_note(mock_message):
    """Test AudioFileFilter accepts video notes."""
    video_note = MagicMock(spec=VideoNote)
    video_note.file_id = "video_id"
    video_note.file_unique_id = "unique_id"
    video_note.file_size = 8000
    video_note.duration = 15

    mock_message.video_note = video_note

    filter_instance = AudioFileFilter()
    result = await filter_instance(mock_message)

    assert result is not False
    assert result["file_info"]["file_name"] == "video_note.mp4"


@pytest.mark.asyncio
async def test_audio_file_filter_document_audio(mock_message):
    """Test AudioFileFilter accepts audio documents."""
    document = MagicMock(spec=Document)
    document.file_id = "doc_id"
    document.file_unique_id = "unique_id"
    document.file_size = 10000
    document.file_name = "recording.wav"
    document.mime_type = "audio/wav"

    mock_message.document = document

    filter_instance = AudioFileFilter()
    result = await filter_instance(mock_message)

    assert result is not False
    assert result["file_info"]["file_name"] == "recording.wav"


@pytest.mark.asyncio
async def test_audio_file_filter_document_not_audio(mock_message):
    """Test AudioFileFilter rejects non-audio documents."""
    document = MagicMock(spec=Document)
    document.file_id = "doc_id"
    document.file_unique_id = "unique_id"
    document.file_size = 10000
    document.file_name = "document.pdf"
    document.mime_type = "application/pdf"

    mock_message.document = document

    filter_instance = AudioFileFilter()
    result = await filter_instance(mock_message)

    assert result is False


@pytest.mark.asyncio
async def test_audio_file_filter_no_file(mock_message):
    """Test AudioFileFilter rejects messages without files."""
    filter_instance = AudioFileFilter()
    result = await filter_instance(mock_message)

    assert result is False


@pytest.mark.asyncio
async def test_audio_file_filter_custom_extensions(mock_message):
    """Test AudioFileFilter with custom allowed extensions."""
    document = MagicMock(spec=Document)
    document.file_id = "doc_id"
    document.file_unique_id = "unique_id"
    document.file_size = 10000
    document.file_name = "test.opus"
    document.mime_type = "audio/opus"

    mock_message.document = document

    # Only allow mp3
    filter_instance = AudioFileFilter(allowed_extensions={".mp3"})
    result = await filter_instance(mock_message)

    # Should still pass because mime type is audio/
    assert result is not False


# --- FileSizeFilter Tests ---

@pytest.mark.asyncio
async def test_file_size_filter_within_limit(mock_message):
    """Test FileSizeFilter accepts files within limit."""
    audio = MagicMock(spec=Audio)
    audio.file_size = 100 * 1024 * 1024  # 100 MB

    mock_message.audio = audio

    filter_instance = FileSizeFilter(max_size_bytes=200 * 1024 * 1024)
    result = await filter_instance(mock_message)

    assert result is True


@pytest.mark.asyncio
async def test_file_size_filter_exceeds_limit(mock_message):
    """Test FileSizeFilter rejects files exceeding limit."""
    audio = MagicMock(spec=Audio)
    audio.file_size = 300 * 1024 * 1024  # 300 MB

    mock_message.audio = audio

    filter_instance = FileSizeFilter(max_size_bytes=200 * 1024 * 1024)
    result = await filter_instance(mock_message)

    assert result is False


@pytest.mark.asyncio
async def test_file_size_filter_no_file(mock_message):
    """Test FileSizeFilter handles messages without files."""
    filter_instance = FileSizeFilter()
    result = await filter_instance(mock_message)

    # No file = 0 bytes = within limit
    assert result is True


@pytest.mark.asyncio
async def test_file_size_filter_none_file_size(mock_message):
    """Test FileSizeFilter handles None file_size."""
    audio = MagicMock(spec=Audio)
    audio.file_size = None

    mock_message.audio = audio

    filter_instance = FileSizeFilter()
    result = await filter_instance(mock_message)

    # None treated as 0 = within limit
    assert result is True


# --- UserRateLimitFilter Tests ---

@pytest.mark.asyncio
async def test_user_rate_limit_filter_allowed(mock_message):
    """Test UserRateLimitFilter allows requests within limits."""
    mock_limiter = MagicMock()
    mock_limiter.check_user.return_value = (True, {"remaining": 5})

    filter_instance = UserRateLimitFilter(rate_limiter=mock_limiter)
    result = await filter_instance(mock_message)

    assert result is not False
    assert isinstance(result, dict)
    assert "rate_limit_info" in result


@pytest.mark.asyncio
async def test_user_rate_limit_filter_blocked(mock_message):
    """Test UserRateLimitFilter blocks requests exceeding limits."""
    mock_limiter = MagicMock()
    mock_limiter.check_user.return_value = (False, {"reason": "hourly_limit"})

    filter_instance = UserRateLimitFilter(rate_limiter=mock_limiter)
    result = await filter_instance(mock_message)

    assert result is False


@pytest.mark.asyncio
async def test_user_rate_limit_filter_no_limiter(mock_message):
    """Test UserRateLimitFilter allows all when no limiter set."""
    filter_instance = UserRateLimitFilter(rate_limiter=None)
    result = await filter_instance(mock_message)

    assert result is True


@pytest.mark.asyncio
async def test_user_rate_limit_filter_no_user(mock_message):
    """Test UserRateLimitFilter handles messages without user."""
    mock_message.from_user = None
    mock_limiter = MagicMock()
    mock_limiter.check_user.return_value = (True, {})

    filter_instance = UserRateLimitFilter(rate_limiter=mock_limiter)
    result = await filter_instance(mock_message)

    # Should check user_id 0
    mock_limiter.check_user.assert_called_once_with(0)


# --- AdminFilter Tests ---

@pytest.mark.asyncio
async def test_admin_filter_is_admin(mock_message):
    """Test AdminFilter allows admin users."""
    filter_instance = AdminFilter(admin_ids={12345, 67890})
    result = await filter_instance(mock_message)

    assert result is True


@pytest.mark.asyncio
async def test_admin_filter_not_admin(mock_message):
    """Test AdminFilter blocks non-admin users."""
    filter_instance = AdminFilter(admin_ids={99999})
    result = await filter_instance(mock_message)

    assert result is False


@pytest.mark.asyncio
async def test_admin_filter_no_user(mock_message):
    """Test AdminFilter blocks messages without user."""
    mock_message.from_user = None
    filter_instance = AdminFilter(admin_ids={12345})
    result = await filter_instance(mock_message)

    assert result is False


@pytest.mark.asyncio
async def test_admin_filter_empty_admin_list(mock_message):
    """Test AdminFilter blocks all when no admins configured."""
    filter_instance = AdminFilter(admin_ids=set())
    result = await filter_instance(mock_message)

    assert result is False


# --- get_file_info_from_message Tests ---

def test_get_file_info_audio(mock_message):
    """Test extracting file info from audio message."""
    audio = MagicMock(spec=Audio)
    audio.file_id = "audio_123"
    audio.file_unique_id = "unique_123"
    audio.file_size = 5000
    audio.file_name = "song.mp3"
    audio.mime_type = "audio/mpeg"
    audio.duration = 180

    mock_message.audio = audio

    info = get_file_info_from_message(mock_message)

    assert info is not None
    assert info["file_id"] == "audio_123"
    assert info["file_size"] == 5000
    assert info["file_name"] == "song.mp3"
    assert info["mime_type"] == "audio/mpeg"
    assert info["duration"] == 180
    assert info["type"] == "audio"


def test_get_file_info_voice(mock_message):
    """Test extracting file info from voice message."""
    voice = MagicMock(spec=Voice)
    voice.file_id = "voice_123"
    voice.file_unique_id = "unique_123"
    voice.file_size = 2000
    voice.mime_type = "audio/ogg"
    voice.duration = 5

    mock_message.voice = voice

    info = get_file_info_from_message(mock_message)

    assert info is not None
    assert info["file_id"] == "voice_123"
    assert info["file_name"] == "voice.ogg"
    assert info["type"] == "voice"


def test_get_file_info_video_note(mock_message):
    """Test extracting file info from video note."""
    video_note = MagicMock(spec=VideoNote)
    video_note.file_id = "video_123"
    video_note.file_unique_id = "unique_123"
    video_note.file_size = 10000
    video_note.duration = 30

    mock_message.video_note = video_note

    info = get_file_info_from_message(mock_message)

    assert info is not None
    assert info["file_id"] == "video_123"
    assert info["file_name"] == "video_note.mp4"
    assert info["mime_type"] == "video/mp4"
    assert info["type"] == "video_note"


def test_get_file_info_document_audio(mock_message):
    """Test extracting file info from audio document."""
    document = MagicMock(spec=Document)
    document.file_id = "doc_123"
    document.file_unique_id = "unique_123"
    document.file_size = 15000
    document.file_name = "podcast.mp3"
    document.mime_type = "audio/mpeg"

    mock_message.document = document

    info = get_file_info_from_message(mock_message)

    assert info is not None
    assert info["file_id"] == "doc_123"
    assert info["file_name"] == "podcast.mp3"
    assert info["type"] == "document"


def test_get_file_info_document_not_audio(mock_message):
    """Test get_file_info returns None for non-audio documents."""
    document = MagicMock(spec=Document)
    document.file_id = "doc_123"
    document.file_unique_id = "unique_123"
    document.file_size = 15000
    document.file_name = "report.pdf"
    document.mime_type = "application/pdf"

    mock_message.document = document

    info = get_file_info_from_message(mock_message)

    assert info is None


def test_get_file_info_no_file(mock_message):
    """Test get_file_info returns None when no file present."""
    info = get_file_info_from_message(mock_message)
    assert info is None


def test_get_file_info_audio_no_filename(mock_message):
    """Test get_file_info handles audio without filename."""
    audio = MagicMock(spec=Audio)
    audio.file_id = "audio_123"
    audio.file_unique_id = "unique_123"
    audio.file_size = 5000
    audio.file_name = None
    audio.mime_type = "audio/mpeg"
    audio.duration = 180

    mock_message.audio = audio

    info = get_file_info_from_message(mock_message)

    assert info is not None
    assert info["file_name"] == "audio.mp3"  # Default filename


def test_get_file_info_document_by_extension(mock_message):
    """Test get_file_info recognizes audio by extension when mime type is wrong."""
    document = MagicMock(spec=Document)
    document.file_id = "doc_123"
    document.file_unique_id = "unique_123"
    document.file_size = 15000
    document.file_name = "audio.mp3"
    document.mime_type = "application/octet-stream"  # Generic type

    mock_message.document = document

    info = get_file_info_from_message(mock_message)

    # Should recognize as audio by extension
    assert info is not None
    assert info["file_name"] == "audio.mp3"
