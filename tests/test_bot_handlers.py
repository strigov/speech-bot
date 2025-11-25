"""Comprehensive tests for bot handlers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.types import Message, User, Chat, CallbackQuery, Audio, Voice, Document
from aiogram.fsm.context import FSMContext

from src.bot.handlers import (
    cmd_start,
    cmd_help,
    cmd_status,
    cmd_cancel,
    handle_audio,
    callback_cancel,
    callback_refresh,
    callback_download,
    state,
    BotState,
    progress_callback,
)
from src.worker import TaskStatus, TaskProgress, create_task
from src.pipeline.aggregator import AggregationResult
from src.utils.rate_limiter import RateLimiter, RateLimitConfig
from config.settings import Settings


@pytest.fixture
def mock_user():
    """Create a mock Telegram user."""
    user = MagicMock(spec=User)
    user.id = 12345
    user.first_name = "Test"
    user.username = "testuser"
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
    message.answer = AsyncMock()
    message.reply = AsyncMock()
    message.edit_text = AsyncMock()
    message.message_id = 1
    return message


@pytest.fixture
def mock_bot():
    """Create a mock bot."""
    bot = MagicMock(spec=Bot)
    bot.get_file = AsyncMock()
    bot.download_file = AsyncMock()
    bot.send_message = AsyncMock()
    bot.edit_message_text = AsyncMock()
    bot.send_document = AsyncMock()
    return bot


@pytest.fixture
def mock_worker():
    """Create a mock worker."""
    worker = MagicMock()
    worker.queue = MagicMock()
    worker.queue.is_full = False
    worker.queue.can_user_submit = MagicMock(return_value=True)
    worker.queue.submit = AsyncMock(return_value=True)
    worker.queue.get_user_tasks = MagicMock(return_value=[])
    worker.queue.get_task = MagicMock(return_value=None)
    worker.queue.cancel_task = AsyncMock(return_value=True)
    worker.queue.size = 0
    worker.queue.get_queue_stats = MagicMock(return_value={
        "queue_size": 0,
        "max_size": 50,
        "total_tasks": 0,
        "active_users": 0,
        "tasks_by_status": {}
    })
    worker.gpu_monitor = MagicMock()
    worker.get_status = MagicMock(return_value={
        "is_running": True,
        "current_task": None,
        "queue_stats": {"queue_size": 0, "max_size": 50}
    })
    return worker


@pytest.fixture
def mock_settings(tmp_path):
    """Create mock settings."""
    settings = MagicMock()
    settings.telegram = MagicMock()
    settings.telegram.bot_token = "test_token"
    settings.processing = MagicMock()
    settings.processing.max_file_size_mb = 500
    settings.paths = MagicMock()
    settings.paths.temp_dir = tmp_path
    settings.limits = MagicMock()
    settings.limits.rate_limits = MagicMock()
    settings.limits.rate_limits.per_user = MagicMock()
    settings.limits.rate_limits.per_user.max_concurrent = 3
    settings.limits.rate_limits.per_user.max_per_hour = 10
    settings.limits.rate_limits.per_user.cooldown_seconds = 60
    return settings


@pytest.fixture(autouse=True)
def setup_bot_state(mock_worker, mock_bot, mock_settings):
    """Set up bot state before each test."""
    # Reset state
    state.__init__()
    state.set_worker(mock_worker)
    state.set_bot(mock_bot)
    state.set_settings(mock_settings)
    yield
    # Clean up
    state.__init__()


# --- Command Handler Tests ---

@pytest.mark.asyncio
async def test_cmd_start(mock_message):
    """Test /start command sends welcome message."""
    await cmd_start(mock_message)

    mock_message.answer.assert_called_once()
    call_args = mock_message.answer.call_args
    assert "Audio Transcriber Bot" in call_args[0][0]
    assert call_args[1]["parse_mode"] == "Markdown"


@pytest.mark.asyncio
async def test_cmd_help(mock_message):
    """Test /help command sends help text."""
    await cmd_help(mock_message)

    mock_message.answer.assert_called_once()
    call_args = mock_message.answer.call_args
    assert "Help" in call_args[0][0]
    assert "Supported formats" in call_args[0][0]
    assert call_args[1]["parse_mode"] == "Markdown"


@pytest.mark.asyncio
async def test_cmd_status_no_tasks(mock_message):
    """Test /status command with no active tasks."""
    await cmd_status(mock_message)

    mock_message.answer.assert_called_once()
    call_args = mock_message.answer.call_args
    assert "No active tasks" in call_args[0][0]


@pytest.mark.asyncio
async def test_cmd_status_with_tasks(mock_message, mock_worker):
    """Test /status command with active tasks."""
    # Create mock task
    mock_task = MagicMock()
    mock_task.original_filename = "test.mp3"
    mock_task.progress.status = TaskStatus.PROCESSING
    mock_task.progress.progress_percent = 50
    mock_task.progress.current_step = "Transcribing"

    mock_worker.queue.get_user_tasks.return_value = [mock_task]

    await cmd_status(mock_message)

    mock_message.answer.assert_called_once()
    call_args = mock_message.answer.call_args
    assert "test.mp3" in call_args[0][0]
    assert "50%" in call_args[0][0]
    assert "Transcribing" in call_args[0][0]


@pytest.mark.asyncio
async def test_cmd_status_no_queue(mock_message):
    """Test /status when queue is not initialized."""
    state.queue = None

    await cmd_status(mock_message)

    mock_message.answer.assert_called_once()
    assert "starting up" in mock_message.answer.call_args[0][0].lower()


@pytest.mark.asyncio
async def test_cmd_cancel_no_pending_tasks(mock_message):
    """Test /cancel command with no pending tasks."""
    await cmd_cancel(mock_message)

    mock_message.answer.assert_called_once()
    assert "No pending tasks" in mock_message.answer.call_args[0][0]


@pytest.mark.asyncio
async def test_cmd_cancel_success(mock_message, mock_worker):
    """Test /cancel command successfully cancels tasks."""
    # Create pending task
    mock_task = MagicMock()
    mock_task.task_id = "test_123"
    mock_task.progress.status = TaskStatus.QUEUED

    mock_worker.queue.get_user_tasks.return_value = [mock_task]

    await cmd_cancel(mock_message)

    mock_worker.queue.cancel_task.assert_called_once_with("test_123")
    mock_message.answer.assert_called_once()
    assert "Cancelled 1" in mock_message.answer.call_args[0][0]


# --- Audio Handler Tests ---

@pytest.mark.asyncio
async def test_handle_audio_no_file_info(mock_message):
    """Test audio handler with invalid message type."""
    mock_message.audio = None
    mock_message.voice = None
    mock_message.video_note = None
    mock_message.document = None

    with patch("src.bot.handlers.get_file_info_from_message", return_value=None):
        await handle_audio(mock_message)

    mock_message.answer.assert_called_once()
    assert "send an audio" in mock_message.answer.call_args[0][0].lower()


@pytest.mark.asyncio
async def test_handle_audio_queue_full(mock_message, mock_worker):
    """Test audio handler when queue is full."""
    mock_worker.queue.is_full = True

    file_info = {
        "file_id": "test_id",
        "file_size": 1000,
        "file_name": "test.mp3",
        "mime_type": "audio/mpeg",
        "duration": 60,
        "type": "audio"
    }

    with patch("src.bot.handlers.get_file_info_from_message", return_value=file_info):
        await handle_audio(mock_message)

    mock_message.answer.assert_called_once()
    # Should show queue full message


@pytest.mark.asyncio
async def test_handle_audio_rate_limit_exceeded(mock_message, mock_worker):
    """Test audio handler when rate limit is exceeded."""
    file_info = {
        "file_id": "test_id",
        "file_size": 1000,
        "file_name": "test.mp3",
        "mime_type": "audio/mpeg",
        "duration": 60,
        "type": "audio"
    }

    # Mock rate limiter to reject
    state.rate_limiter.check_user = MagicMock(return_value=(False, {"message": "Too many requests"}))

    with patch("src.bot.handlers.get_file_info_from_message", return_value=file_info):
        await handle_audio(mock_message)

    mock_message.answer.assert_called_once()
    assert "Too many requests" in mock_message.answer.call_args[0][0]


@pytest.mark.asyncio
async def test_handle_audio_file_too_large(mock_message, mock_worker, mock_settings):
    """Test audio handler with file exceeding size limit."""
    file_info = {
        "file_id": "test_id",
        "file_size": 600 * 1024 * 1024,  # 600 MB
        "file_name": "huge.mp3",
        "mime_type": "audio/mpeg",
        "duration": 60,
        "type": "audio"
    }

    with patch("src.bot.handlers.get_file_info_from_message", return_value=file_info):
        await handle_audio(mock_message)

    mock_message.answer.assert_called_once()
    assert "too large" in mock_message.answer.call_args[0][0].lower()


@pytest.mark.asyncio
async def test_handle_audio_success(mock_message, mock_worker, mock_bot, tmp_path):
    """Test successful audio file handling."""
    file_info = {
        "file_id": "test_id",
        "file_size": 1000,
        "file_name": "test.mp3",
        "mime_type": "audio/mpeg",
        "duration": 60,
        "type": "audio"
    }

    # Mock file download
    mock_file = MagicMock()
    mock_file.file_path = "test/path.mp3"
    mock_bot.get_file.return_value = mock_file

    status_msg = AsyncMock()
    status_msg.message_id = 999
    mock_message.answer.return_value = status_msg

    with patch("src.bot.handlers.get_file_info_from_message", return_value=file_info):
        with patch("src.utils.validation.FileValidator.validate_file", return_value=(True, "")):
            await handle_audio(mock_message)

    # Verify file was downloaded
    mock_bot.get_file.assert_called_once_with("test_id")
    mock_bot.download_file.assert_called_once()

    # Verify task was submitted
    mock_worker.queue.submit.assert_called_once()

    # Verify rate limiter recorded request
    assert state.rate_limiter.record_request.called


@pytest.mark.asyncio
async def test_handle_audio_invalid_file(mock_message, mock_worker, mock_bot, tmp_path):
    """Test audio handler with invalid file format."""
    file_info = {
        "file_id": "test_id",
        "file_size": 1000,
        "file_name": "fake.mp3",
        "mime_type": "audio/mpeg",
        "duration": 60,
        "type": "audio"
    }

    mock_file = MagicMock()
    mock_file.file_path = "test/path.mp3"
    mock_bot.get_file.return_value = mock_file

    status_msg = AsyncMock()
    mock_message.answer.return_value = status_msg

    with patch("src.bot.handlers.get_file_info_from_message", return_value=file_info):
        with patch("src.utils.validation.FileValidator.validate_file", return_value=(False, "Invalid file format")):
            await handle_audio(mock_message)

    # Should edit status message with error
    status_msg.edit_text.assert_called_once()
    assert "Invalid file format" in status_msg.edit_text.call_args[0][0]


# --- Callback Handler Tests ---

@pytest.mark.asyncio
async def test_callback_cancel_success(mock_worker):
    """Test cancel callback successfully cancels task."""
    callback = MagicMock(spec=CallbackQuery)
    callback.data = "cancel_test_123"
    callback.answer = AsyncMock()
    callback.message = MagicMock()
    callback.message.edit_text = AsyncMock()

    await callback_cancel(callback)

    mock_worker.queue.cancel_task.assert_called_once_with("test_123")
    callback.answer.assert_called_once_with("Task cancelled")
    callback.message.edit_text.assert_called_once()


@pytest.mark.asyncio
async def test_callback_cancel_already_processing(mock_worker):
    """Test cancel callback when task is already processing."""
    mock_worker.queue.cancel_task.return_value = False

    callback = MagicMock(spec=CallbackQuery)
    callback.data = "cancel_test_123"
    callback.answer = AsyncMock()
    callback.message = MagicMock()

    await callback_cancel(callback)

    callback.answer.assert_called_once()
    assert "Cannot cancel" in callback.answer.call_args[0][0]


@pytest.mark.asyncio
async def test_callback_refresh(mock_worker):
    """Test refresh callback updates task status."""
    mock_task = MagicMock()
    mock_task.progress.status = TaskStatus.TRANSCRIBING
    mock_task.progress.progress_percent = 75
    mock_task.progress.current_step = "Processing"
    mock_task.progress.message = "Almost done"

    mock_worker.queue.get_task.return_value = mock_task

    callback = MagicMock(spec=CallbackQuery)
    callback.data = "refresh_test_123"
    callback.answer = AsyncMock()
    callback.message = MagicMock()
    callback.message.edit_text = AsyncMock()

    await callback_refresh(callback)

    callback.message.edit_text.assert_called_once()
    callback.answer.assert_called_once_with("Refreshed")


@pytest.mark.asyncio
async def test_callback_download_success(mock_worker, tmp_path):
    """Test download callback sends file."""
    # Create temporary file
    output_file = tmp_path / "transcript.txt"
    output_file.write_text("Test transcript")

    mock_task = MagicMock()
    mock_task.output_files = {"transcript": output_file}

    mock_worker.queue.get_task.return_value = mock_task

    callback = MagicMock(spec=CallbackQuery)
    callback.data = "download_txt_test_123"
    callback.answer = AsyncMock()
    callback.message = MagicMock()
    callback.message.answer_document = AsyncMock()

    await callback_download(callback)

    callback.message.answer_document.assert_called_once()
    callback.answer.assert_called_once_with("File sent!")


@pytest.mark.asyncio
async def test_callback_download_file_not_found(mock_worker, tmp_path):
    """Test download callback when file doesn't exist."""
    output_file = tmp_path / "nonexistent.txt"

    mock_task = MagicMock()
    mock_task.output_files = {"transcript": output_file}

    mock_worker.queue.get_task.return_value = mock_task

    callback = MagicMock(spec=CallbackQuery)
    callback.data = "download_txt_test_123"
    callback.answer = AsyncMock()
    callback.message = MagicMock()

    await callback_download(callback)

    callback.answer.assert_called_once_with("File not found")


# --- Progress Callback Tests ---

@pytest.mark.asyncio
async def test_progress_callback_completed(mock_bot, mock_worker, tmp_path):
    """Test progress callback on task completion."""
    task_id = "test_123"

    # Set up tracking
    state._user_progress_chats[task_id] = 12345
    state._progress_messages[task_id] = 999

    # Create completed task
    output_file = tmp_path / "transcript.txt"
    output_file.write_text("Test")

    mock_task = MagicMock()
    mock_task.result = MagicMock()
    mock_task.result.total_duration_seconds = 60
    mock_task.result.num_speakers = 2
    mock_task.result.total_words = 100
    mock_task.result.processing_time_seconds = 30
    mock_task.output_files = {"transcript": output_file}

    mock_worker.queue.get_task.return_value = mock_task

    progress = TaskProgress(
        status=TaskStatus.COMPLETED,
        progress_percent=100,
        current_step="Done",
        message="Completed"
    )

    await progress_callback(task_id, progress)

    # Should edit message and send transcript
    mock_bot.edit_message_text.assert_called_once()
    mock_bot.send_document.assert_called_once()


@pytest.mark.asyncio
async def test_progress_callback_failed(mock_bot, mock_worker):
    """Test progress callback on task failure."""
    task_id = "test_123"

    state._user_progress_chats[task_id] = 12345
    state._progress_messages[task_id] = 999

    mock_task = MagicMock()
    mock_task.error = "Processing failed"

    mock_worker.queue.get_task.return_value = mock_task

    progress = TaskProgress(
        status=TaskStatus.FAILED,
        progress_percent=50,
        current_step="Error",
        message="Failed"
    )

    await progress_callback(task_id, progress)

    # Should edit message with error
    mock_bot.edit_message_text.assert_called_once()
    assert "Processing failed" in str(mock_bot.edit_message_text.call_args)


@pytest.mark.asyncio
async def test_progress_callback_processing(mock_bot, mock_worker):
    """Test progress callback during processing."""
    task_id = "test_123"

    state._user_progress_chats[task_id] = 12345
    state._progress_messages[task_id] = 999

    progress = TaskProgress(
        status=TaskStatus.TRANSCRIBING,
        progress_percent=60,
        current_step="Transcribing audio",
        message="In progress"
    )

    await progress_callback(task_id, progress)

    # Should update message
    mock_bot.edit_message_text.assert_called_once()


# --- BotState Tests ---

def test_bot_state_initialization():
    """Test BotState initializes correctly."""
    new_state = BotState()
    assert new_state.worker is None
    assert new_state.queue is None
    assert new_state.bot is None
    assert new_state.settings is None
    assert new_state.rate_limiter is None
    assert len(new_state.admin_ids) == 0


def test_bot_state_set_worker(mock_worker):
    """Test setting worker updates queue reference."""
    new_state = BotState()
    new_state.set_worker(mock_worker)
    assert new_state.worker == mock_worker
    assert new_state.queue == mock_worker.queue


def test_bot_state_set_settings(mock_settings):
    """Test setting settings initializes rate limiter."""
    new_state = BotState()
    new_state.set_settings(mock_settings)
    assert new_state.settings == mock_settings
    assert new_state.rate_limiter is not None
    assert isinstance(new_state.rate_limiter, RateLimiter)


def test_bot_state_add_admin():
    """Test adding admin users."""
    new_state = BotState()
    new_state.add_admin(123)
    new_state.add_admin(456)
    assert 123 in new_state.admin_ids
    assert 456 in new_state.admin_ids
