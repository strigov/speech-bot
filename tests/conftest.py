"""Shared pytest fixtures and configuration for all tests."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
import tempfile
import shutil

from config.settings import Settings, TelegramSettings
from src.worker import Worker, TaskQueue
from src.utils.rate_limiter import RateLimiter, RateLimitConfig


@pytest.fixture
def tmp_test_dir():
    """Create a temporary directory for test files."""
    temp_dir = Path(tempfile.mkdtemp(prefix="speech_bot_test_"))
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_settings(tmp_test_dir):
    """Create mock settings for testing."""
    settings = MagicMock(spec=Settings)
    settings.telegram.bot_token = "test_token_123"
    settings.processing.max_file_size_mb = 500
    settings.processing.max_audio_duration_minutes = 180
    settings.processing.max_queue_size = 50
    settings.processing.max_user_concurrent = 3
    settings.paths.temp_dir = tmp_test_dir
    settings.paths.log_dir = tmp_test_dir / "logs"
    settings.paths.checkpoint_dir = tmp_test_dir / "checkpoints"
    settings.limits.rate_limits.per_user.max_concurrent = 3
    settings.limits.rate_limits.per_user.max_per_hour = 10
    settings.limits.rate_limits.per_user.cooldown_seconds = 60
    settings.dev.debug = False
    settings.dev.log_level = "INFO"
    return settings


@pytest.fixture
def rate_limiter():
    """Create a rate limiter instance for testing."""
    config = RateLimitConfig(
        max_concurrent=3,
        max_per_hour=10,
        cooldown_seconds=60
    )
    return RateLimiter(config)


@pytest.fixture
def mock_task_queue():
    """Create a mock task queue."""
    queue = MagicMock(spec=TaskQueue)
    queue.is_full = False
    queue.size = 0
    queue.max_size = 50
    queue.can_user_submit = MagicMock(return_value=True)
    queue.submit = AsyncMock(return_value=True)
    queue.get_user_tasks = MagicMock(return_value=[])
    queue.get_task = MagicMock(return_value=None)
    queue.cancel_task = AsyncMock(return_value=True)
    queue.get_queue_stats = MagicMock(return_value={
        "queue_size": 0,
        "max_size": 50,
        "total_tasks": 0,
        "active_users": 0,
        "tasks_by_status": {}
    })
    return queue


@pytest.fixture
def mock_worker(mock_task_queue):
    """Create a mock worker instance."""
    worker = MagicMock(spec=Worker)
    worker.queue = mock_task_queue
    worker.gpu_monitor = MagicMock()
    worker.gpu_monitor.get_stats = MagicMock(return_value=MagicMock(
        is_available=True,
        device_name="Mock GPU",
        total_memory_gb=16.0,
        allocated_memory_gb=2.0,
        free_memory_gb=14.0,
        utilization_percent=12.5
    ))
    worker.get_status = MagicMock(return_value={
        "is_running": True,
        "current_task": None,
        "queue_stats": {"queue_size": 0, "max_size": 50}
    })
    worker.start = AsyncMock()
    worker.stop = AsyncMock()
    return worker


@pytest.fixture
def mock_file_manager(tmp_test_dir):
    """Create a mock file manager."""
    file_manager = MagicMock()
    file_manager.temp_dir = tmp_test_dir
    file_manager.check_user_quota = MagicMock(return_value=True)
    file_manager.get_user_dir = MagicMock(return_value=tmp_test_dir / "user")
    file_manager.get_task_dir = MagicMock(return_value=tmp_test_dir / "task")
    file_manager.get_results_dir = MagicMock(return_value=tmp_test_dir / "results")
    file_manager.cleanup_user_files = MagicMock()
    return file_manager


@pytest.fixture
def sample_audio_file(tmp_test_dir):
    """Create a sample audio file for testing."""
    audio_file = tmp_test_dir / "test_audio.mp3"
    # Create fake MP3 file with valid header
    with open(audio_file, "wb") as f:
        f.write(b"\xFF\xFB" + b"\x00" * 1000)
    return audio_file


@pytest.fixture
def sample_wav_file(tmp_test_dir):
    """Create a sample WAV file for testing."""
    wav_file = tmp_test_dir / "test_audio.wav"
    # Create fake WAV file with valid header
    with open(wav_file, "wb") as f:
        f.write(b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 1000)
    return wav_file


@pytest.fixture
def mock_telegram_user():
    """Create a mock Telegram user."""
    from aiogram.types import User
    user = MagicMock(spec=User)
    user.id = 12345
    user.first_name = "Test"
    user.last_name = "User"
    user.username = "testuser"
    user.is_bot = False
    return user


@pytest.fixture
def mock_telegram_chat():
    """Create a mock Telegram chat."""
    from aiogram.types import Chat
    chat = MagicMock(spec=Chat)
    chat.id = 12345
    chat.type = "private"
    return chat


@pytest.fixture
def mock_telegram_message(mock_telegram_user, mock_telegram_chat):
    """Create a mock Telegram message."""
    from aiogram.types import Message
    message = MagicMock(spec=Message)
    message.from_user = mock_telegram_user
    message.chat = mock_telegram_chat
    message.message_id = 1
    message.date = MagicMock()
    message.answer = AsyncMock()
    message.reply = AsyncMock()
    message.edit_text = AsyncMock()
    message.audio = None
    message.voice = None
    message.document = None
    message.video_note = None
    return message


@pytest.fixture
def mock_bot():
    """Create a mock Telegram bot."""
    from aiogram import Bot
    bot = MagicMock(spec=Bot)
    bot.token = "test_token"
    bot.get_file = AsyncMock()
    bot.download_file = AsyncMock()
    bot.send_message = AsyncMock()
    bot.edit_message_text = AsyncMock()
    bot.send_document = AsyncMock()
    bot.delete_message = AsyncMock()
    return bot


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # This prevents state leakage between tests
    yield
    # Cleanup code here if needed


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
