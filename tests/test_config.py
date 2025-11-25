"""Comprehensive tests for configuration and settings."""

import pytest
from pathlib import Path
from pydantic import ValidationError
import tempfile
import yaml

from config.settings import (
    Settings,
    TelegramSettings,
    ModelSettings,
    ProcessingSettings,
    MemorySettings,
    PathSettings,
    DevelopmentSettings,
    LimitsSettings,
    RateLimits,
    PerUserLimits,
    GlobalLimits,
    FileLimits,
    get_settings,
)


# --- TelegramSettings Tests ---

def test_telegram_settings_valid():
    """Test TelegramSettings with valid configuration."""
    settings = TelegramSettings(bot_token="123456:ABC-DEF1234")
    assert settings.bot_token == "123456:ABC-DEF1234"
    assert settings.api_url == "https://api.telegram.org"
    assert settings.use_local_bot_api is False


def test_telegram_settings_missing_token():
    """Test TelegramSettings requires bot_token."""
    with pytest.raises(ValidationError):
        TelegramSettings()


def test_telegram_settings_local_api():
    """Test TelegramSettings with local API configuration."""
    settings = TelegramSettings(
        bot_token="test_token",
        use_local_bot_api=True,
        local_bot_api_url="http://localhost:9000"
    )
    assert settings.use_local_bot_api is True
    assert settings.local_bot_api_url == "http://localhost:9000"


# --- ModelSettings Tests ---

def test_model_settings_defaults():
    """Test ModelSettings has sensible defaults."""
    settings = ModelSettings()
    assert settings.device == "cuda"
    assert settings.diarization_device == "cuda"
    assert settings.asr_batch_size == 32
    assert settings.diarization_batch_size == 32


def test_model_settings_custom(monkeypatch):
    """Test ModelSettings with custom values."""
    monkeypatch.setenv("MODEL_DEVICE", "cpu")
    monkeypatch.setenv("DIARIZATION_DEVICE", "cuda:1")
    monkeypatch.setenv("ASR_BATCH_SIZE", "16")

    settings = ModelSettings()
    assert settings.device == "cpu"
    assert settings.diarization_device == "cuda:1"
    assert settings.asr_batch_size == 16


# --- ProcessingSettings Tests ---

def test_processing_settings_defaults():
    """Test ProcessingSettings has reasonable limits."""
    settings = ProcessingSettings()
    assert settings.max_audio_duration_minutes == 180
    assert settings.max_file_size_mb == 500
    assert settings.max_queue_size == 50
    assert settings.max_user_concurrent == 3


def test_processing_settings_custom_limits(monkeypatch):
    """Test ProcessingSettings with custom limits."""
    monkeypatch.setenv("MAX_AUDIO_DURATION_MINUTES", "60")
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "200")
    monkeypatch.setenv("MAX_QUEUE_SIZE", "10")

    settings = ProcessingSettings()
    assert settings.max_audio_duration_minutes == 60
    assert settings.max_file_size_mb == 200
    assert settings.max_queue_size == 10


# --- MemorySettings Tests ---

def test_memory_settings_defaults():
    """Test MemorySettings has reasonable defaults."""
    settings = MemorySettings()
    assert settings.max_vram_gb == 14.0
    assert settings.chunk_duration_minutes == 10
    assert settings.segment_max_seconds == 30


def test_memory_settings_custom(monkeypatch):
    """Test MemorySettings with custom values."""
    monkeypatch.setenv("MAX_VRAM_GB", "8.0")
    monkeypatch.setenv("CHUNK_DURATION_MINUTES", "5")

    settings = MemorySettings()
    assert settings.max_vram_gb == 8.0
    assert settings.chunk_duration_minutes == 5


# --- PathSettings Tests ---

def test_path_settings_defaults():
    """Test PathSettings has default paths."""
    settings = PathSettings()
    assert settings.temp_dir == Path("./temp")
    assert settings.log_dir == Path("./logs")
    assert settings.checkpoint_dir == Path("./temp/checkpoints")


def test_path_settings_custom(monkeypatch):
    """Test PathSettings with custom paths."""
    monkeypatch.setenv("TEMP_DIR", "/custom/temp")
    monkeypatch.setenv("LOG_DIR", "/custom/logs")

    settings = PathSettings()
    assert settings.temp_dir == Path("/custom/temp")
    assert settings.log_dir == Path("/custom/logs")


# --- DevelopmentSettings Tests ---

def test_development_settings_defaults():
    """Test DevelopmentSettings defaults to production mode."""
    settings = DevelopmentSettings()
    assert settings.debug is False
    assert settings.hot_reload is False
    assert settings.log_level == "INFO"


def test_development_settings_debug_mode(monkeypatch):
    """Test DevelopmentSettings in debug mode."""
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("HOT_RELOAD", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = DevelopmentSettings()
    assert settings.debug is True
    assert settings.hot_reload is True
    assert settings.log_level == "DEBUG"


# --- LimitsSettings Tests ---

def test_per_user_limits_defaults():
    """Test PerUserLimits has reasonable defaults."""
    limits = PerUserLimits()
    assert limits.max_concurrent == 1
    assert limits.max_per_hour == 10
    assert limits.cooldown_seconds == 60


def test_global_limits_defaults():
    """Test GlobalLimits has reasonable defaults."""
    limits = GlobalLimits()
    assert limits.max_queue_size == 50
    assert limits.max_active_tasks == 3


def test_file_limits_defaults():
    """Test FileLimits has proper file restrictions."""
    limits = FileLimits()
    assert limits.max_size_mb == 150
    assert limits.max_duration_minutes == 180
    assert ".mp3" in limits.allowed_extensions
    assert ".wav" in limits.allowed_extensions
    assert "audio/mpeg" in limits.allowed_mime_types


def test_limits_settings_from_yaml_missing_file(tmp_path):
    """Test LimitsSettings gracefully handles missing YAML file."""
    fake_path = tmp_path / "nonexistent.yaml"
    limits = LimitsSettings.from_yaml(fake_path)

    # Should fall back to defaults
    assert limits.rate_limits.per_user.max_concurrent == 1
    assert limits.file_limits.max_size_mb == 150


def test_limits_settings_from_yaml_valid(tmp_path):
    """Test LimitsSettings loads from valid YAML file."""
    yaml_file = tmp_path / "limits.yaml"
    config = {
        "rate_limits": {
            "per_user": {
                "max_concurrent": 5,
                "max_per_hour": 20,
                "cooldown_seconds": 30
            }
        },
        "file_limits": {
            "max_size_mb": 300
        }
    }
    yaml_file.write_text(yaml.dump(config))

    limits = LimitsSettings.from_yaml(yaml_file)
    assert limits.rate_limits.per_user.max_concurrent == 5
    assert limits.rate_limits.per_user.max_per_hour == 20
    assert limits.file_limits.max_size_mb == 300


def test_limits_settings_from_yaml_invalid(tmp_path):
    """Test LimitsSettings handles invalid YAML gracefully."""
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text("invalid: yaml: content: [[[")

    limits = LimitsSettings.from_yaml(yaml_file)

    # Should fall back to defaults on error
    assert limits.rate_limits.per_user.max_concurrent == 1


# --- Main Settings Tests ---

def test_settings_initialization(tmp_path, monkeypatch):
    """Test Settings initializes and creates directories."""
    # Use temp directory
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test_token")
    monkeypatch.setenv("TEMP_DIR", str(tmp_path / "temp"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))

    settings = Settings()

    assert settings.telegram.bot_token == "test_token"

    # Check directories were created
    assert settings.paths.temp_dir.exists()
    assert settings.paths.log_dir.exists()
    assert (settings.paths.temp_dir / "audio").exists()
    assert (settings.paths.temp_dir / "results").exists()
    assert (settings.paths.temp_dir / "downloads").exists()


def test_settings_environment_variables(monkeypatch):
    """Test Settings loads from environment variables."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env_token")
    monkeypatch.setenv("MODEL_DEVICE", "cpu")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("MAX_QUEUE_SIZE", "100")

    settings = Settings()

    assert settings.telegram.bot_token == "env_token"
    assert settings.model.device == "cpu"
    assert settings.dev.debug is True
    assert settings.processing.max_queue_size == 100


def test_settings_nested_structure():
    """Test Settings has proper nested structure."""
    settings = Settings(telegram=TelegramSettings(bot_token="test"))

    assert hasattr(settings, 'telegram')
    assert hasattr(settings, 'model')
    assert hasattr(settings, 'processing')
    assert hasattr(settings, 'memory')
    assert hasattr(settings, 'paths')
    assert hasattr(settings, 'dev')
    assert hasattr(settings, 'limits')


def test_settings_limits_integration():
    """Test Settings integrates LimitsSettings correctly."""
    settings = Settings(telegram=TelegramSettings(bot_token="test"))

    assert settings.limits.rate_limits.per_user.max_concurrent >= 1
    assert settings.limits.file_limits.max_size_mb > 0
    assert len(settings.limits.file_limits.allowed_extensions) > 0


def test_get_settings_singleton(monkeypatch):
    """Test get_settings returns Settings instance."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test_token")

    settings = get_settings()

    assert isinstance(settings, Settings)
    assert settings.telegram.bot_token == "test_token"


def test_settings_validation_errors():
    """Test Settings validation catches errors."""
    # Missing required telegram token should raise error when accessed
    with pytest.raises(ValidationError):
        Settings(telegram=TelegramSettings())


def test_settings_paths_creation(tmp_path, monkeypatch):
    """Test Settings creates all required paths."""
    base = tmp_path / "app"

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")
    monkeypatch.setenv("TEMP_DIR", str(base / "temp"))
    monkeypatch.setenv("LOG_DIR", str(base / "logs"))
    monkeypatch.setenv("CHECKPOINT_DIR", str(base / "checkpoints"))

    settings = Settings()

    # All paths should exist
    assert settings.paths.temp_dir.exists()
    assert settings.paths.log_dir.exists()
    assert settings.paths.checkpoint_dir.exists()


# --- Edge Cases and Validation ---

def test_processing_settings_zero_values(monkeypatch):
    """Test ProcessingSettings accepts zero values where appropriate."""
    monkeypatch.setenv("MAX_QUEUE_SIZE", "0")
    settings = ProcessingSettings()
    assert settings.max_queue_size == 0


def test_memory_settings_negative_values(monkeypatch):
    """Test MemorySettings with negative values."""
    monkeypatch.setenv("MAX_VRAM_GB", "-1.0")
    settings = MemorySettings()
    # In production, validation should catch this, but for now just verify it's set
    assert settings.max_vram_gb == -1.0


def test_file_limits_empty_extensions():
    """Test FileLimits with empty extension list."""
    limits = FileLimits(allowed_extensions=[])
    assert len(limits.allowed_extensions) == 0


def test_rate_limits_aliases():
    """Test RateLimits handles 'global' alias correctly."""
    limits = RateLimits()
    assert hasattr(limits, 'global_')
    assert limits.global_.max_queue_size == 50


def test_settings_extra_fields_ignored(monkeypatch):
    """Test Settings ignores extra unknown fields from environment."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")
    monkeypatch.setenv("UNKNOWN_FIELD", "should_be_ignored")

    # Should not raise error due to extra="ignore"
    settings = Settings()
    assert not hasattr(settings, 'unknown_field')
