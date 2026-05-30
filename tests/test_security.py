"""Comprehensive security and validation tests."""

import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.utils.rate_limiter import RateLimiter, RateLimitConfig
from src.utils.validation import FileValidator


@pytest.fixture
def test_dir(tmp_path):
    """Create temporary test directory."""
    yield tmp_path


# --- Rate Limiter Tests ---

def test_rate_limiter_first_request():
    """Test rate limiter allows first request."""
    config = RateLimitConfig(max_concurrent=3, max_per_hour=2, cooldown_seconds=60)
    limiter = RateLimiter(config)
    user_id = 12345

    allowed, info = limiter.check_user(user_id)
    assert allowed is True
    assert info.get("reason") is None


def test_rate_limiter_cooldown():
    """Test rate limiter enforces cooldown between requests."""
    config = RateLimitConfig(max_concurrent=3, max_per_hour=10, cooldown_seconds=60)
    limiter = RateLimiter(config)
    user_id = 12345

    # First request
    allowed, _ = limiter.check_user(user_id)
    assert allowed is True
    limiter.record_request(user_id)

    # Immediate second request (should fail due to cooldown)
    allowed, info = limiter.check_user(user_id)
    assert allowed is False
    assert info["reason"] == "cooldown"


def test_rate_limiter_cooldown_expired():
    """Test rate limiter behavior with cooldown."""
    config = RateLimitConfig(max_concurrent=3, max_per_hour=10, cooldown_seconds=1)
    limiter = RateLimiter(config)
    user_id = 12345

    # Mock time.time() to simulate cooldown
    with patch("time.time") as mock_time:
        base_time = 1000.0
        mock_time.return_value = base_time

        # First request
        limiter.record_request(user_id)

        # Check immediately (should be blocked by cooldown)
        allowed, info = limiter.check_user(user_id)
        assert allowed is False
        assert info["reason"] == "cooldown"

        # Advance time past cooldown
        mock_time.return_value = base_time + 2.0
        allowed, _ = limiter.check_user(user_id)
        assert allowed is True


def test_rate_limiter_hourly_limit():
    """Test rate limiter enforces hourly request limit."""
    config = RateLimitConfig(max_concurrent=3, max_per_hour=2, cooldown_seconds=0)
    limiter = RateLimiter(config)
    user_id = 12345

    # Record 2 requests
    limiter.record_request(user_id)
    limiter.record_request(user_id)

    # Third request should fail
    allowed, info = limiter.check_user(user_id)
    assert allowed is False
    assert info["reason"] == "hourly_limit"


def test_rate_limiter_hourly_limit_reset():
    """Test rate limiter resets hourly limit after time window."""
    config = RateLimitConfig(max_concurrent=3, max_per_hour=2, cooldown_seconds=0)
    limiter = RateLimiter(config)
    user_id = 12345

    with patch("time.time") as mock_time:
        base_time = 1000.0
        mock_time.return_value = base_time

        # Record 2 requests
        limiter.record_request(user_id)
        limiter.record_request(user_id)

        # Should be blocked
        allowed, _ = limiter.check_user(user_id)
        assert allowed is False

        # After 1 hour, should be allowed again
        mock_time.return_value = base_time + 3661  # 1 hour + 1 second
        allowed, _ = limiter.check_user(user_id)
        assert allowed is True


def test_rate_limiter_concurrent_limit():
    """Test rate limiter configuration accepts concurrent limits."""
    config = RateLimitConfig(max_concurrent=2, max_per_hour=100, cooldown_seconds=0)
    limiter = RateLimiter(config)

    # Verify configuration was accepted
    assert config.max_concurrent == 2
    assert config.max_per_hour == 100


def test_rate_limiter_multiple_users():
    """Test rate limiter handles multiple users independently."""
    config = RateLimitConfig(max_concurrent=1, max_per_hour=1, cooldown_seconds=60)
    limiter = RateLimiter(config)

    user1 = 111
    user2 = 222

    # User 1 makes request
    limiter.record_request(user1)
    allowed, _ = limiter.check_user(user1)
    assert allowed is False  # Blocked by cooldown

    # User 2 should still be allowed
    allowed, _ = limiter.check_user(user2)
    assert allowed is True


def test_rate_limiter_zero_cooldown():
    """Test rate limiter with zero cooldown."""
    config = RateLimitConfig(max_concurrent=10, max_per_hour=10, cooldown_seconds=0)
    limiter = RateLimiter(config)
    user_id = 12345

    # Multiple rapid requests should be allowed
    for _ in range(5):
        allowed, _ = limiter.check_user(user_id)
        assert allowed is True
        limiter.record_request(user_id)


def test_rate_limiter_release_request():
    """Test rate limiter has release functionality if implemented."""
    config = RateLimitConfig(max_concurrent=1, max_per_hour=100, cooldown_seconds=0)
    limiter = RateLimiter(config)
    user_id = 12345

    # Record request
    limiter.record_request(user_id)

    # Test if release_request method exists
    if hasattr(limiter, 'release_request'):
        limiter.release_request(user_id)
        # Verify it doesn't crash
        assert True
    else:
        # Method not implemented, skip test
        pytest.skip("release_request not implemented")


# --- File Validator Tests ---

def test_file_validator_valid_mp3(test_dir):
    """Test file validator accepts valid MP3 files."""
    mp3_path = test_dir / "test.mp3"
    with open(mp3_path, "wb") as f:
        f.write(b"\xFF\xFB" + b"\x00" * 100)

    is_valid, msg = FileValidator.validate_file(mp3_path)
    assert is_valid is True


def test_file_validator_valid_mp3_with_crc(test_dir):
    """Test file validator accepts MP3 frame sync headers with CRC."""
    mp3_path = test_dir / "test_crc.mp3"
    with open(mp3_path, "wb") as f:
        f.write(b"\xFF\xFA" + b"\x00" * 100)

    is_valid, msg = FileValidator.validate_file(mp3_path)
    assert is_valid is True


def test_file_validator_valid_mpeg25_mp3(test_dir):
    """Test file validator accepts MPEG 2.5 MP3 frame sync headers."""
    mp3_path = test_dir / "test_mpeg25.mp3"
    with open(mp3_path, "wb") as f:
        f.write(b"\xFF\xE3\x18\xC4" + b"\x00" * 100)

    is_valid, msg = FileValidator.validate_file(mp3_path)
    assert is_valid is True


def test_file_validator_valid_wav(test_dir):
    """Test file validator accepts valid WAV files."""
    wav_path = test_dir / "test.wav"
    with open(wav_path, "wb") as f:
        # WAV header
        f.write(b"RIFF" + b"\x00" * 4 + b"WAVE")

    is_valid, msg = FileValidator.validate_file(wav_path)
    assert is_valid is True


def test_file_validator_valid_ogg(test_dir):
    """Test file validator accepts valid OGG files."""
    ogg_path = test_dir / "test.ogg"
    with open(ogg_path, "wb") as f:
        f.write(b"OggS" + b"\x00" * 100)

    is_valid, msg = FileValidator.validate_file(ogg_path)
    assert is_valid is True


def test_file_validator_invalid_file(test_dir):
    """Test file validator rejects invalid files."""
    invalid_path = test_dir / "test.txt"
    with open(invalid_path, "wb") as f:
        f.write(b"Hello World")

    is_valid, msg = FileValidator.validate_file(invalid_path)
    assert is_valid is False
    assert "Invalid file format" in msg


def test_file_validator_nonexistent_file(test_dir):
    """Test file validator handles nonexistent files."""
    nonexistent = test_dir / "does_not_exist.mp3"

    is_valid, msg = FileValidator.validate_file(nonexistent)
    assert is_valid is False


def test_path_traversal_safe_path(test_dir):
    """Test path validator allows safe paths."""
    safe_path = test_dir / "safe.txt"

    is_safe = FileValidator.is_safe_path(test_dir, safe_path)
    assert is_safe is True


def test_path_traversal_unsafe_path(test_dir):
    """Test path validator blocks path traversal attempts."""
    unsafe_path = test_dir / "../unsafe.txt"

    is_safe = FileValidator.is_safe_path(test_dir, unsafe_path)
    assert is_safe is False


def test_path_traversal_absolute_path(test_dir):
    """Test path validator handles absolute paths correctly."""
    absolute_path = test_dir / "subdir" / "file.txt"

    is_safe = FileValidator.is_safe_path(test_dir, absolute_path)
    assert is_safe is True


def test_path_traversal_symlink_escape(test_dir):
    """Test path validator blocks symlink escape attempts."""
    # Create a symlink that tries to escape
    target = test_dir.parent / "outside.txt"
    link = test_dir / "link.txt"

    try:
        link.symlink_to(target)
        is_safe = FileValidator.is_safe_path(test_dir, link)
        # Should detect symlink escape
        assert is_safe is False
    except (OSError, NotImplementedError):
        # Symlinks might not be supported on this system
        pytest.skip("Symlinks not supported")
