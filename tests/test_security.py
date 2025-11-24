import unittest
import time
import shutil
from pathlib import Path
from src.utils.rate_limiter import RateLimiter, RateLimitConfig
from src.utils.validation import FileValidator

class TestSecurity(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_security")
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_rate_limiter(self):
        config = RateLimitConfig(max_concurrent=3, max_per_hour=2, cooldown_seconds=1)
        limiter = RateLimiter(config)
        user_id = 12345

        # First request
        allowed, _ = limiter.check_user(user_id)
        self.assertTrue(allowed)
        limiter.record_request(user_id)

        # Immediate second request (should fail due to cooldown)
        allowed, info = limiter.check_user(user_id)
        self.assertFalse(allowed)
        self.assertEqual(info["reason"], "cooldown")

        # Wait for cooldown
        time.sleep(1.1)
        allowed, _ = limiter.check_user(user_id)
        self.assertTrue(allowed)
        limiter.record_request(user_id)

        # Third request (should fail due to hourly limit)
        time.sleep(1.1)
        allowed, info = limiter.check_user(user_id)
        self.assertFalse(allowed)
        self.assertEqual(info["reason"], "hourly_limit")

    def test_file_validator(self):
        # Create a fake MP3 file
        mp3_path = self.test_dir / "test.mp3"
        with open(mp3_path, "wb") as f:
            f.write(b"\xFF\xFB" + b"\x00" * 100)
        
        is_valid, _ = FileValidator.validate_file(mp3_path)
        self.assertTrue(is_valid)

        # Create a fake invalid file
        invalid_path = self.test_dir / "test.txt"
        with open(invalid_path, "wb") as f:
            f.write(b"Hello World")
        
        is_valid, msg = FileValidator.validate_file(invalid_path)
        self.assertFalse(is_valid)
        self.assertIn("Invalid file format", msg)

    def test_path_traversal(self):
        base_dir = self.test_dir
        safe_path = base_dir / "safe.txt"
        unsafe_path = base_dir / "../unsafe.txt"
        
        self.assertTrue(FileValidator.is_safe_path(base_dir, safe_path))
        self.assertFalse(FileValidator.is_safe_path(base_dir, unsafe_path))

if __name__ == "__main__":
    unittest.main()
