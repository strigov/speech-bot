"""Application entry point for GigaAM & PyAnnote Telegram Transcriber."""

import asyncio
import os
import signal
import sys
from pathlib import Path

# CRITICAL: Add FFmpeg DLL directory for Python 3.8+ (before any imports that use FFmpeg)
# This is required for PyAnnote's AudioDecoder/torchcodec to find FFmpeg libraries
_ffmpeg_bin = r"D:\Projects\ffmpeg-6.1-full_build-shared\bin"
if os.path.exists(_ffmpeg_bin):
    os.add_dll_directory(_ffmpeg_bin)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_settings
from src.utils.logging import setup_logging, get_logger
from src.utils.dependencies import check_dependencies


class SingleInstanceLock:
    def __init__(self, lockfile="bot.lock"):
        self.lockfile = Path(lockfile)
        self.lockfile_fd = None

    def __enter__(self):
        if self.lockfile.exists():
            try:
                pid = int(self.lockfile.read_text().strip())
                # Check if process is running
                try:
                    os.kill(pid, 0)
                    print(f"ERROR: Another instance is running (PID {pid}).")
                    sys.exit(1)
                except (OSError, SystemError):
                    # Process not running or access denied (stale lock or zombie)
                    # On Windows, os.kill(pid, 0) can be flaky, so we might want to be careful.
                    # If we get here, we assume the process is dead or we can't check it.
                    # But if it WAS running, os.kill usually succeeds.
                    # If it raises SystemError, it's likely dead or invalid PID.
                    pass
                except Exception as e:
                    # Fallback for any other error
                    print(f"WARNING: Failed to check lock file PID: {e}")
                    pass
            except ValueError:
                pass
        
        self.lockfile.write_text(str(os.getpid()))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lockfile.exists():
            try:
                # Only delete if it contains our PID
                if int(self.lockfile.read_text().strip()) == os.getpid():
                    self.lockfile.unlink()
            except (ValueError, OSError):
                pass


async def main() -> None:
    """Main application entry point."""
    # Check dependencies first (before logging setup)
    deps_ok, deps_error = check_dependencies()
    if not deps_ok:
        print("\n" + "="*60)
        print("ERROR: Missing system dependencies")
        print("="*60)
        print(deps_error)
        print("="*60 + "\n")
        sys.exit(1)

    # Load settings
    settings = get_settings()

    # Setup logging
    setup_logging(
        log_level=settings.dev.log_level,
        log_dir=settings.paths.log_dir if not settings.dev.debug else None,
    )

    logger = get_logger(__name__)
    logger.info("Starting GigaAM & PyAnnote Telegram Transcriber")
    logger.info(
        "Configuration loaded",
        debug=settings.dev.debug,
        device=settings.model.device,
        max_vram_gb=settings.memory.max_vram_gb,
    )

    # Import bot components (lazy import to avoid loading models at startup)
    from src.bot.handlers import create_bot, run_bot, setup_worker
    from src.worker import create_worker

    try:
        # Create worker and wire into bot handlers
        worker = create_worker(
            temp_dir=settings.paths.temp_dir,
            checkpoint_dir=settings.paths.checkpoint_dir,
            hf_token=settings.model.hf_token,
            max_queue_size=settings.processing.max_queue_size,
            max_per_user=settings.processing.max_user_concurrent,
            max_vram_gb=settings.memory.max_vram_gb,
            timeout_seconds=settings.processing.processing_timeout_minutes * 60,
        )

        # Create and run bot
        bot, dp = create_bot(settings)
        setup_worker(worker)
        
        # Use lock file to prevent multiple instances
        with SingleInstanceLock():
            await worker.start()
            await run_bot(bot, dp)
    except Exception as e:
        logger.exception("Fatal error occurred", error=str(e))
        raise
    finally:
        try:
            await worker.stop()
        except Exception:
            pass


def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    logger = get_logger(__name__)
    logger.info("Shutdown signal received", signal=signum)
    sys.exit(0)


if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Run the application
    asyncio.run(main())
