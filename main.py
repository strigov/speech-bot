"""Application entry point for GigaAM & PyAnnote Telegram Transcriber."""

import asyncio
import signal
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_settings
from src.utils.logging import setup_logging, get_logger


async def main() -> None:
    """Main application entry point."""
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
    from src.bot.handlers import create_bot, run_bot

    try:
        # Create and run bot
        bot, dp = create_bot(settings)
        await run_bot(bot, dp)
    except Exception as e:
        logger.exception("Fatal error occurred", error=str(e))
        raise


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
