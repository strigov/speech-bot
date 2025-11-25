"""Telegram Client API wrapper for downloading large files."""

import asyncio
from pathlib import Path
from typing import Optional

from telethon import TelegramClient
from telethon.tl.types import Document, MessageMediaDocument
import structlog

logger = structlog.get_logger(__name__)

# Bot API file size limit (20 MB)
BOT_API_FILE_SIZE_LIMIT = 20 * 1024 * 1024


class TelegramFileDownloader:
    """Download large files using Telegram Client API (MTProto)."""

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        session_name: str = "bot_session",
        session_dir: Optional[Path] = None,
    ):
        """
        Initialize Telegram Client.

        Args:
            api_id: Telegram API ID from my.telegram.org
            api_hash: Telegram API Hash from my.telegram.org
            session_name: Session file name
            session_dir: Directory for session files
        """
        self.api_id = api_id
        self.api_hash = api_hash

        # Set session path
        if session_dir:
            session_dir.mkdir(parents=True, exist_ok=True)
            session_path = session_dir / session_name
        else:
            session_path = session_name

        self.client = TelegramClient(str(session_path), api_id, api_hash)
        self._started = False
        self._lock = asyncio.Lock()

    async def start(self, bot_token: str) -> None:
        """
        Start the client and authorize as a bot.

        Args:
            bot_token: Bot token for authorization
        """
        async with self._lock:
            if not self._started:
                await self.client.start(bot_token=bot_token)
                self._started = True
                logger.info("telegram_client_started")

    async def stop(self) -> None:
        """Stop the client."""
        async with self._lock:
            if self._started:
                await self.client.disconnect()
                self._started = False
                logger.info("telegram_client_stopped")

    async def download_file(
        self,
        chat_id: int,
        message_id: int,
        destination: Path,
        progress_callback: Optional[callable] = None,
    ) -> bool:
        """
        Download file from Telegram using Client API.

        Args:
            chat_id: Chat ID where the message is
            message_id: Message ID containing the file
            destination: Path where to save the file
            progress_callback: Optional callback(current, total)

        Returns:
            True if successful, False otherwise
        """
        if not self._started:
            logger.error("telegram_client_not_started")
            return False

        try:
            # Get the message
            message = await self.client.get_messages(chat_id, ids=message_id)

            if not message:
                logger.error("message_not_found", chat_id=chat_id, message_id=message_id)
                return False

            # Check if message has media
            if not message.media:
                logger.error("message_has_no_media", message_id=message_id)
                return False

            # Create parent directory
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Download the file
            logger.info(
                "downloading_file_via_client_api",
                chat_id=chat_id,
                message_id=message_id,
                destination=str(destination),
            )

            await self.client.download_media(
                message.media,
                file=str(destination),
                progress_callback=progress_callback,
            )

            if not destination.exists():
                logger.error("download_failed_file_not_created", destination=str(destination))
                return False

            logger.info(
                "file_downloaded_via_client_api",
                destination=str(destination),
                size=destination.stat().st_size,
            )
            return True

        except Exception as e:
            logger.error(
                "download_error_client_api",
                error=str(e),
                chat_id=chat_id,
                message_id=message_id,
            )
            return False

    def is_started(self) -> bool:
        """Check if client is started."""
        return self._started

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.stop()


# Global client instance
_client: Optional[TelegramFileDownloader] = None


def init_telegram_client(
    api_id: int,
    api_hash: str,
    bot_token: str,
    session_name: str = "bot_session",
    session_dir: Optional[Path] = None,
) -> TelegramFileDownloader:
    """
    Initialize global Telegram Client.

    Args:
        api_id: Telegram API ID
        api_hash: Telegram API Hash
        bot_token: Bot token
        session_name: Session file name
        session_dir: Directory for session files

    Returns:
        TelegramFileDownloader instance
    """
    global _client

    if _client is None:
        _client = TelegramFileDownloader(
            api_id=api_id,
            api_hash=api_hash,
            session_name=session_name,
            session_dir=session_dir,
        )

        # Start in background
        asyncio.create_task(_client.start(bot_token))

    return _client


def get_telegram_client() -> Optional[TelegramFileDownloader]:
    """Get global Telegram Client instance."""
    return _client


async def should_use_client_api(file_size: int) -> bool:
    """
    Determine if Client API should be used for downloading.

    Args:
        file_size: File size in bytes

    Returns:
        True if file is larger than Bot API limit
    """
    return file_size > BOT_API_FILE_SIZE_LIMIT
