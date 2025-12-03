"""Rate limiting utility for tracking user request frequency."""

import time
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, Set

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_concurrent: int = 3
    max_per_hour: int = 10
    cooldown_seconds: int = 60

class RateLimiter:
    """
    Tracks and enforces rate limits for users.

    Implements:
    - Hourly limit (sliding window)
    - Cooldown between requests
    - Admin bypass (admins are not rate limited)
    """

    def __init__(self, config: RateLimitConfig, admin_ids: Optional[Set[int]] = None):
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration
            admin_ids: Set of admin user IDs who bypass rate limits
        """
        self.config = config
        self.admin_ids = admin_ids or set()
        # Stores timestamps of requests for each user
        self._user_requests: Dict[int, deque] = defaultdict(deque)
        # Stores timestamp of last request for each user
        self._last_request_time: Dict[int, float] = defaultdict(float)

    def check_user(self, user_id: int) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if user is allowed to make a request.

        Args:
            user_id: User ID to check

        Returns:
            Tuple of (is_allowed, info_dict)
            info_dict contains 'reason' and details if not allowed
        """
        # Admins bypass all rate limits
        if user_id in self.admin_ids:
            return True, {"is_admin": True}

        now = time.time()

        # Check cooldown
        last_request = self._last_request_time[user_id]
        if now - last_request < self.config.cooldown_seconds:
             remaining = int(self.config.cooldown_seconds - (now - last_request))
             return False, {
                 "reason": "cooldown",
                 "remaining": remaining,
                 "message": f"Please wait {remaining}s before sending another file."
             }

        # Check hourly limit
        requests = self._user_requests[user_id]
        # Remove requests older than 1 hour
        while requests and requests[0] < now - 3600:
            requests.popleft()

        if len(requests) >= self.config.max_per_hour:
             return False, {
                 "reason": "hourly_limit",
                 "limit": self.config.max_per_hour,
                 "message": f"Hourly limit reached ({self.config.max_per_hour} files/hour)."
             }

        return True, {}

    def record_request(self, user_id: int) -> None:
        """
        Record a successful request for a user.
        
        Args:
            user_id: User ID to record
        """
        now = time.time()
        self._last_request_time[user_id] = now
        self._user_requests[user_id].append(now)
