import asyncio
import time
from collections import defaultdict, deque


class FixedWindowRateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.requests_per_minute = requests_per_minute
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - 60
        async with self._lock:
            timestamps = self._requests[key]
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()
            if len(timestamps) >= self.requests_per_minute:
                return False
            timestamps.append(now)
            return True
