import queue
import asyncio
from typing import Any
from enum import Enum

class RequestType(Enum):
    PREDICT = "predict"
    ADD_SAMPLE = "add_sample"

class RequestQueue:
    """Thread-safe queue for API requests."""
    
    def __init__(self):
        self._queue = queue.Queue()
    
    def put(self, request_type: RequestType, data: dict) -> asyncio.Future:
        """Add request and return future for result."""
        future = asyncio.Future()
        self._queue.put((request_type, data, future))
        return future
    
    def get_all(self) -> list:
        """Get all pending requests (non-blocking)."""
        requests = []
        while not self._queue.empty():
            try:
                requests.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return requests
    
    def is_empty(self) -> bool:
        return self._queue.empty()