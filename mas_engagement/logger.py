import json
import os
import threading
import time
from datetime import datetime
from typing import Any


class JsonlLogger:
    """Append-only JSONL session logger; one JSON object per line."""

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(log_dir, f"session_{session_id}.jsonl")
        self._lock = threading.Lock()

    def log(self, agent: str, event: str, **kwargs: Any) -> None:
        """Append a single event record with fields {ts, agent, event, **kwargs}."""
        record = {"ts": time.time(), "agent": agent, "event": event, **kwargs}
        line = json.dumps(record) + "\n"
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line)

    @property
    def path(self) -> str:
        """Return the absolute path of the current log file."""
        return self._path
