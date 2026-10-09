"""Structured (JSON lines) logging to a file. The terminal stays clean for the user."""

from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path

_STANDARD_ATTRS = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Anything passed via `extra=` becomes a structured field.
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                entry[key] = value
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


def setup_logging(log_file: Path, level: str = "INFO") -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("atlas")
    for old in list(root.handlers):
        root.removeHandler(old)
        old.close()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
