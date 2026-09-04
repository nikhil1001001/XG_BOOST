"""Shared logging setup. get_logger(name) configures stdout + per-phase file handlers (see brief Sec 12)."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False


def setup_logging(phase: str, logs_dir: str | Path = "logs", level: int = logging.INFO) -> Path:
    """Configure the root logger once per process: a StreamHandler to stdout plus a FileHandler
    writing to logs/<phase>_<timestamp>.log. Safe to call multiple times per process (e.g. from
    tests) — only the first call attaches the stdout handler; each call attaches its own file
    handler so re-running a script mid-session still gets a fresh log file.
    """
    global _configured
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"{phase}_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(level)

    if not _configured:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(stream_handler)
        _configured = True

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(file_handler)

    return log_path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
