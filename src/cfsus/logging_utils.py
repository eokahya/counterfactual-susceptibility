"""Small, idempotent logging setup for future command-line entry points."""

from __future__ import annotations

import logging
from typing import TextIO

PROJECT_LOGGER_NAME = "cfsus"


def configure_logging(
    level: int = logging.INFO, *, stream: TextIO | None = None
) -> logging.Logger:
    """Configure and return the project logger without touching the root logger."""

    logger = logging.getLogger(PROJECT_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    owned_handlers = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_cfsus_owned", False)
    ]
    if owned_handlers:
        handler = owned_handlers[0]
        handler.setLevel(level)
        return logger

    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler._cfsus_owned = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the project logger or one of its children."""

    if name is None or not name.strip():
        return logging.getLogger(PROJECT_LOGGER_NAME)
    return logging.getLogger(f"{PROJECT_LOGGER_NAME}.{name}")
