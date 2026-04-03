"""
DocuMind AI — Logging Configuration
Provides consistent structured logging across all modules.
"""

import logging
import sys
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for any module.
    Usage: logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Console handler — human-readable format
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
