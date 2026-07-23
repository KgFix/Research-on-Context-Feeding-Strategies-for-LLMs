"""Logging configuration for the ctxfeed pipeline.

Call setup_logging() once at pipeline startup to configure
a clean, timestamped console format for all modules.
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a clean console format.

    Args:
        level: Logging level (default INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
