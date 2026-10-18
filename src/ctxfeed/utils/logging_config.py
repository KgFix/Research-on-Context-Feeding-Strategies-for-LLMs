# simple console logger setup for eval runs
import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    # basic console format with timestamps

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
