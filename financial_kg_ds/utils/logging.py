import logging
import sys
from pathlib import Path
from typing import Optional

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(
    name: str = "financial_kg_ds",
    level: int = logging.INFO,
    log_file: Optional[str] = "logs/training.log",
    fmt: str = DEFAULT_LOG_FORMAT,
) -> logging.Logger:
    """Return a configured logger shared across the project.

    The logger is configured once per name. Subsequent calls return the same
    logger without adding duplicate handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(fmt)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Optional file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger