import logging
from pathlib import Path


def get_logger(name: str):

    Path("logs").mkdir(exist_ok=True)

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        file_handler = logging.FileHandler("logs/project.log")

        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger