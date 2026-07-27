import logging
from pathlib import Path

from src.utils.config import config


def get_logger(name):

    Path("logs").mkdir(exist_ok=True)

    logger = logging.getLogger(name)

    logger.setLevel(config.config["logging"]["level"])

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(config.config["logging"]["file"])

    console_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    return logger
