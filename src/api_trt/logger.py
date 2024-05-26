import logging
import os
from logging import handlers
import colorlog


def configure_logger(name: str):
    log_level = os.getenv('LOG_LEVEL', 'INFO')

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Set the 'isatty' attribute to True to force color detection.
    console_handler.isatty = lambda: True

    # Define the format for log messages with color.
    formatter = colorlog.ColoredFormatter(
        '[{asctime}]{log_color}[{levelname:^8s}] ({filename}:{lineno} ({funcName})): {message}',
        style='{',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red'
        })

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = configure_logger(__name__)
