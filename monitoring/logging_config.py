# monitoring/logging_config.py

import logging
import sys
import os
from pythonjsonlogger import jsonlogger


def setup_logging(
    log_level: str = "INFO",
    log_file:  str = "logs/app.log"
):
    os.makedirs("logs", exist_ok=True)

    formatter = jsonlogger.JsonFormatter(
        fmt    = "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt = "%Y-%m-%dT%H:%M:%S"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(
        getattr(logging, log_level.upper(), logging.INFO)
    )
    root_logger.handlers.clear()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
