import logging
import json
from datetime import datetime


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage()
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(log_record)


def setup_logger(name: str):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(JsonFormatter())

        logger.addHandler(console_handler)

    return logger