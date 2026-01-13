import datetime
import logging


class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",    # Cyan
        logging.INFO: "\033[32m",     # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",    # Red
        logging.CRITICAL: "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        record.colored_levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)

    def formatTime(self, record: logging.LogRecord, datefmt: str = None) -> str:
        dt = datetime.datetime.fromtimestamp(record.created)
        return f"{dt.strftime('%Y-%m-%d %H:%M:%S')}.{int(record.msecs):03d}"


# Configure logging on import
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter("[%(asctime)s] :: %(colored_levelname)s :: %(message)s"))

root = logging.getLogger()
root.handlers.clear()
root.addHandler(handler)
root.setLevel(logging.WARNING)

# Enable DEBUG only for project loggers
logging.getLogger("src").setLevel(logging.DEBUG)
logging.getLogger("__main__").setLevel(logging.DEBUG)

# Global logger
g_logger = logging.getLogger("llm-from-scratch")
