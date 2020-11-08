"""
Set of helpers.

(c) Nikola Jajcay
"""

import logging
import os
import time
from functools import wraps

PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.abspath(__file__), "..", "..", "..")
)
CODE_ROOT = os.path.join(PROJECT_ROOT, "code")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results_csv")
PLOTS_ROOT = os.path.join(PROJECT_ROOT, "results_figs")


LOG_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_EXT = ".log"


def set_logger(log_filename=None, log_level=logging.INFO):
    """
    Prepare logger.

    :param log_filename: filename for the log, if None, will not use logger
    :type log_filename: str|None
    :param log_level: logging level
    :type log_level: int
    """
    formatting = "[%(asctime)s] %(levelname)s: %(message)s"
    log_formatter = logging.Formatter(formatting, LOG_DATETIME_FORMAT)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = []

    # set terminal logging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # set file logging
    if log_filename is not None:
        if not log_filename.endswith(LOG_EXT):
            log_filename += LOG_EXT
        logging.warning(f"{log_filename} already exists, removing...")
        if os.path.exists(log_filename):
            os.remove(log_filename)
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)


def timer(method):
    """
    Decorator for timing functions. Writes the time to logger.
    """

    @wraps(method)
    def decorator(*args, **kwargs):
        time_start = time.time()
        result = method(*args, **kwargs)
        logging.info(
            f"`{method.__name__}` call took {time.time() - time_start:.2f} s"
        )
        return result

    return decorator


def make_dirs(path):
    """
    Create directory.

    :param path: path for new directory
    :type path: str
    """
    try:
        os.makedirs(path)
    except OSError as error:
        logging.warning(f"{path} could not be created: {error}")


if __name__ == "__main__":
    set_logger(log_filename=None)
    logging.info(f"Project root: {PROJECT_ROOT}")
    logging.info(f"Code folder: {CODE_ROOT}")
    logging.info(f"Data folder: {DATA_ROOT}")
