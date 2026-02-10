import logging
import os

def setup_logger(log_file: str = "app.log", logger_name: str | None = None) -> logging.Logger:
    """
    Sets up the logger to record logs to a specified file with UTF-8 encoding.

    Args:
        log_file (str): Path to the log file.
        logger_name (str | None): Optional logger name. Defaults to a stable name derived from log_file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the log directory exists
    log_directory = os.path.dirname(log_file)
    if log_directory and not os.path.exists(log_directory):
        os.makedirs(log_directory, exist_ok=True)

    if logger_name is None:
        # One logger per log file to avoid handler reuse across different components.
        logger_name = f"logger:{os.path.normpath(log_file)}"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Prevent adding multiple handlers if the logger already has handlers
    if not logger.handlers:
        # Create file handler which logs messages with UTF-8 encoding
        try:
            fh = logging.FileHandler(log_file, encoding='utf-8')
        except TypeError:
            # For Python versions < 3.9 where 'encoding' might not be supported
            fh = logging.FileHandler(log_file)
            fh.stream = open(log_file, 'a', encoding='utf-8')

        fh.setLevel(logging.INFO)

        # Create console handler for real-time feedback
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Define log message format
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
