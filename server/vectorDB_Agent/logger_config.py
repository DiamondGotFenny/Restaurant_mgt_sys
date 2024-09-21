import logging
import os

def setup_logger(log_file: str = "vector_store_agent.log") -> logging.Logger:
    """
    Sets up the logger to record logs to a specified file.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the log directory exists
    log_directory = os.path.dirname(log_file)
    if log_directory and not os.path.exists(log_directory):
        os.makedirs(log_directory, exist_ok=True)

    # Configure the logger
    logger = logging.getLogger("VectorStoreAgentLogger")
    logger.setLevel(logging.INFO)

    # Create file handler which logs messages
    fh = logging.FileHandler(log_file)
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
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger