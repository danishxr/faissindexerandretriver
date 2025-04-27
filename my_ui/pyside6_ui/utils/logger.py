import os
import logging
from logging.handlers import RotatingFileHandler
import sys

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(logs_dir, exist_ok=True)

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up a logger with both stream and file handlers.
    
    Args:
        name: Name of the logger
        log_file: Path to the log file (if None, a default path will be used)
        level: Logging level
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided or use default
    if log_file is None:
        log_file = os.path.join(logs_dir, f"{name}.log")
    
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger