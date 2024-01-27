# logger.py
import logging

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup a logger."""
    handler = logging.FileHandler(log_file)    
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def setup_console_logger(name, level=logging.INFO):
    """Function to setup a console logger."""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger