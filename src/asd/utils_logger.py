import logging
import logging.handlers
import sys

# Constants
LOGGING_LEVEL = logging.INFO
LOGGING_FILE = "/var/log/asd.log"


# Singleton pattern for logging
class LoggerSetup:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerSetup, cls).__new__(cls)
            cls._setup_logger()
        return cls._instance

    @staticmethod
    def _setup_logger():
        """
        Logger function that define the module level logging settings

        Returns:
            None
        """
        logger = logging.getLogger()
        logger.setLevel(LOGGING_LEVEL)

        detailed_formatter = logging.Formatter("%(asctime)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s")

        # Create a handler for writing to a file, with a maximum size of 20MB
        # If the filesize reaches 20MB the file is overwritten with new entries
        file_handler = logging.handlers.RotatingFileHandler(LOGGING_FILE, maxBytes=20 * 1024 * 1024, backupCount=1)
        file_handler.setFormatter(detailed_formatter)

        # Create a handler for writing to stdout
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(detailed_formatter)

        # Add both handlers to the logger object
        logger.addHandler(file_handler)
        logger.addHandler(stdout_handler)


# Initialize logger
LoggerSetup()
