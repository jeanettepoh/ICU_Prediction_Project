import os
import logging
from datetime import datetime

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Define log file name with timestamp
log_filename = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_file = os.path.join(logs_dir, log_filename)

# Configure the logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging_format = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")

# Create file handler
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging_format)
logger.addHandler(file_handler)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging_format)
logger.addHandler(console_handler)

# Check if above code works
if __name__ == "__main__":
    logger.info('This is a log message')