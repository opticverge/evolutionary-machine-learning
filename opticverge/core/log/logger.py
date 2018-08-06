import logging
import os
from datetime import datetime

from opticverge.core.config import log_directory, data_directory

_date_format = "%Y-%m-%d"
_log_date = datetime.now()
_log_format = "%(asctime)s|%(levelname)s|%(message)s"

DATA = 25
logging.addLevelName(DATA, "DATA")
logging.basicConfig(filemode='a', )


def setup_logger(name: str, directory: str, file_name: str, level=logging.INFO) -> logging.Logger:
    """ Setup a logging instance for the application

    Args:
        name(str): The name of the logging instance
        directory(str): The directory where the logs are logged
        file_name(str): The name of the file
        level(int): The default logging level

    Returns:
        logging.Logger

    """

    # create the directory if it does not exist and update the permissions
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.chmod(directory, 0o755)

    handler = logging.FileHandler(os.path.join(directory, file_name))
    handler.setFormatter(logging.Formatter(_log_format))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


# generate the logger for the application, typically handles exceptions etc
application_logger = setup_logger("opticverge_application", log_directory,
                                  "{}-app.log".format(_log_date.strftime(_date_format)))

# a logger for strictly handling data logs
data_logger = setup_logger("opticverge_data", data_directory, "{}-data.log".format(_log_date.strftime(_date_format)))
