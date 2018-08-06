import os
from pathlib import Path

application_name = "opticverge"

application_directory = os.path.join(Path.home(), application_name)

log_directory = os.path.join(application_directory, 'log')

data_directory = os.path.join(application_directory, 'data')
