""" The default sample size for all distributions """
import psutil

DEFAULT_SAMPLE_SIZE = 10

""" The max value within the int32 number space """
INT32_MAX = (2**31) - 1

""" The default number of jobs to be done in parallel"""
DEFAULT_NUM_JOBS = max(1, int(psutil.cpu_count() / 2))
