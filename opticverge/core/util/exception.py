import signal


class TimeoutException(Exception):
    pass


def timeout(signum, frame):
    raise TimeoutException


def register_signal():
    signal.signal(signal.SIGALRM, timeout)


def signal_ttl(allowed_time_to_process):
    signal.alarm(allowed_time_to_process)


def reset_signal():
    signal.alarm(0)
