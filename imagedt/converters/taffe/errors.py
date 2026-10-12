import sys


class KaffeError(Exception):
    pass


def print_stderr(msg):
    sys.stderr.write(f"{msg}\n")
