"""Small network utilities."""

import socket


def getip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(2)
            sock.connect(("www.baidu.com", 0))
            return sock.getsockname()[0]
    except OSError:
        return "x.x.x.x"
