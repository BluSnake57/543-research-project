"""Check the browser endpoint, VNC server, and X display without a login."""

import socket
import subprocess
import urllib.request


def main():
    subprocess.run(["xdpyinfo"], check=True, stdout=subprocess.DEVNULL, timeout=3)
    with urllib.request.urlopen("http://127.0.0.1:6080/vnc.html", timeout=3) as response:
        if response.status != 200:
            raise RuntimeError("noVNC is unavailable")
    with socket.create_connection(("127.0.0.1", 5900), timeout=3) as connection:
        if connection.recv(12) != b"RFB 003.008\n":
            raise RuntimeError("VNC did not return its protocol greeting")


if __name__ == "__main__":
    main()
