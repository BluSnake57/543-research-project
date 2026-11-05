"""Run inside the hardened container to verify desktop, authentication and isolation."""

import base64
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import urllib.error
import urllib.request

from healthcheck import main as healthcheck


def receive(connection, length):
    value = b""
    while len(value) < length:
        chunk = connection.recv(length - len(value))
        if not chunk:
            raise RuntimeError("VNC closed the connection prematurely")
        value += chunk
    return value


def main():
    healthcheck()
    with urllib.request.urlopen("http://127.0.0.1:6080/", timeout=3) as response:
        if (
            response.headers.get_content_type() != "text/html"
            or b"/exports/" not in response.read()
        ):
            raise RuntimeError("Desktop landing page is unavailable")
    with socket.create_connection(("127.0.0.1", 6080), timeout=3) as connection:
        connection.sendall(
            b"GET /websockify HTTP/1.1\r\nHost: localhost:6080\r\n"
            b"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
            b"Sec-WebSocket-Protocol: binary\r\n"
            b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n"
        )
        stream = connection.makefile("rb")
        if b"101" not in stream.readline():
            raise RuntimeError("noVNC WebSocket upgrade failed")
        while True:
            line = stream.readline()
            if line == b"\r\n":
                break
            if not line:
                raise RuntimeError("WebSocket upgrade headers were incomplete")
        header = stream.read(2)
        if len(header) != 2 or header[0] != 0x82:
            raise RuntimeError("Expected a binary WebSocket frame")
        payload = stream.read(header[1] & 0x7F)
        if payload != b"RFB 003.008\n":
            raise RuntimeError("WebSocket did not reach the VNC server")
    # A client must never be offered the unauthenticated RFB security type (1).
    with socket.create_connection(("127.0.0.1", 5900), timeout=3) as connection:
        greeting = receive(connection, 12)
        connection.sendall(greeting)
        count = receive(connection, 1)[0]
        methods = receive(connection, count)
        if not count or 1 in methods or 2 not in methods:
            raise RuntimeError(f"Unexpected VNC authentication methods: {list(methods)}")
        connection.sendall(b"\x02")
        receive(connection, 16)
        connection.sendall(b"\x00" * 16)
        if struct.unpack("!I", receive(connection, 4))[0] == 0:
            raise RuntimeError("VNC accepted an incorrect password")

    if os.getuid() == 0:
        raise RuntimeError("Desktop must run as a non-root user")
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
    if int(status["CapEff"], 16) != 0 or int(status["NoNewPrivs"]) != 1:
        raise RuntimeError("Capabilities or privilege escalation are enabled")
    if not os.statvfs("/").f_flag & os.ST_RDONLY:
        raise RuntimeError("Root filesystem is not mounted read-only")
    try:
        with tempfile.TemporaryFile(dir="/etc"):
            pass
    except OSError:
        pass
    else:
        raise RuntimeError("System files are writable")
    with tempfile.TemporaryFile(dir=Path.home()):
        pass
    password = Path("/run/secrets/vnc_password").read_text().strip()
    authorization = "Basic " + base64.b64encode(f"sentinel:{password}".encode()).decode()
    with tempfile.TemporaryDirectory(dir=Path.home() / "Exports") as directory:
        export = Path(directory) / "conversation.html"
        export.write_text("<script>alert('should download, never execute')</script>")
        url = f"http://127.0.0.1:6080/exports/{export.parent.name}/{export.name}"
        try:
            urllib.request.urlopen(url, timeout=3)
        except urllib.error.HTTPError as exc:
            if exc.code != 401:
                raise
        else:
            raise RuntimeError("Download was accessible without a password")
        request = urllib.request.Request(url, headers={"Authorization": authorization})
        with urllib.request.urlopen(request, timeout=3) as response:
            if response.read().decode() != export.read_text():
                raise RuntimeError("Downloaded file contents differ")
            if response.headers.get_content_type() != "application/octet-stream":
                raise RuntimeError("Exported HTML could execute in the browser")
        request = urllib.request.Request(
            url, headers={"Authorization": authorization}, data=b"overwrite", method="PUT"
        )
        try:
            urllib.request.urlopen(request, timeout=3)
        except urllib.error.HTTPError as exc:
            if exc.code != 403:
                raise
        else:
            raise RuntimeError("Download endpoint accepted an upload")
        link = export.parent / "system-file"
        link.symlink_to("/etc/passwd")
        request = urllib.request.Request(
            url.rsplit("/", 1)[0] + "/system-file", headers={"Authorization": authorization}
        )
        try:
            urllib.request.urlopen(request, timeout=3)
        except urllib.error.HTTPError as exc:
            if exc.code not in (403, 404):
                raise
        else:
            raise RuntimeError("Download endpoint followed a symlink outside Exports")
    windows = subprocess.check_output(["xwininfo", "-root", "-tree"], text=True, timeout=5)
    if "Sentinel-X" not in windows:
        raise RuntimeError("Sentinel-X has no visible window")
    print("Desktop, WebSocket, VNC auth, download protection, and container isolation passed.")


if __name__ == "__main__":
    main()
