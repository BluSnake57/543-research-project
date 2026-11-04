"""Create the local VNC password without replacing an existing secret."""

import os
from pathlib import Path
import secrets
import string

ROOT = Path(__file__).resolve().parents[1]


def create_secret(directory):
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    destination = directory / "vnc_password"
    # The private parent protects the host copy. A world-readable file is needed
    # for Compose's read-only bind mount to a non-root container with UID 1000.
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        os.fchmod(stream.fileno(), 0o644)
        stream.write(
            "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(8))
        )
        stream.write("\n")
    return destination


if __name__ == "__main__":
    try:
        path = create_secret(ROOT / ".secrets")
        print(f"Created {path}. Read this file when noVNC asks for the password.")
    except FileExistsError:
        if not (ROOT / ".secrets/vnc_password").is_file():
            raise SystemExit("Expected .secrets/vnc_password to be a file; fix it before setup.")
        print("VNC password already exists; kept the existing secret.")
