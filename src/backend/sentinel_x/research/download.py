"""Explicit, checksum-verified public research source downloads."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def fetch(lock, output):
    records = json.loads(Path(lock).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, record in records.items():
        if Path(name).name != name:
            raise ValueError("Source names must be filenames")
        path = output / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]:
            continue
        temporary = path.with_suffix(path.suffix + ".part")
        with urlopen(record["url"], timeout=120) as response, temporary.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Source hash mismatch: {name}; original file was preserved")
        temporary.replace(path)
    return {"verified_files": len(records)}
