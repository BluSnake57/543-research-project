"""Validate all Compose variants without a Docker daemon (stdlib only)."""

import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def validate(config, *, cpu, bind_address="0.0.0.0", port="36080"):
    desktop = config["services"]["desktop"]
    variant = "cpu" if cpu else "cuda"
    assert desktop["image"] == f"sentinel-x-desktop:local-{variant}"
    assert desktop["pull_policy"] == "never", "Application image must never be pulled"
    build_args = desktop["build"]["args"]
    # Compose 2.24 can emit merged build arguments as KEY=value strings.
    if isinstance(build_args, list):
        build_args = dict(item.split("=", 1) for item in build_args)
    assert build_args["TORCH_INDEX_URL"].endswith("/cpu" if cpu else "/cu128")
    devices = (
        desktop.get("deploy", {}).get("resources", {}).get("reservations", {}).get("devices", [])
    )
    assert bool(devices) == (not cpu), "Wrong GPU reservation"
    assert not config["networks"]["desktop"].get("internal", False), (
        "Published ports require an external network"
    )
    assert len(desktop["ports"]) == 1, "Only HTTP/WebSocket should be published"
    published = desktop["ports"][0]
    assert published["host_ip"] == bind_address
    assert str(published["published"]) == port
    assert published["target"] == 6080
    assert desktop["read_only"] and desktop["user"] == "1000:1000"
    mounts = {mount["target"]: mount for mount in desktop["volumes"]}
    assert mounts["/home/sentinel"]["type"] == "volume"
    assert mounts["/home/sentinel"]["source"] == "desktop-home"
    assert mounts["/run/secrets/vnc_password"]["read_only"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compose", help="Standalone Compose executable; defaults to docker compose"
    )
    args = parser.parse_args()
    command = [args.compose] if args.compose else ["docker", "compose"]
    for cpu in (False, True):
        files = ["compose.yaml"]
        if cpu:
            files.append("compose.cpu.yaml")
        invocation = command + [arg for file in files for arg in ("-f", file)]
        for bind_address, port in (("", ""), ("127.0.0.1", "6080"), ("0.0.0.0", "36080")):
            output = subprocess.check_output(
                invocation + ["config", "--format", "json"],
                cwd=ROOT,
                text=True,
                env={**os.environ, "NOVNC_BIND_ADDRESS": bind_address, "NOVNC_PORT": port},
            )
            validate(
                json.loads(output),
                cpu=cpu,
                bind_address=bind_address or "0.0.0.0",
                port=port or "36080",
            )
            print(f"Validated {' + '.join(files)} at {bind_address or '0.0.0.0'}:{port or '36080'}")


if __name__ == "__main__":
    main()
