"""Validate all Compose variants without a Docker daemon (stdlib only)."""

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def validate(config, *, cpu, online):
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
    assert config["networks"]["desktop"].get("internal", False) == (not online)
    assert desktop["ports"][0]["host_ip"] == "127.0.0.1"
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
        for online in (False, True):
            files = ["compose.yaml"]
            if cpu:
                files.append("compose.cpu.yaml")
            if online:
                files.append("compose.online.yaml")
            invocation = command + [arg for file in files for arg in ("-f", file)]
            output = subprocess.check_output(
                invocation + ["config", "--format", "json"], cwd=ROOT, text=True
            )
            validate(json.loads(output), cpu=cpu, online=online)
            print(f"Validated {' + '.join(files)}")


if __name__ == "__main__":
    main()
