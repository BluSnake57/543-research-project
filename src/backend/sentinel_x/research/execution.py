"""Automatic durable logs and terminal run status for unattended CLI experiments."""

from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

from .audit import write_json, digest
from .locking import run_lock


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def capture_log(path):
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    with Path(path).open("a") as stream:
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for fd, original in zip((1, 2), saved):
                os.dup2(original, fd)
                os.close(original)


def execute(command, config, output, operation):
    output = Path(output)
    with run_lock(output):
        status = {
            "command": command,
            "config_sha256": digest(config),
            "pid": os.getpid(),
            "started_utc": now(),
            "status": "running",
        }
        write_json(output / "status.json", status)
        print("Logging to", output / "run.log", flush=True)
        with capture_log(output / "run.log"):
            print("START", status, flush=True)
            try:
                result = operation(config, output)
                if command == "evaluate":
                    from .reporting import score_run

                    score_run(output, config["cases"])
                incomplete = result.get("completed", 0) < result.get(
                    "expected", result.get("total", result.get("latest_cases", 0))
                )
                status.update(
                    status="completed_with_failures" if incomplete else "completed", result=result
                )
                return result
            except KeyboardInterrupt:
                status.update(status="interrupted", error="Cancelled by user")
                raise
            except Exception as exc:
                status.update(status="failed", error=str(exc))
                print("FAILED", repr(exc), flush=True)
                raise
            finally:
                status["finished_utc"] = now()
                write_json(output / "status.json", status)
                print("FINISHED", status["status"], status["finished_utc"], flush=True)
