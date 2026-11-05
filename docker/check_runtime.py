"""Check container dependencies and real worker cleanup without downloading models."""

import argparse
from importlib.metadata import version
import multiprocessing
from pathlib import Path
import tempfile

from sentinel_x.runtime import ScopeRuntime
from sentinel_x.state import Store


def idle_worker(connection, cancelled):
    """Exercise shutdown of a spawned worker that is waiting for cancellation."""
    connection.send("ready")
    cancelled.wait(30)
    connection.close()


def check_worker_cleanup():
    import psutil

    with tempfile.TemporaryDirectory() as directory:
        store = Store(Path(directory))
        runtime = ScopeRuntime(store)
        # Reject before model verification or hardware access, but use the real worker.
        store.configure("scope_device", "invalid")
        children = {child.pid for child in multiprocessing.active_children()}
        try:
            runtime.start()
        except ValueError as exc:
            assert "Choose CUDA" in str(exc), str(exc)
        else:
            raise AssertionError("Invalid device unexpectedly started Scope")
        finally:
            runtime.stop()
        assert runtime.process is None and runtime.connection is None
        assert {child.pid for child in multiprocessing.active_children()} == children

        context = multiprocessing.get_context("spawn")
        runtime.connection, child = context.Pipe()
        runtime.cancelled = context.Event()
        runtime.process = context.Process(
            target=idle_worker, args=(child, runtime.cancelled), daemon=True
        )
        runtime.process.start()
        child.close()
        pid = runtime.process.pid
        try:
            assert runtime.connection.poll(20), "Worker did not start"
            assert runtime.connection.recv() == "ready"
        finally:
            runtime.stop()
        assert not psutil.pid_exists(pid), "Scope worker survived shutdown"
        assert runtime.process is None and runtime.connection is None
    print("Scope startup failure and cancellation reap spawned workers.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()

    import accelerate  # noqa: F401
    import bitsandbytes as bnb
    import psutil  # noqa: F401
    from PySide6 import QtWidgets  # noqa: F401
    import safetensors  # noqa: F401
    import sentence_transformers  # noqa: F401
    import torch
    from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

    for name, expected in {
        "torch": "2.9.1",
        "transformers": "5.22.1",
        "bitsandbytes": "0.50.2",
        "accelerate": "1.14.0",
        "safetensors": "0.8.0",
        "psutil": "7.2.2",
    }.items():
        assert version(name).split("+")[0] == expected, (name, version(name))
    assert torch.version.cuda == ("12.8" if args.variant == "cuda" else None)
    assert Qwen3_5ForConditionalGeneration is not None
    assert BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4").load_in_4bit
    # A small real NF4 operation catches missing CPU backend libraries on either image.
    torch.manual_seed(0)
    layer = bnb.nn.Linear4bit(64, 16, quant_type="nf4", compress_statistics=True).to("cpu")
    with torch.inference_mode():
        result = layer(torch.ones(1, 64))
    assert result.shape == (1, 16) and torch.isfinite(result).all()
    check_worker_cleanup()
    print(f"{args.variant} dependencies, imports, and CPU NF4 operation passed.")


if __name__ == "__main__":
    main()
