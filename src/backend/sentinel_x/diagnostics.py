"""Offline startup check for source and frozen distributions; no weights required."""

import json
import multiprocessing
import tempfile
from pathlib import Path


def probe_worker(connection):
    try:
        import accelerate
        import safetensors
        import torch
        import transformers
        from transformers import Qwen3_5ForConditionalGeneration
        from .config import manifest

        connection.send(
            {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "accelerate": accelerate.__version__,
                "safetensors": safetensors.__version__,
                "architecture": Qwen3_5ForConditionalGeneration.__name__,
                "scope_revision": manifest()["scope"]["revision"],
            }
        )
    except Exception as exc:
        connection.send({"error": str(exc)})
    finally:
        connection.close()


def startup_check():
    from PySide6.QtWidgets import QApplication
    from .desktop import Window
    from .service import Service

    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory(prefix="sentinel-startup-") as folder:
        service = Service(Path(folder))
        window = Window(service, auto_setup=False)
        window.show()
        app.processEvents()
        window.close()
        service.jobs.shutdown(wait=True)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=probe_worker, args=(child,))
    process.start()
    child.close()
    try:
        if not parent.poll(60):
            raise RuntimeError("Packaged worker startup timed out")
        result = parent.recv()
        if "error" in result:
            raise RuntimeError(result["error"])
        print(json.dumps({"startup": "passed", "weights_loaded": False, **result}))
    finally:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        process.close()
        parent.close()
    return 0
