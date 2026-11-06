"""Check the selected GPU backend and NF4 without model downloads."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/backend"))
from sentinel_x.scope_worker import check_quantization, preflight, tensor_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("rocm", "cuda", "cpu"), default="rocm")
    args = parser.parse_args(argv)
    import torch

    try:
        dtype = preflight(args.device, context=2048, model="2b", offload="sae")
        check_quantization(args.device, dtype)
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "backend": args.device,
                "torch_device": tensor_device(args.device),
                "pytorch": torch.__version__,
                "hip": torch.version.hip,
                "cuda": torch.version.cuda,
                "nf4_check": "passed",
                "model_loaded": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
