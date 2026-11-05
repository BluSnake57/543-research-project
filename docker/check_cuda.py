"""Fail unless CUDA is usable, including a real GPU tensor calculation."""

import torch


def main():
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA unavailable. Check the NVIDIA driver, Container Toolkit, and GPU reservation."
        )
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            values = torch.ones((64, 64), device=f"cuda:{index}")
            result = values @ values
            torch.cuda.synchronize()
            if not torch.all(result == 64).item():
                raise RuntimeError("CUDA matrix calculation failed")
            print(f"GPU {index}: {torch.cuda.get_device_name(index)}; matrix calculation passed")
    print(f"PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}")


if __name__ == "__main__":
    main()
