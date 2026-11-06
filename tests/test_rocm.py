"""ROCm admission and placement remain explicit without requiring an AMD CI host."""

from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from sentinel_x.config import model_profile
from sentinel_x.scope_worker import Engine, check_quantization, placement, preflight, tensor_device


class RocmTests(unittest.TestCase):
    @contextmanager
    def gpu(self, hip="7.2.0", available=True, free_gib=16, bf16=True):
        with (
            patch("torch.version.hip", hip),
            patch("torch.cuda.is_available", return_value=available),
            patch("torch.cuda.mem_get_info", return_value=(free_gib * 1024**3, 16 * 1024**3)),
            patch("torch.cuda.is_bf16_supported", return_value=bf16),
            patch("psutil.virtual_memory", return_value=SimpleNamespace(available=32 * 1024**3)),
        ):
            yield

    def test_rocm_uses_hip_cuda_interface_and_preserves_offloading(self):
        self.assertEqual(tensor_device("rocm"), "cuda")
        for model in ("9b", "2b"):
            profile = model_profile(model)
            self.assertEqual(placement("rocm", "none", profile), ("cuda", "cuda"))
            self.assertEqual(placement("rocm", "sae", profile), ("cuda", "cpu"))
            devices, sae = placement("rocm", "modules", profile)
            self.assertEqual(sae, "cpu")
            self.assertEqual(devices["lm_head"], "cpu")
            self.assertEqual(devices["model.language_model.layers.16"], "cuda")
            with self.gpu():
                self.assertEqual(preflight("rocm", 2048, model, "modules"), torch.bfloat16)

    def test_rocm_rejects_non_hip_torch_even_when_a_cuda_gpu_is_available(self):
        with self.gpu(hip=None):
            with self.assertRaisesRegex(ValueError, "ROCm-enabled PyTorch"):
                preflight("rocm", 2048, "2b")

    def test_cuda_selection_on_hip_build_requires_explicit_rocm_choice(self):
        with self.gpu():
            with self.assertRaisesRegex(ValueError, "Select ROCm"):
                preflight("cuda", 2048, "2b")
        with self.gpu(hip=None):
            self.assertEqual(preflight("cuda", 2048, "2b"), torch.bfloat16)

    def test_driver_and_bf16_failures_do_not_fall_back_to_cpu(self):
        for options, message in (
            ({"available": False}, "ROCm is unavailable"),
            ({"bf16": False}, "ROCm BF16 support"),
        ):
            with self.subTest(options=options), self.gpu(**options):
                with self.assertRaisesRegex(ValueError, message):
                    preflight("rocm", 2048, "2b")

    def test_low_reported_memory_does_not_reject_model_before_allocation(self):
        with (
            self.gpu(free_gib=0.1),
            patch("psutil.virtual_memory", return_value=SimpleNamespace(available=1)),
        ):
            self.assertEqual(preflight("rocm", 32768, "9b"), torch.bfloat16)

    def test_nf4_failure_retains_backend_and_original_diagnostic(self):
        with patch("bitsandbytes.nn.Linear4bit", side_effect=RuntimeError("missing HIP kernel")):
            with self.assertRaisesRegex(ValueError, "ROCM NF4 check failed.*missing HIP kernel"):
                check_quantization("rocm", torch.bfloat16)

    def test_worker_rejects_bad_rocm_kernel_before_reading_model_files(self):
        with (
            patch("sentinel_x.scope_worker.preflight", return_value=torch.bfloat16),
            patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(total_memory=16 * 1024**3),
            ),
            patch("torch.cuda.set_per_process_memory_fraction") as allocator,
            patch(
                "sentinel_x.scope_worker.check_quantization",
                side_effect=ValueError("bad NF4 kernel"),
            ) as probe,
            patch("sentinel_x.downloads.Models.verify") as verify,
        ):
            with self.assertRaisesRegex(ValueError, "bad NF4 kernel"):
                Engine({"device": "rocm", "model": "2b", "context": 2048, "offload": "sae"})
            probe.assert_called_once_with("rocm", torch.bfloat16)
            allocator.assert_not_called()
            verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
