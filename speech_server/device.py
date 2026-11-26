"""GPU architecture abstraction. This package has no custom CUDA kernels: every graph is standard ONNX plus ONNX Runtime
contrib ops (com.microsoft.MultiHeadAttention), whose kernels ORT selects per architecture (fused attention on sm80+,
memory-efficient attention, unfused fallback). Everything architecture-dependent in this package is a profile entry
here, chosen from the device's compute capability, so a new GPU generation (sm90, sm100, sm12x, ...) is a table row,
not code:

- encoder graph preference order (fused-attention graph first, plain-ONNX graphs as fallbacks: the engine tries them
  in order and keeps the first that loads and runs a probe batch);
- whether the TensorRT EP may be tried (TensorRT 10.5+ dropped sm70; sm75+ is supported);
- notes on measured per-architecture tunings (sm70 was tuned on a Volta GPU; other rows start from the defaults).
"""
from __future__ import annotations

import subprocess

# (min compute capability, profile); the last row whose minimum <= the device's applies
PROFILES = [
    ((0, 0), {"name": "cpu-or-unknown", "encoder_graphs": ["encoder5", "encoder4", "encoder3", "encoder2", "encoder"], "tensorrt": False}),
    ((7, 0), {"name": "sm70", "encoder_graphs": ["encoder5", "encoder4", "encoder3", "encoder2", "encoder"], "tensorrt": False,
              "note": "tuned on Volta: TopK over fp32 logits, HEURISTIC cuDNN search, fp16 graphs"}),
    ((7, 5), {"name": "sm75+", "encoder_graphs": ["encoder5", "encoder4", "encoder3", "encoder2", "encoder"], "tensorrt": True}),
    ((8, 0), {"name": "sm80+", "encoder_graphs": ["encoder5", "encoder4", "encoder3", "encoder2", "encoder"], "tensorrt": True,
              "note": "ORT uses fused attention kernels for MultiHeadAttention here"}),
    ((10, 0), {"name": "sm100+", "encoder_graphs": ["encoder5", "encoder4", "encoder3", "encoder2", "encoder"], "tensorrt": True,
               "note": "untested: needs an onnxruntime-gpu built with sm100 (CUDA >= 12.8); the engine probes and "
                       "falls back to plain-ONNX graphs if the contrib-op kernels are missing"}),
]


def compute_capability(device_id: int = 0) -> tuple[int, int] | None:
    """(major, minor) of a CUDA device via nvidia-smi, or None (no GPU / no driver tool)."""
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader",
                                       "-i", str(device_id)], timeout=10, text=True).strip()
        a, b = out.split(".")
        return int(a), int(b)
    except Exception:  # noqa: BLE001
        return None


def profile(cc: tuple[int, int] | None) -> dict:
    cc = cc or (0, 0)
    best = PROFILES[0][1]
    for lo, prof in PROFILES:
        if cc >= lo:
            best = prof
    return dict(best, compute_capability=cc)


# CUDA 12 / cuDNN 9 user-space libraries, in load order (dependencies first)
_CUDA_LIBS = [("cuda_runtime", "libcudart.so.12"), ("cuda_nvrtc", "libnvrtc.so.12"), ("nvjitlink", "libnvJitLink.so.12"),
              ("cublas", "libcublasLt.so.12"), ("cublas", "libcublas.so.12"), ("cufft", "libcufft.so.11"),
              ("curand", "libcurand.so.10"), ("cudnn", "libcudnn.so.9")]


def load_cuda_libs() -> list[str]:
    """Load the CUDA / cuDNN libraries from the `nvidia-*-cu12` pip wheels wherever they are installed (ORT's own
    preload_dlls only looks next to its own site-packages), RTLD_GLOBAL so ORT's CUDA provider finds them. No-op for
    libraries the system loader already provides. Returns the libraries loaded."""
    import ctypes
    import glob
    import os
    try:
        import nvidia  # namespace package of the nvidia-*-cu12 wheels
        roots = list(nvidia.__path__)
    except ImportError:
        return []
    done = []
    for sub, lib in _CUDA_LIBS:
        for root in roots:
            for p in glob.glob(os.path.join(root, sub, "lib", lib)):
                try:
                    ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                    done.append(lib)
                except OSError:
                    pass
                break
    for p in sorted(glob.glob(os.path.join(roots[0], "cudnn", "lib", "libcudnn_*.so.9"))) if roots else []:
        try:
            ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
    return done
