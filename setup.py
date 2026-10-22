"""
Flash Attention Legacy — Build & Install
Compiles CUDA kernels for Pascal (SM 6.x), Volta (SM 7.0) and Turing (SM 7.5).

Target architectures:
  FLASH_ATTN_LEGACY_ARCHS, e.g. "7.5" or "6.0;6.1;7.0" ("+PTX" also embeds PTX
  for that arch). Default: 6.0, 6.1, 7.0, 7.5+PTX. PyTorch's TORCH_CUDA_ARCH_LIST
  is deliberately ignored (images often set it globally, for other GPUs).
  Architectures the installed nvcc cannot compile (e.g. CUDA 13 dropped
  SM 6.x/7.0) are skipped with a warning, so a recent toolkit still builds for
  the GPUs it supports. Fewer architectures also mean a much faster build.
"""

import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
VERSION = "0.5.0"

# ---------------------------------------------------------------------------
# Check CUDA availability
# ---------------------------------------------------------------------------
if CUDA_HOME is None:
    raise RuntimeError(
        "CUDA_HOME not found. Please install CUDA toolkit and ensure "
        "nvcc is on your PATH."
    )

# ---------------------------------------------------------------------------
# CUDA source files
# ---------------------------------------------------------------------------
this_dir = Path(__file__).resolve().parent
csrc_dir = this_dir / "csrc"

cuda_sources = [
    str(csrc_dir / "flash_attn_legacy.cpp"),
    str(csrc_dir / "kernels" / "flash_attn_fwd_volta.cu"),
    str(csrc_dir / "kernels" / "flash_attn_fwd_turing.cu"),
    str(csrc_dir / "kernels" / "flash_attn_fwd_pascal.cu"),
    str(csrc_dir / "kernels" / "flash_attn_bwd.cu"),
    str(csrc_dir / "kernels" / "flash_attn_decode.cu"),
]

include_dirs = [
    str(csrc_dir),
    str(csrc_dir / "include"),
]

# ---------------------------------------------------------------------------
# Target architectures
# ---------------------------------------------------------------------------
DEFAULT_ARCHS = ["6.0", "6.1", "7.0", "7.5+PTX"]  # P100, GTX 1080/Ti, V100, T4 (+ PTX)


def _nvcc_supported_archs():
    """Compute capabilities the installed nvcc can compile, e.g. {"7.5", "8.0"}."""
    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
    try:
        out = subprocess.run([nvcc, "--list-gpu-arch"], capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None  # old nvcc without --list-gpu-arch: trust the requested list
    return {f"{m[:-1]}.{m[-1]}" for m in re.findall(r"compute_(\d+)", out)}  # 100 -> 10.0


def _note(message):
    print(f"flash_attn_legacy: {message}", file=sys.stderr)  # pip shows stderr of failed builds


def _target_archs():
    requested = os.environ.get("FLASH_ATTN_LEGACY_ARCHS")
    archs = re.split(r"[;\s,]+", requested.strip()) if requested else DEFAULT_ARCHS
    supported = _nvcc_supported_archs()
    keep, skipped = [], []
    for arch in filter(None, archs):
        base = arch.replace("+PTX", "")
        if not re.fullmatch(r"\d+\.\d+", base):
            raise ValueError(f"FLASH_ATTN_LEGACY_ARCHS: invalid architecture {arch!r} "
                             "(expected e.g. '7.5' or '7.5+PTX')")
        if float(base) < 6.0:
            raise ValueError(f"FLASH_ATTN_LEGACY_ARCHS: {arch!r} is not supported (SM >= 6.0)")
        if float(base) > 7.5:
            _note(f"SM {base} is not a tested target (it runs the Volta/WMMA path)")
        (keep if supported is None or base in supported else skipped).append(arch)
    if skipped:
        message = (f"nvcc in {CUDA_HOME} cannot compile {skipped}; building for {keep} only "
                   "(use a CUDA 12.x toolkit for SM 6.x/7.0)")
        _note(message)
        warnings.warn(f"flash_attn_legacy: {message}")
    if not keep:
        raise RuntimeError(f"none of the requested architectures {archs} is supported by "
                           f"the nvcc in {CUDA_HOME}")
    return keep


def _gencode_flags(archs):
    flags = []
    for arch in archs:
        cc = arch.replace("+PTX", "").replace(".", "")
        flags += ["-gencode", f"arch=compute_{cc},code=sm_{cc}"]
        if arch.endswith("+PTX"):
            flags += ["-gencode", f"arch=compute_{cc},code=compute_{cc}"]
    return flags


TARGET_ARCHS = _target_archs()
_note(f"building for CUDA architectures {TARGET_ARCHS}")
# The extension knows what it was built for (clear error on other GPUs, and
# flash_attn_legacy_cuda.compiled_archs()): SASS list and the PTX arch (0: none).
_SASS = sorted({a.replace("+PTX", "").replace(".", "") for a in TARGET_ARCHS}, key=int)
# The lowest PTX arch decides which newer GPUs can JIT the kernels.
_PTX = min([int(a.replace("+PTX", "").replace(".", "")) for a in TARGET_ARCHS if a.endswith("+PTX")],
           default=0)
# C++ flags only (used by csrc/flash_attn_legacy.cpp): nvcc would split the
# comma-separated list into separate options.
ARCH_DEFINES = [f"-DFLASH_ATTN_LEGACY_SASS={','.join(_SASS)}", f"-DFLASH_ATTN_LEGACY_PTX={_PTX}"]

# ---------------------------------------------------------------------------
# NVCC flags (explicit -gencode flags: PyTorch then adds none of its own)
# ---------------------------------------------------------------------------
nvcc_flags = [
    "-O3",
    "--use_fast_math",
    *_gencode_flags(TARGET_ARCHS),
    # FP16 support
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-std=c++17",
    # Cache global loads in L1 and L2 (default on sm_70, opt-in on sm_60/61)
    "-Xptxas", "-dlcm=ca",
]

# Platform-specific compiler flags
cxx_flags = ["-O3", "-std=c++17", *ARCH_DEFINES]
if sys.platform != "win32":
    cxx_flags.append("-fdiagnostics-color=always")

# ---------------------------------------------------------------------------
# Extension module
# ---------------------------------------------------------------------------
ext_modules = [
    CUDAExtension(
        name="flash_attn_legacy_cuda",
        sources=cuda_sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    ),
]

# ---------------------------------------------------------------------------
# Long description from README
# ---------------------------------------------------------------------------
long_description = ""
readme_path = this_dir / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup(
    name="flash-attn-legacy",
    version=VERSION,
    author="Stefano Campese",
    author_email="sircampydevelop@gmail.com",
    description="Flash Attention v2 for Pascal (SM 6.x), Volta (SM 7.0) and Turing (SM 7.5) GPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sirCamp/flash-attention-legacy",
    packages=find_packages(include=["flash_attn_legacy", "flash_attn_legacy.*", "examples"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
    ],
    extras_require={
        "test": ["pytest", "numpy"],
        "bench": ["tabulate", "matplotlib", "sentence-transformers", "datasets"],
        "transformers": ["transformers>=4.34.0", "accelerate", "sentencepiece"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    keywords="flash-attention transformer cuda pascal volta turing gpu",
)
