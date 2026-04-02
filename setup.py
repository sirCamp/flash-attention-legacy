"""
Flash Attention Legacy — Build & Install
Compiles CUDA kernels for Pascal (SM 6.x) and Volta (SM 7.0)
"""

import sys
from pathlib import Path

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
VERSION = "0.3.0"

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
    str(csrc_dir / "kernels" / "flash_attn_fwd_pascal.cu"),
    str(csrc_dir / "kernels" / "flash_attn_bwd.cu"),
]

include_dirs = [
    str(csrc_dir),
    str(csrc_dir / "include"),
]

# ---------------------------------------------------------------------------
# NVCC flags
# Target: SM 6.0 (Pascal P100), SM 6.1 (Pascal GTX 1080), SM 7.0 (Volta V100)
# ---------------------------------------------------------------------------
nvcc_flags = [
    "-O3",
    "--use_fast_math",
    # Generate code for all target architectures
    "-gencode", "arch=compute_60,code=sm_60",   # Pascal P100
    "-gencode", "arch=compute_61,code=sm_61",   # Pascal GTX 1080/Ti
    "-gencode", "arch=compute_70,code=sm_70",   # Volta V100
    "-gencode", "arch=compute_70,code=compute_70",  # PTX for forward compat
    # FP16 support
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-std=c++17",
    # Maximize shared memory
    "-Xptxas", "-dlcm=ca",  # cache all levels
]

# Platform-specific compiler flags
cxx_flags = ["-O3", "-std=c++17"]
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
    description="Flash Attention v2 for Pascal (SM 6.x) and Volta (SM 7.0) GPUs",
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
        "bench": ["tabulate", "matplotlib"],
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
    keywords="flash-attention transformer cuda pascal volta gpu",
)
