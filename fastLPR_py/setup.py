"""
setup.py - Build Cython extension for fastLPR acceleration

This file enables automatic compilation of the Cython extension during pip install.
The extension provides OpenMP-parallelized NUFFT operations for all dimensions.

Usage:
    pip install .                    # Install with Cython acceleration (default)
    pip install . --no-build-isolation  # For development with local Cython
    python setup.py build_ext --inplace  # Build extension in-place for development
"""
import os
import sys
import numpy as np
from setuptools import setup, Extension, find_packages

# Try to import Cython - required for building the extension
try:
    from Cython.Build import cythonize
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    cythonize = None

# Detect platform and set appropriate compiler flags for OpenMP
if sys.platform == 'win32':
    # Windows with MSVC
    extra_compile_args = ['/openmp', '/O2']
    extra_link_args = []
elif sys.platform == 'darwin':
    # macOS - OpenMP requires libomp from Homebrew
    # brew install libomp
    extra_compile_args = ['-Xpreprocessor', '-fopenmp', '-O3']
    extra_link_args = ['-lomp']
else:
    # Linux/Unix with GCC
    extra_compile_args = ['-fopenmp', '-O3', '-ffast-math']
    extra_link_args = ['-fopenmp']

# Define the Cython extension
ext_modules = []
if HAS_CYTHON:
    pyx_file = os.path.join("src", "fastlpr", "_nufft_accel.pyx")
    if os.path.exists(pyx_file):
        extensions = [
            Extension(
                "fastlpr._nufft_accel",
                sources=[pyx_file],
                include_dirs=[np.get_include()],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            )
        ]
        ext_modules = cythonize(
            extensions,
            language_level=3,
            compiler_directives={
                'boundscheck': False,
                'wraparound': False,
                'cdivision': True,
            }
        )
    else:
        print(f"Warning: Cython source file not found: {pyx_file}")
else:
    print("Note: Cython not available. Installing pure Python version.")
    print("      For maximum performance, install Cython: pip install cython>=3.0")
    print("      Then reinstall: pip install --force-reinstall .")

# Setup configuration
setup(
    name="fastlpr",
    version="1.0.0",
    description="Fast Local Polynomial Regression and Kernel Density Estimation using NUFFT",
    author="Ying Wang, Min Li",
    author_email="yingwangrigel@gmail.com",
    license="GPL-3.0-or-later",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.9,<3.14",  # Numba required, not yet compatible with 3.14+
    install_requires=[
        "numpy>=1.23",
        "scipy>=1.10",
    ],
    extras_require={
        "plot": ["matplotlib>=3.7"],
        "accel": ["cython>=3.0", "numba>=0.57", "psutil>=5.9"],
        "dev": ["pytest>=7.4", "pytest-cov>=4.1", "black>=23.7", "ruff>=0.1.6", "mypy>=1.5"],
        "docs": ["sphinx>=7.0", "furo>=2023.9.10"],
    },
)
