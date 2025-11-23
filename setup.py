# setup.py

from setuptools import setup, find_packages

setup(
    name="aegle",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "tifffile",
        "matplotlib",
        "Pillow",
        "pyyaml",
        "zstandard",
        "tqdm",
    ],
    extras_require={
        "gpu": [
            # CuPy for GPU-accelerated computing
            # Install the appropriate version for your CUDA:
            # - CUDA 11.x: pip install cupy-cuda11x
            # - CUDA 12.x: pip install cupy-cuda12x
            # Note: Will be installed separately based on system CUDA version
            # Uncomment the appropriate line below:
            # "cupy-cuda11x>=12.0.0",  # For CUDA 11.x
            # "cupy-cuda12x>=12.0.0",  # For CUDA 12.x
        ],
    },
    author="Da Kuang",
    author_email="kuangda@seas.upenn.edu",
    description="A package for CODEX image analysis",
    url="https://github.com/kuang-da/aegle",
)
