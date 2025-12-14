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
    python_requires=">=3.8",
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "scipy>=1.7.0",
            "scikit-image>=0.19.0",
            "opencv-python-headless>=4.5.0",
        ],
        "analysis": [
            # Dependencies for aegle_analysis module
            "scanpy>=1.9.0",
            "anndata>=0.8.0",
            "seaborn>=0.12.0",
            "statsmodels>=0.13.0",
            "scipy>=1.7.0",
            "leidenalg>=0.9.0",  # Required for Leiden clustering in scanpy
        ],
        "gpu": [
            # CuPy for GPU-accelerated computing
            # Install the appropriate version for your CUDA:
            # - CUDA 11.x: pip install cupy-cuda11x
            # - CUDA 12.x: pip install cupy-cuda12x
            # Note: Will be installed separately based on system CUDA version
        ],
    },
    author="Da Kuang",
    author_email="kuangda@seas.upenn.edu",
    description="A package for CODEX image analysis",
    url="https://github.com/kuang-da/aegle",
)
