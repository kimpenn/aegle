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
    ],
    author="Da Kuang",
    author_email="kuangda@seas.upenn.edu",
    description="A package for CODEX image analysis",
    url="https://github.com/kuang-da/aegle",
)
