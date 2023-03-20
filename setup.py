"""
A hacky setup script for Python <=3.8
"""
from setuptools import setup
import sys
import subprocess

__version__ = "0.0.1"

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

setup(
    name="mosaic-flax",
    description="mosaic-flax",
    license="MIT",
    version=__version__,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)