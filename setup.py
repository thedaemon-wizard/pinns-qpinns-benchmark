#!/usr/bin/env python3
"""
Setup script for PINNs vs QPINNs Benchmark
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="pinns-qpinns-benchmark",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Benchmark comparing Physics-Informed Neural Networks with Quantum Physics-Informed Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pinns-qpinns-benchmark",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/pinns-qpinns-benchmark/issues",
        "Documentation": "https://github.com/yourusername/pinns-qpinns-benchmark#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "physics-informed neural networks",
        "quantum computing",
        "quantum machine learning",
        "heat equation",
        "PDE solver",
        "variational quantum circuits",
        "GQE",
        "GPT",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    py_modules=[
        "main",
        "config",
        "data_utils",
        "pinn_model",
        "qpinn_model",
        "gpt_model",
        "energy_estimator",
        "gqe_generator",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "pennylane>=0.43.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "transformers>=4.30.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "pennylane-lightning-gpu",
        ],
    },
    entry_points={
        "console_scripts": [
            "pinns-qpinns-benchmark=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
