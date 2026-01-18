#!/usr/bin/env python3
"""Setup script for Weather-DB v1"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="weather-db",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Weather Database System with AllSky camera integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Weather-DB_v1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "requests>=2.31.0",
        "sqlalchemy>=2.0.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "ephem>=4.1.0",
        "astropy>=5.3.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "flask>=3.0.0",
        "matplotlib>=3.8.0",
    ],
    entry_points={
        "console_scripts": [
            "weather-db=main:main",
        ],
    },
)
