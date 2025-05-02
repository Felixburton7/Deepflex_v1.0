import os
from setuptools import setup, find_packages

# Read the content of README.md
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="ensembleflex",
    version="0.1.0",
    description="ML pipeline for protein flexibility prediction with multi-temperature analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Felix Burton",
    author_email="felixburton2002@gmail.comcom",
    url="
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "tqdm>=4.64.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "torch>=1.9.0",
        "pyyaml>=6.0",
        "click>=8.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "joblib>=1.0.0",
        "optuna>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "ensembleflex=ensembleflex.cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)