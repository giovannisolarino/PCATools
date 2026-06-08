"""Setup script for pca_tools.

Install into a virtual environment with::

    python -m venv .venv
    .venv\\Scripts\\activate        # Windows
    # source .venv/bin/activate     # Linux/macOS
    pip install -e .

Then ``import pca_tools`` / ``from pca_tools import PCACV`` from anywhere.
"""

from setuptools import setup

setup(
    name="pca_tools",
    version="0.1.0",
    description="Scikit-learn-native cross-validated component selection for PCA "
                "(rkf / ekf / cekf / EM PRESS).",
    long_description=__doc__,
    author="Giovanni Solarino",
    author_email="giovanni.solarino@unito.it",
    license="MIT",
    py_modules=["pca_tools"],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-learn>=1.6",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)
