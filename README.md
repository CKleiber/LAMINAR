# Welcome to NFLAM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/CKleiber/NFLAM/ci.yml?branch=main)](https://github.com/CKleiber/NFLAM/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/NFLAM/badge/)](https://NFLAM.readthedocs.io/)
[![codecov](https://codecov.io/gh/CKleiber/NFLAM/branch/main/graph/badge.svg)](https://codecov.io/gh/CKleiber/NFLAM)

## Installation

The Python package `NFLAM` can be installed from PyPI:

```
python -m pip install NFLAM
```

## Development installation

If you want to contribute to the development of `NFLAM`, we recommend
the following editable installation from this repository:

```
git clone https://github.com/CKleiber/NFLAM.git
cd NFLAM
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
