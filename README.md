# Welcome to LAMINAR

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/CKleiber/LAMINAR/ci.yml?branch=main)](https://github.com/CKleiber/LAMINAR/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/LAMINAR/badge/)](https://laminar-learn.readthedocs.io/)
[![codecov](https://codecov.io/gh/CKleiber/LAMINAR/branch/main/graph/badge.svg)](https://codecov.io/gh/CKleiber/LAMINAR)

## Installation

The Python package `LAMINAR` can be installed from PyPI:

```
python -m pip install LAMINAR
```

## Development installation

If you want to contribute to the development of `LAMINAR`, we recommend
the following editable installation from this repository:

```
git clone https://github.com/CKleiber/LAMINAR.git
cd LAMINAR
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
