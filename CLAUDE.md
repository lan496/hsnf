# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

hsnf is a Python library for computing Hermite normal form (HNF) and Smith normal form (SNF) of integer matrices, with transformation matrices. It also provides utilities for integer lattice operations and solving integer/modular linear systems.

## Development Commands

```bash
# Install in development mode
pip install -e .[dev,docs]

# Run all tests with coverage
python -m pytest -v --cov=hsnf --cov-config=setup.cfg --cov-report=xml tests/

# Run a single test file
python -m pytest tests/test_normal_form.py -v

# Run a single test function
python -m pytest tests/test_normal_form.py::test_smith_normal_form -v

# Linting/formatting (via pre-commit)
pre-commit run --all-files

# Individual tools
black --line-length 99 hsnf/ tests/
flake8 hsnf/ tests/
mypy hsnf/
isort --profile black hsnf/ tests/
```

## Architecture

The library has four modules under `hsnf/`:

- **Z_module.py** -- Core algorithms. `ZmoduleHomomorphism` class implements SNF and row-style HNF via elementary row/column operations on Z-module homomorphisms. Public API functions (`smith_normal_form`, `row_style_hermite_normal_form`, `column_style_hermite_normal_form`) are thin wrappers that construct a `ZmoduleHomomorphism` with standard basis and call the corresponding method.
- **utils.py** -- Low-level helpers: pivot selection (`get_nonzero_min_abs*`), extended GCD, Sieve of Eratosthenes for prime factorization, Chinese Remainder Theorem, and the `NDArrayInt` type alias.
- **integer_system.py** -- Solvers built on top of SNF/HNF: `solve_integer_linear_system` (Ax=b over Z), `solve_frobenius_congruent` (Ax=b mod R/Z), and `solve_modular_integer_linear_system` (Ax=b mod q). The modular solver uses prime-power decomposition + CRT.
- **lattice.py** -- Lattice operations: equivalence checking, union, intersection, and dual lattice computation. Uses the identity dual(L1 intersect L2) = union(dual(L1), dual(L2)).

Public API is re-exported from `hsnf/__init__.py`: `smith_normal_form`, `row_style_hermite_normal_form`, `column_style_hermite_normal_form`.

## Key Conventions

- Line length: 99 (black + flake8)
- Import sorting: isort with black profile
- Type annotations use `NDArrayInt` (alias for `npt.NDArray[np.int_]`) from `hsnf/utils.py`
- Version is inferred at build time via `setuptools_scm` (git tags)
- Python 3.8+ compatibility: uses `from __future__ import annotations` and `typing_extensions.TypeAlias`
- Default branch is `develop`; CI runs on push to `master`/`develop` and PRs to `master`
