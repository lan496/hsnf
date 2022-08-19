# Change Log

## v0.3.15
- Fix general solutions of integer linear system: `hsnf.integer_system.solve_integer_linear_system`

## v0.3.14
- Fix docs

## v0.3.9
* Add solver for Ax=b (mod q): `hsnf.integer_system.solve_modular_integer_linear_system`

## v0.3.8
* Fix to enable to import ``__version__`` in PyPI package
* Drop Python3.7 due to use ``importlib.metadata``

## v0.3.7
Initial release with

* Calculating Smith normal form and Hermite normal form
* Solving integer linear system by using HNF
* Basic lattice algorithm: union, intersection, and dual of lattices
