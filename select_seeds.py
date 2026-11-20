"""Compatibility entry point for :mod:`sim_engine.tools.select_seeds`."""

from .tools import select_seeds as _implementation


if __name__ == "__main__":
    _implementation.main()
else:
    import sys

    sys.modules[__name__] = _implementation
