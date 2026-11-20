"""Compatibility entry point for :mod:`sim_engine.tools.step_policy`."""

from .tools import step_policy as _implementation


if __name__ == "__main__":
    _implementation.main()
else:
    import sys

    sys.modules[__name__] = _implementation
