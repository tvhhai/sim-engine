"""Compatibility entry point for :mod:`sim_engine.tools.make_replay`."""

from .tools import make_replay as _implementation


if __name__ == "__main__":
    _implementation.main()
else:
    import sys

    sys.modules[__name__] = _implementation
