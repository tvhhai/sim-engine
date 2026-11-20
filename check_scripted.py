"""Compatibility entry point for :mod:`sim_engine.tools.check_scripted`."""

from .tools import check_scripted as _implementation


if __name__ == "__main__":
    _implementation.main()
else:
    import sys

    sys.modules[__name__] = _implementation
