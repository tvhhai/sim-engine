"""Compatibility entry point for :mod:`sim_engine.tools.trace_episode`."""

from .tools import trace_episode as _implementation


if __name__ == "__main__":
    _implementation.main()
else:
    import sys

    sys.modules[__name__] = _implementation
