"""Compatibility entry point for :mod:`sim_engine.studio.server`."""

from .studio import server as _implementation


if __name__ == "__main__":
    _implementation.main()
else:
    import sys

    sys.modules[__name__] = _implementation
