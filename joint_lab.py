"""Compatibility entry point for :mod:`sim_engine.tools.joint_lab`."""

from .tools import joint_lab as _implementation


if __name__ == "__main__":
    _implementation.main()
else:
    import sys

    sys.modules[__name__] = _implementation
