"""Policy implementations for SimEngine."""

import importlib


def policy_class(spec=None):
    """Lớp policy từ chuỗi "module:Lớp"; không truyền thì dùng config.DEFAULT_POLICY (bản kết hợp V1+V2)."""
    from .. import config
    module_name, class_name = (spec or config.DEFAULT_POLICY).split(":")
    return getattr(importlib.import_module(module_name), class_name)


def default_policy(env):
    """Policy viết tay mặc định (config.DEFAULT_POLICY) gắn với env."""
    return policy_class()(env)
