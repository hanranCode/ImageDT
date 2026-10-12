"""Shared lazy export resolver; optional backends load only when requested."""

from importlib import import_module


def resolve(name, exports, namespace, package):
    if name not in exports:
        raise AttributeError(f"module {package!r} has no attribute {name!r}")
    module_name, attribute = exports[name]
    module = import_module(module_name, package)
    value = getattr(module, attribute) if attribute else module
    namespace[name] = value
    return value
