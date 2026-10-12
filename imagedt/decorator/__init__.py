"""Compatibility exports, loaded on first use."""

from imagedt._lazy import resolve as _resolve

_EXPORTS = {"time_cost": (".decorator_time", "time_cost")}
__all__ = list(_EXPORTS)


def __getattr__(name):
    return _resolve(name, _EXPORTS, globals(), __name__)


def __dir__():
    return sorted(set(globals()) | set(__all__))
