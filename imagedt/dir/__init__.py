"""Compatibility exports, loaded on first use."""

from imagedt._lazy import resolve as _resolve

_EXPORTS = {
    "LoopFile": (".dir_loop", "LoopFile"),
    "loop": (".dir_loop", "loop"),
    "get_dir_images": (".dir_loop", "get_dir_images"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    return _resolve(name, _EXPORTS, globals(), __name__)


def __dir__():
    return sorted(set(globals()) | set(__all__))
