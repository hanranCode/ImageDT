"""Caffe graph utilities; TensorFlow loads only when its backend is requested."""

from imagedt._lazy import resolve as _resolve

_EXPORTS = {
    "GraphBuilder": (".graph", "GraphBuilder"),
    "NodeMapper": (".graph", "NodeMapper"),
    "KaffeError": (".errors", "KaffeError"),
    "print_stderr": (".errors", "print_stderr"),
    "tensorflow": (".tensorflow", None),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    return _resolve(name, _EXPORTS, globals(), __name__)


def __dir__():
    return sorted(set(globals()) | set(__all__))
