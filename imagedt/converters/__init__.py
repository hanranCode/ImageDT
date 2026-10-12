"""Compatibility exports, loaded on first use."""

from imagedt._lazy import resolve as _resolve

_EXPORTS = {
    "caffe_coreml": (".caffe_coreml", None),
    "caffe_tfmodel": (".caffe_tfmodel", None),
    "tfrecords": (".tfrecords", None),
    "tfmodel_lite": (".tfmodel_lite", None),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    return _resolve(name, _EXPORTS, globals(), __name__)


def __dir__():
    return sorted(set(globals()) | set(__all__))
