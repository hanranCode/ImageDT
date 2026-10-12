"""Compatibility exports, loaded on first use."""

from imagedt._lazy import resolve as _resolve

_EXPORTS = {
    "getip": (".com_tools", "getip"),
    "detect_eval": (".metrics.detect_eval", None),
    "f1_score": (".metrics.f1_score", None),
    "plot_tools": (".plot_curve", "plot_tools"),
    "text_generator": (".text_generator.text_generator", None),
    "pdf": (".pdf", None),
    "metrics": (".metrics", None),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    return _resolve(name, _EXPORTS, globals(), __name__)


def __dir__():
    return sorted(set(globals()) | set(__all__))
