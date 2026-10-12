"""Compatibility exports, loaded on first use."""

from imagedt._lazy import resolve as _resolve

_EXPORTS = {
    "parse_rec": (".detect_eval", "parse_rec"),
    "parse_detect": (".detect_eval", "parse_detect"),
    "load_detect_lines": (".detect_eval", "load_detect_lines"),
    "get_xmls_basename": (".detect_eval", "get_xmls_basename"),
    "voc_eval": (".detect_eval", "voc_eval"),
    "load_label_Dict": (".f1_score", "load_label_Dict"),
    "write_predict_details": (".f1_score", "write_predict_details"),
    "gernerate_infos_dict": (".f1_score", "gernerate_infos_dict"),
    "calculate_f1_score": (".f1_score", "calculate_f1_score"),
    "roc": (".roc_curve", "roc"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    return _resolve(name, _EXPORTS, globals(), __name__)


def __dir__():
    return sorted(set(globals()) | set(__all__))
