"""Compatibility exports, loaded on first use."""

from imagedt._lazy import resolve as _resolve

_EXPORTS = {
    "DIR_TYPR": (".file_operate", "DIR_TYPR"),
    "FilesOp": (".file_operate", "FilesOp"),
    "Anno_OP": (".parse_annotation", "Anno_OP"),
    "write_txt": (".file_write", "write_txt"),
    "write_csv": (".file_write", "write_csv"),
    "read_csv": (".file_write", "read_csv"),
    "readlines": (".file_write", "readlines"),
    "write_json": (".file_write", "write_json"),
    "read_json": (".file_write", "read_json"),
    "decode_strs": (".file_write", "decode_strs"),
    "writeExcel": (".file_write", "writeExcel"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    return _resolve(name, _EXPORTS, globals(), __name__)


def __dir__():
    return sorted(set(globals()) | set(__all__))
