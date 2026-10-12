"""Optional backend loading without changing the caller's global runtime."""

from importlib import import_module


def require(module, extra):
    try:
        return import_module(module)
    except ModuleNotFoundError as error:
        if error.name == module or module.startswith(error.name + "."):
            raise ImportError(f"Install imagedt[{extra}] to use {module}") from error
        raise


def tensorflow_v1():
    return require("tensorflow", "tensorflow").compat.v1
