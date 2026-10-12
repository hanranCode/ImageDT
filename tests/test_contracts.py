"""Guard the original public source interfaces and all installable module paths."""

import ast
import importlib
import inspect
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
API = json.loads((ROOT / "tests" / "fixtures" / "legacy_api.json").read_text())


@pytest.mark.parametrize("path,definitions", list(API.items()), ids=list(API))
def test_legacy_function_names_and_arguments(path, definitions):
    tree = ast.parse((ROOT / path).read_text())
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    for name, contract in definitions.items():
        if name not in nodes:
            # Consolidated implementations may be re-exported at the old path.
            pytest.importorskip("tensorflow")
            module = importlib.import_module(path[:-3].replace("/", "."))
            value = getattr(module, name)
            if isinstance(contract, list):
                assert list(inspect.signature(value).parameters) == contract
            else:
                for method, arguments in contract.items():
                    assert (
                        list(inspect.signature(getattr(value, method)).parameters)
                        == arguments
                    )
            continue
        node = nodes[name]
        if isinstance(contract, list):
            assert [arg.arg for arg in node.args.args] == contract
        else:
            methods = {
                method.name: method
                for method in node.body
                if isinstance(method, ast.FunctionDef)
            }
            for method, arguments in contract.items():
                assert method in methods, f"{path}: removed {name}.{method}"
                assert [arg.arg for arg in methods[method].args.args] == arguments


@pytest.mark.parametrize(
    "module",
    [
        path.relative_to(ROOT)
        .with_suffix("")
        .as_posix()
        .replace("/", ".")
        .removesuffix(".__init__")
        for path in (ROOT / "imagedt").rglob("*.py")
        if path.name != "test_lenet.py"
    ],
)
def test_module_imports(module):
    if ".tensorflow" in module or module in (
        "imagedt.converters.caffe_tfmodel",
        "imagedt.converters.tfrecords",
    ):
        pytest.importorskip("tensorflow")
    if ".taffe" in module:
        pytest.importorskip("google.protobuf")
    if module in ("imagedt.tools.plot_curve", "imagedt.tools.metrics.roc_curve"):
        pytest.importorskip("matplotlib")
    importlib.import_module(module)
