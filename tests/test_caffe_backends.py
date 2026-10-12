from types import SimpleNamespace

import numpy as np
import pytest
from google.protobuf import text_format

from imagedt.converters.taffe.caffe import caffe_pb2
from imagedt.converters.taffe.errors import KaffeError
from imagedt.converters.taffe.graph import Graph, GraphBuilder, Node
from imagedt.converters.taffe.layers import (
    KernelParameters,
    LayerAdapter,
    NodeDispatch,
    NodeKind,
)
from imagedt.converters.taffe.shapes import (
    TensorShape,
    get_filter_output_shape,
    shape_not_implemented,
    shape_scalar,
)
from imagedt.converters.taffe.transformers import (
    DataInjector,
    DataReshaper,
    NodeRenamer,
    ParameterNamer,
    ReLUFuser,
)


def test_protobuf_roundtrip_and_shape():
    net = caffe_pb2.NetParameter(name="Tiny")
    text_format.Merge('input: "data" input_shape { dim: 1 dim: 3 dim: 4 dim: 4 }', net)
    assert caffe_pb2.NetParameter.FromString(net.SerializeToString()) == net
    assert get_filter_output_shape(
        10, 10, KernelParameters(3, 3, 2, 2, 0, 0), np.floor
    ) == (4, 4)
    assert shape_scalar(None) == TensorShape(1, 1, 1, 1)
    with pytest.raises(NotImplementedError):
        shape_not_implemented(None)
    assert NodeDispatch.get_handler_name("InnerProduct") == "inner_product"
    assert LayerAdapter.get_kernel_value(0, [3], 0) == 3
    assert LayerAdapter.get_kernel_value(0, [], 0, 1) == 1
    with pytest.raises(ValueError):
        LayerAdapter.get_kernel_value(0, [], 0)


def test_graph_sort_cycle_and_rename():
    first, second = Node("a", NodeKind.Data), Node("b", NodeKind.ReLU)
    second.add_parent(first)
    graph = Graph([second, first], name="Tiny")
    assert graph.topologically_sorted() == [first, second]
    assert graph.get_input_nodes() == [first] and graph.get_output_nodes() == [second]
    NodeRenamer(lambda node: "new_" + node.name)(graph)
    assert graph.get_node("new_a") is first
    with pytest.raises(KaffeError):
        graph.get_node("missing")
    first.add_parent(second)
    with pytest.raises(KaffeError):
        graph.topologically_sorted()


def test_large_graph_without_recursion():
    nodes = [Node(str(i), NodeKind.ReLU) for i in range(2000)]
    for parent, child in zip(nodes, nodes[1:], strict=False):
        child.add_parent(parent)
    assert Graph(nodes).topologically_sorted() == nodes


def test_relu_fuse_and_data_reshape():
    layer = caffe_pb2.LayerParameter(name="conv", type="Convolution")
    layer.convolution_param.bias_term = True
    parent = Node("conv", NodeKind.Convolution, layer)
    relu = Node("relu", NodeKind.ReLU)
    relu.add_parent(parent)
    child = Node("next", NodeKind.Softmax)
    child.add_parent(relu)
    parent.data = [np.ones((2, 3, 1, 1)), np.zeros(2)]
    graph = ReLUFuser()(Graph([parent, relu, child]))
    assert graph.nodes == [parent, child] and child.parents == [parent]
    assert parent.metadata["relu"]
    DataReshaper({NodeKind.Convolution: (2, 3, 1, 0)})(graph)
    assert parent.data[0].shape == (1, 1, 3, 2)
    ParameterNamer()(graph)
    assert set(parent.data) == {"weights", "biases"}


def make_model(tmp_path):
    definition = tmp_path / "tiny.prototxt"
    definition.write_text("""name: "Tiny"
input: "data"
input_shape { dim: 1 dim: 3 dim: 2 dim: 2 }
layer { name: "score" type: "InnerProduct" bottom: "data" top: "score" inner_product_param { num_output: 2 } }
layer { name: "prob" type: "Softmax" bottom: "score" top: "prob" }
""")
    weights = caffe_pb2.NetParameter(name="Tiny")
    layer = weights.layer.add(name="score", type="InnerProduct")
    blob = layer.blobs.add()
    blob.shape.dim.extend([2, 12])
    blob.data.extend([0.0] * 24)
    bias = layer.blobs.add()
    bias.shape.dim.extend([2])
    bias.data.extend([0.0, 2.0])
    model = tmp_path / "tiny.caffemodel"
    model.write_bytes(weights.SerializeToString())
    return definition, model


def test_graph_builder_and_weights(tmp_path):
    definition, model = make_model(tmp_path)
    graph = GraphBuilder(definition).build()
    assert graph.get_node("score").output_shape == TensorShape(1, 2, 1, 1)
    injector = DataInjector(definition, model)
    injector(graph)
    assert graph.get_node("score").data[0].shape == (2, 12)
    assert "score" in str(graph)


@pytest.mark.tensorflow
@pytest.mark.integration
def test_caffe_to_frozen_graph(tmp_path):
    pytest.importorskip("tensorflow")
    from imagedt.converters.caffe_tfmodel import convert
    from imagedt.tensorflow.tools.tfmodel_wrapper import TFmodel_Wrapper

    definition, model = make_model(tmp_path)
    convert(str(definition), str(model))
    with TFmodel_Wrapper(tmp_path / "freeze_graph_model.pb", "data", "prob") as wrapper:
        output = wrapper.predict(np.zeros((1, 2, 2, 3), np.float32))
        assert output[0]["class"] == 1 and output[0]["confidence"] > 0.8


def test_coreml_removed_backend(monkeypatch):
    from imagedt.converters import caffe_coreml

    with pytest.raises(NotImplementedError, match="Caffe"):
        caffe_coreml.convert("model", "definition", "labels")


def test_optional_dependency_diagnostic(monkeypatch):
    from imagedt import _backends

    def missing(name):
        raise ModuleNotFoundError("missing", name=name)

    monkeypatch.setattr(_backends, "import_module", missing)
    with pytest.raises(ImportError, match=r"imagedt\[tensorflow\]"):
        _backends.tensorflow_v1()


def test_caffe_shape_layer_variants(tmp_path):
    definition = tmp_path / "conv.prototxt"
    definition.write_text("""name: "ConvTiny"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 4 dim: 8 dim: 8 } } }
layer { name: "conv" type: "Convolution" bottom: "data" top: "conv" convolution_param { num_output: 4 kernel_size: 3 pad: 1 group: 2 } }
layer { name: "pool" type: "Pooling" bottom: "conv" top: "pool" pooling_param { pool: MAX kernel_size: 2 stride: 2 } }
layer { name: "pool2" type: "Pooling" bottom: "conv" top: "pool2" pooling_param { pool: MAX kernel_size: 2 stride: 2 } }
layer { name: "cat" type: "Concat" bottom: "pool" bottom: "pool2" top: "cat" concat_param { axis: 1 } }
""")
    graph = GraphBuilder(definition).build()
    assert graph.get_node("data").output_shape == TensorShape(1, 4, 8, 8)
    assert graph.get_node("conv").output_shape == TensorShape(1, 4, 8, 8)
    assert graph.get_node("pool").output_shape == TensorShape(1, 4, 4, 4)
    assert graph.get_node("cat").output_shape.channels == 8
    # Filter excludes the specified phase, independent of include rules.
    builder = GraphBuilder(definition)
    layer = caffe_pb2.LayerParameter(name="excluded", type="ReLU")
    layer.exclude.add(phase=caffe_pb2.TEST)
    assert builder.filter_layers([layer]) == []


def test_transformer_source_quotes_and_padding():
    pytest.importorskip("tensorflow")
    from imagedt.converters.taffe.tensorflow.transformer import (
        TensorFlowEmitter,
        TensorFlowNode,
        get_padding_type,
    )

    node = TensorFlowNode("feed", "name'withquote", axis=3)
    node.node = SimpleNamespace(name="safe")
    assert "\\'" in node.emit() or '"name\'withquote"' in node.emit()
    assert (
        get_padding_type(
            KernelParameters(3, 3, 1, 1, 0, 0),
            TensorShape(1, 3, 8, 8),
            TensorShape(1, 4, 6, 6),
        )
        == "VALID"
    )
    assert (
        get_padding_type(
            KernelParameters(3, 3, 1, 1, 1, 1),
            TensorShape(1, 3, 8, 8),
            TensorShape(1, 4, 8, 8),
        )
        == "SAME"
    )
    with pytest.raises(KaffeError):
        TensorFlowEmitter().emit("invalid-name", [])


def test_batchnorm_stats_and_kernel_validation():
    from imagedt.converters.taffe.transformers import (
        BatchNormPreprocessor,
        BatchNormScaleBiasFuser,
    )

    layer = caffe_pb2.LayerParameter(name="bn", type="BatchNorm")
    node = Node("bn", NodeKind.BatchNorm, layer)
    node.data = [np.ones(3), np.ones(3) * 2, np.array(2.0)]
    graph = BatchNormPreprocessor()(Graph([node]))
    np.testing.assert_allclose(node.data[0], 0.5)
    np.testing.assert_allclose(node.data[1], 1.0)
    scale_layer = caffe_pb2.LayerParameter(name="scale", type="Scale")
    scale_layer.scale_param.bias_term = True
    scale = Node("scale", NodeKind.Scale, scale_layer)
    scale.data = [np.ones(3), np.zeros(3)]
    scale.add_parent(node)
    graph = BatchNormScaleBiasFuser()(Graph([node, scale]))
    assert graph.nodes == [node] and node.scale_bias_node is scale
    with pytest.raises(KaffeError):
        Graph([node]).transformed([lambda _: None])
    with pytest.raises(KaffeError):
        node.get_only_parent()
    assert NodeKind.map_raw_kind("invalid") is None


@pytest.mark.tensorflow
@pytest.mark.integration
def test_terminal_fc_and_serialized_batchnorm(tmp_path):
    # Test value: protects=Caffe terminal-layer output and serialized BN scaling;
    # failure=wrong op name/4D stats; oracle=analytic normalized scores; seam=none.
    pytest.importorskip("tensorflow")
    from imagedt.converters.caffe_tfmodel import convert
    from imagedt.tensorflow.tools.tfmodel_wrapper import TFmodel_Wrapper

    definition, model = make_model(tmp_path)
    definition.write_text(definition.read_text().split('layer { name: "prob"')[0])
    weights = caffe_pb2.NetParameter.FromString(model.read_bytes())
    # Nonzero FC weights also expose flatten/channel-order errors.
    weights.layer[0].blobs[0].data[:] = np.arange(24, dtype=np.float32).tolist()
    model.write_bytes(weights.SerializeToString())
    data = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3)
    expected = data.transpose(0, 3, 1, 2).reshape(1, 12) @ np.arange(
        24, dtype=np.float32
    ).reshape(2, 12).T + [0, 2]
    convert(str(definition), str(model), output_model_name="fc.pb")
    with TFmodel_Wrapper(tmp_path / "fc.pb", "data", "score") as wrapper:
        actual = wrapper.sess.run(
            wrapper.tensor_dict[wrapper.output_node[0]], {wrapper.image_tensor: data}
        )
        np.testing.assert_allclose(actual, expected)
    definition.write_text(
        definition.read_text()
        + 'layer { name: "bn" type: "BatchNorm" bottom: "score" top: "bn" }\n'
    )
    layer = weights.layer.add(name="bn", type="BatchNorm")
    for values in ([2.0, 4.0], [8.0, 8.0], [2.0]):
        blob = layer.blobs.add()
        blob.shape.dim.extend([len(values)])
        blob.data.extend(values)
    model.write_bytes(weights.SerializeToString())
    convert(str(definition), str(model), output_model_name="bn.pb")
    with TFmodel_Wrapper(tmp_path / "bn.pb", "data", "bn") as wrapper:
        actual = wrapper.sess.run(
            wrapper.tensor_dict[wrapper.output_node[0]], {wrapper.image_tensor: data}
        )
        np.testing.assert_allclose(
            actual, (expected - [1, 2]) / np.sqrt(4.0 + 1e-5), rtol=1e-5
        )

    definition.write_text(
        definition.read_text()
        + 'layer { name: "scale" type: "Scale" bottom: "bn" top: "scale" '
        + "scale_param { bias_term: true } }\n"
    )
    layer = weights.layer.add(name="scale", type="Scale")
    for values in ([2.0, 3.0], [-1.0, 5.0]):
        blob = layer.blobs.add()
        blob.shape.dim.extend([len(values)])
        blob.data.extend(values)
    model.write_bytes(weights.SerializeToString())
    convert(str(definition), str(model), output_model_name="scale.pb")
    with TFmodel_Wrapper(tmp_path / "scale.pb", "data", "bn") as wrapper:
        actual = wrapper.sess.run(
            wrapper.tensor_dict[wrapper.output_node[0]], {wrapper.image_tensor: data}
        )
        np.testing.assert_allclose(
            actual,
            (expected - [1, 2]) / np.sqrt(4.0 + 1e-5) * [2, 3] + [-1, 5],
            rtol=1e-5,
        )


def test_caffe_padding_rejects_non_equivalent_same():
    # Test value: protects=conversion fidelity for even inputs and stride two;
    # failure=equal output shape hides asymmetric TF padding; oracle=explicit rejection; seam=none.
    pytest.importorskip("tensorflow")
    from imagedt.converters.taffe.tensorflow.transformer import get_padding_type

    assert (
        get_padding_type(
            KernelParameters(3, 3, 2, 2, 1, 1),
            TensorShape(1, 1, 4, 4),
            TensorShape(1, 1, 2, 2),
        )
        is None
    )
    assert (
        get_padding_type(
            KernelParameters(3, 3, 2, 2, 1, 1),
            TensorShape(1, 1, 5, 5),
            TensorShape(1, 1, 3, 3),
        )
        == "SAME"
    )
