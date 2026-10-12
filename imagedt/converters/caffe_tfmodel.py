import os

import numpy as np

from imagedt._backends import tensorflow_v1

from .taffe import KaffeError, print_stderr
from .taffe.tensorflow import TensorFlowTransformer

tf = tensorflow_v1()


def fatal_error(msg):
    print_stderr(msg)
    raise KaffeError(msg)


def assert_exist_files(*files):
    for check_file in files:
        # print("check_file: {0}".format(check_file))
        if not os.path.isfile(check_file):
            print(f"{check_file} not file")
            raise FileNotFoundError(check_file)


def convert(
    prototxt_path,
    caffemodel_path,
    phase="test",
    output_model_name="freeze_graph_model.pb",
):
    """
    'def_path', 'Model definition (.prototxt) path'
    'caffemodel', 'Model data (.caffemodel) path'
    'phase', default='test','The phase to convert: test (default) or train'
    'output_model_name', saved tf model name. default= 'standalonehybrid.pb'
    """
    assert_exist_files(prototxt_path, caffemodel_path)

    transformer = TensorFlowTransformer(prototxt_path, caffemodel_path, phase=phase)
    data = transformer.transform_data()
    source = transformer.transform_source()
    directory = os.path.dirname(prototxt_path)
    data_output_path = os.path.join(directory, "output.mat")
    with open(data_output_path, "wb") as stream:
        np.save(stream, data)
    with open(os.path.join(directory, "output.py"), "w", encoding="utf-8") as stream:
        stream.write(source)
    namespace = {}
    exec(compile(source, "<converted-caffe-network>", "exec"), namespace)
    network_class = namespace[transformer.graph.name]
    inputs = transformer.graph.get_input_nodes()
    outputs = transformer.graph.get_output_nodes()
    if len(inputs) != 1 or len(outputs) != 1:
        raise KaffeError(
            "Caffe conversion currently requires exactly one input and one output"
        )
    input_node = inputs[0].name
    output_node = outputs[0].name
    shape = transformer.graph.get_node(input_node).output_shape
    graph = tf.Graph()
    with graph.as_default(), tf.Session(graph=graph) as session:
        placeholder = tf.placeholder(
            tf.float32,
            [shape.batch_size, shape.height, shape.width, shape.channels],
            name=input_node,
        )
        network = network_class({input_node: placeholder})
        session.run(tf.global_variables_initializer())
        network.load(data_output_path, session)
        output_tensor = network.layers[output_node]
        frozen = tf.graph_util.convert_variables_to_constants(
            session, graph.as_graph_def(), [output_tensor.op.name]
        )
    # Variable scopes reserve names even when no op has the bare layer name.
    # Add a stable public alias after freezing instead of accepting an auto suffix.
    if output_tensor.op.name != output_node:
        if any(node.name == output_node for node in frozen.node):
            raise KaffeError(
                f"Output alias would collide with an existing op: {output_node}"
            )
        alias = frozen.node.add(name=output_node, op="Identity")
        alias.input.append(output_tensor.name)
        alias.attr["T"].type = output_tensor.dtype.as_datatype_enum
    output = os.path.join(directory, output_model_name)
    with open(output, "wb") as stream:
        stream.write(frozen.SerializeToString())
