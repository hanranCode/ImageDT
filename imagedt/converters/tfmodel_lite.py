"""Frozen GraphDef to TFLite conversion with the original entry point."""

from pathlib import Path

from imagedt._backends import require


def convert(
    graph_def_file,
    output_file,
    output_format="TFLITE",
    inference_type="FLOAT",
    inference_input_type="FLOAT",
    input_array="data",
    input_shape="1,224,224,3",
    output_arrays="softmax",
):
    if output_format != "TFLITE":
        raise NotImplementedError(
            "GRAPHVIZ_DOT is no longer supported by the modern converter"
        )
    if inference_type != "FLOAT" or inference_input_type != "FLOAT":
        raise NotImplementedError(
            "Legacy uint8 conversion requires calibration data; use a representative dataset with the modern TensorFlow converter"
        )
    arrays = [name.strip() for name in input_array.split(",")]
    shapes = [
        [int(dim) for dim in shape.split(",")] for shape in input_shape.split(":")
    ]
    if len(arrays) != len(shapes) or any(
        not shape or min(shape) < 1 for shape in shapes
    ):
        raise ValueError("input arrays and positive shapes must correspond one-to-one")
    if not Path(graph_def_file).is_file():
        raise FileNotFoundError(graph_def_file)
    tf = require("tensorflow", "tensorflow")
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        str(graph_def_file),
        arrays,
        [name.strip() for name in output_arrays.split(",")],
        input_shapes=dict(zip(arrays, shapes, strict=True)),
    )
    Path(output_file).write_bytes(converter.convert())
