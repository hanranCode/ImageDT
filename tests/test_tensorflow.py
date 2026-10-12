from types import SimpleNamespace

import cv2
import numpy as np
import pytest

# Explicit skip when this optional extra is not installed.
tf = pytest.importorskip("tensorflow")
pytestmark = [pytest.mark.tensorflow, pytest.mark.integration]

from imagedt.converters.tfmodel_lite import convert
from imagedt.tensorflow.lite.lite_wrapper import Lite_Wrapper
from imagedt.tensorflow.tools import dataset_util
from imagedt.tensorflow.tools.data_provider import (
    Data_Provider,
    dynamically_loaded_data,
    resize_image_keep_aspect,
    tf_JitterCut,
    tf_noise_padd,
)
from imagedt.tensorflow.tools.Records import (
    ImageReader,
    RecordWriter,
    write_pbmodel_summery,
)
from imagedt.tensorflow.tools.tfmodel_wrapper import TFmodel_Wrapper


def make_graph(path):
    graph = tf.Graph()
    with graph.as_default():
        data = tf.compat.v1.placeholder(tf.float32, [1, 2, 2, 3], name="data")
        scores = tf.reduce_mean(data, axis=[1, 2], name="features")
        tf.nn.softmax(scores, name="softmax")
    path.write_bytes(graph.as_graph_def().SerializeToString())
    return graph


def test_graph_wrapper_isolation_and_lite(tmp_path):
    path = tmp_path / "tiny.pb"
    make_graph(path)
    data = np.zeros((1, 2, 2, 3), np.float32)
    data[..., 2] = 2.0
    graph = tf.compat.v1.get_default_graph()
    count = len(graph.get_operations())
    with (
        TFmodel_Wrapper(path, "data", "softmax,features") as first,
        TFmodel_Wrapper(path, "data", "softmax,features") as second,
    ):
        assert first.graph is not second.graph
        assert first.predict(data)[0]["class"] == 2
        np.testing.assert_array_equal(first.extract(data), [[0, 0, 2]])
    assert len(graph.get_operations()) == count
    output = tmp_path / "tiny.tflite"
    convert(path, output, input_array="data", input_shape="1,2,2,3")
    model = Lite_Wrapper(output)
    assert (model.height, model.width) == (2, 2)
    label, score = model.predict(data)
    assert label == 2 and score > 0.7
    with pytest.raises(ValueError):
        model.predict(data.astype(np.float64))
    with pytest.raises(ValueError):
        model.predict(data[0])
    write_pbmodel_summery(path, str(tmp_path / "summary"))
    assert list((tmp_path / "summary").iterdir())


@pytest.mark.parametrize(
    "options,error",
    [
        ({"output_format": "GRAPHVIZ_DOT"}, NotImplementedError),
        ({"inference_type": "QUANTIZED_UINT8"}, NotImplementedError),
        ({"input_array": "x,y"}, ValueError),
    ],
)
def test_lite_conversion_errors(tmp_path, options, error):
    with pytest.raises(error):
        convert(tmp_path / "missing", tmp_path / "output", **options)


def test_features_xml_and_reader(tmp_path):
    assert list(dataset_util.int64_feature(2).int64_list.value) == [2]
    assert list(dataset_util.int64_list_feature([1, 2]).int64_list.value) == [1, 2]
    assert list(dataset_util.bytes_feature("你好").bytes_list.value) == [
        "你好".encode()
    ]
    assert list(dataset_util.bytes_list_feature(["a", b"b"]).bytes_list.value) == [
        b"a",
        b"b",
    ]
    assert list(dataset_util.float_feature(0.5).float_list.value) == [0.5]
    assert list(dataset_util.float_list_feature([0.5]).float_list.value) == [0.5]
    from lxml import etree

    xml = etree.fromstring(
        b"<annotation><object><name>a</name></object><object><name>b</name></object></annotation>"
    )
    assert (
        len(dataset_util.recursive_parse_xml_to_dict(xml)["annotation"]["object"]) == 2
    )
    p = tmp_path / "examples.txt"
    p.write_text("a 1\nb 2\n")
    assert dataset_util.read_examples_list(str(p)) == ["a", "b"]
    reader = ImageReader()
    encoded = cv2.imencode(".jpg", np.zeros((3, 4, 3), np.uint8))[1].tobytes()
    assert reader.read_image_dims(None, encoded) == (3, 4)
    with pytest.raises(ValueError):
        reader.decode_jpeg(None, b"bad")


def test_record_writer_no_train_validation_overlap(tmp_path):
    image = np.zeros((6, 9, 3), np.uint8)
    rows = []
    for i in range(4):
        path = tmp_path / f"{i}.jpg"
        cv2.imwrite(str(path), image)
        rows.append(f"{path}\t{i}")
    index = tmp_path / "train.txt"
    index.write_text("\n".join(rows))
    directory = tmp_path / "records"
    RecordWriter.write_cls_records(index, directory, test_ratio=0.5)

    def read_labels(prefix):
        result = []
        for path in directory.glob(prefix + "*.tfrecord"):
            for raw in tf.compat.v1.io.tf_record_iterator(str(path)):
                example = tf.train.Example.FromString(raw)
                feature = example.features.feature
                assert feature["image/format"].bytes_list.value[0] == b"png"
                result.append(feature["image/class/label"].int64_list.value[0])
        return result

    training, validation = read_labels("train_"), read_labels("validation_")
    assert len(training) == len(validation) == 2
    assert set(training).isdisjoint(validation)
    assert set(training + validation) == set(range(4))
    unrelated = directory / "train_notes.tfrecord"
    unrelated.write_bytes(b"keep")
    for ratio in [0, 0.5, 1, 0.5]:
        (directory / "train_999999.tfrecord").write_bytes(b"obsolete")
        RecordWriter.write_cls_records(index, directory, ratio)
        training, validation = read_labels("train_0"), read_labels("validation_0")
        assert len(validation) == int(4 * ratio)
        assert len(training) == 4 - int(4 * ratio)
        assert set(training).isdisjoint(validation)
        assert set(training + validation) == set(range(4))
        assert not (directory / "train_999999.tfrecord").exists()
        assert unrelated.read_bytes() == b"keep"
    with pytest.raises(ValueError):
        RecordWriter.write_cls_records(index, directory, 2)
    example = RecordWriter.create_tf_example(
        None, str(tmp_path / "0.jpg"), [[1, 2, 3, 4, "1", "文字"]]
    )
    assert (
        example.features.feature["image/object/class/text"].bytes_list.value[0]
        == "文字".encode()
    )
    empty = RecordWriter.create_tf_example(None, str(tmp_path / "0.jpg"), [])
    assert len(empty.features.feature["image/object/bbox/xmin"].float_list.value) == 0


@pytest.mark.parametrize("shape", [(3, 8, 3), (8, 3, 3), (1, 20, 3)])
def test_tensor_image_transforms(shape):
    data = tf.constant(np.zeros(shape, np.uint8))
    result = tf_noise_padd(data, 12)
    assert tuple(result.shape) == (12, 12, 3)
    assert max(resize_image_keep_aspect(data, 12).shape[:2]) == 12
    cropped = tf_JitterCut(data, 0.2)
    assert min(cropped.shape[:2]) >= 1
    np.testing.assert_array_equal(tf_JitterCut(data, 0).numpy(), data.numpy())


def test_dynamic_data_and_provider(tmp_path):
    path = tmp_path / "image.jpg"
    cv2.imwrite(str(path), np.zeros((3, 8, 3), np.uint8))
    images, labels = dynamically_loaded_data([str(path)], [2], height=6, width=10)
    assert images.shape == (1, 6, 10, 3) and labels == [2]
    with pytest.raises(ValueError):
        dynamically_loaded_data([str(path)], [])
    RecordWriter.convert_to_tfrecord([[str(path), 2]], tmp_path)
    graph = tf.Graph()
    with graph.as_default():
        provider = Data_Provider(tmp_path, 4)
        images, labels = provider.read_from_tfrecord(
            batch_size=1, image_shape=(6, 10, 3)
        )
        assert images.shape.as_list() == [1, 6, 10, 3]
        with tf.compat.v1.Session(graph=graph) as session:
            session.run(
                [
                    tf.compat.v1.global_variables_initializer(),
                    tf.compat.v1.local_variables_initializer(),
                ]
            )
            coordinator = tf.compat.v1.train.Coordinator()
            threads = tf.compat.v1.train.start_queue_runners(
                sess=session, coord=coordinator
            )
            try:
                batch, label = session.run([images, labels])
                assert batch.shape == (1, 6, 10, 3) and np.argmax(label[0]) == 2
            finally:
                coordinator.request_stop()
                coordinator.join(threads)


def test_losses_and_lenet():
    from imagedt.tensorflow.network.lenet import lenet
    from imagedt.tensorflow.optim.loss import arcface_loss, smooth_l1_loss

    loss = smooth_l1_loss(tf.constant([0.0, 0.0]), tf.constant([0.2, 1.0]))
    np.testing.assert_allclose(loss.numpy(), [0.02, 0.375], atol=1e-6)
    logits, endpoints = lenet(tf.zeros((2, 28, 28, 1)), 3, is_training=False)
    assert logits.shape == (2, 3)
    np.testing.assert_allclose(
        tf.reduce_sum(endpoints["Predictions"], axis=1).numpy(), [1, 1], atol=1e-6
    )
    graph = tf.Graph()
    with graph.as_default(), tf.compat.v1.Session(graph=graph) as session:
        output = arcface_loss(
            tf.constant([[0.0, 0.0], [1.0, 0.0]]), tf.constant([[0], [1]]), 3
        )
        session.run(tf.compat.v1.global_variables_initializer())
        assert np.isfinite(session.run(output)).all()


def test_read_dataset(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("a\nb\n")
    config = SimpleNamespace(
        num_readers=2,
        shuffle=False,
        num_epochs=1,
        read_block_length=1,
        num_parallel_map_calls=1,
        prefetch_size=1,
    )
    dataset = dataset_util.read_dataset(
        tf.data.TextLineDataset, lambda line: line, str(path), config
    )
    assert list(dataset.as_numpy_iterator()) == [b"a", b"b"]
    with tf.Graph().as_default():
        dataset = tf.data.Dataset.from_tensor_slices([1, 2])
        iterator = dataset_util.make_initializable_iterator(dataset)
        with tf.compat.v1.Session() as session:
            session.run(iterator.initializer)
            assert session.run(iterator.get_next()) == 1


def test_detection_records_and_legacy_converter(tmp_path, monkeypatch):
    from imagedt.converters import tfrecords as legacy
    from imagedt.file import Anno_OP
    from imagedt.tensorflow.tools.tfrecords import visual_records

    root = tmp_path / "dataset"
    root.mkdir()
    (root / "JPEGImages").mkdir()
    (root / "Annotations").mkdir()
    path = root / "JPEGImages" / "a.jpg"
    cv2.imwrite(str(path), np.zeros((10, 12, 3), np.uint8))
    Anno_OP.write_xml(root / "Annotations" / "a.xml", [[1, 2, 8, 9]], [1.0], ["7"])
    out = tmp_path / "records"
    assert RecordWriter.converte_anno_info(root, "7")[str(path)][0][-1] == "7"
    RecordWriter.write_detect_records(root, out)
    files = list(out.glob("*.tfrecord"))
    example = tf.train.Example.FromString(
        next(tf.compat.v1.io.tf_record_iterator(str(files[0])))
    )
    assert list(example.features.feature["image/height"].int64_list.value) == [10]
    labelmap = legacy.get_label_dict(legacy.get_files_dir_dict([str(path)]))
    assert labelmap == {"JPEGImages": 0}
    assert legacy.get_image_pairs(root) == [str(path)]
    rows = legacy.write_laebl_lines(out, [str(path)], labelmap)
    legacy.convert_to_tfrecord(rows, out)
    assert (
        legacy.image_to_tfexample(b"x", b"jpg", 2, 3, 1)
        .features.feature["image/height"]
        .int64_list.value[0]
        == 2
    )
    from PIL import Image

    shown = []
    monkeypatch.setattr(Image.Image, "show", lambda self: shown.append(self.size))
    visual_records(files[0])
    assert shown == [(12, 10)]


def test_converted_network_layers():
    from imagedt.converters.taffe.tensorflow.network import Network

    class Tiny(Network):
        def setup(self):
            self.feed("data").conv(3, 3, 4, 1, 1, name="conv", group=2).relu(
                name="relu"
            ).max_pool(2, 2, 2, 2, name="pool")
            self.avg_pool(2, 2, 1, 1, name="avg").lrn(2, 1e-4, 0.75, name="lrn")
            self.batch_normalization(name="bn", relu=True).dropout(1.0, name="drop")
            self.feed("drop", "drop").concat(3, name="concat")
            self.feed("drop", "drop").add(name="add")
            self.feed("drop", "drop").multiply(name="mul")
            self.feed("drop", "drop").max(name="max")
            self.feed("max").fc(3, name="fc").softmax(name="softmax")

    graph = tf.Graph()
    with graph.as_default(), tf.compat.v1.Session(graph=graph) as session:
        net = Tiny({"data": tf.constant(np.ones((1, 8, 8, 4), np.float32))})
        session.run(tf.compat.v1.global_variables_initializer())
        output = session.run(net.get_output())
        assert output.shape == (1, 3)
        assert np.isfinite(output).all()
        np.testing.assert_allclose(output.sum(axis=1), [1.0], atol=1e-5)
        assert net.get_unique_name("conv").startswith("conv_")
        with pytest.raises(KeyError):
            net.feed("missing")


def test_lenet_features_errors_and_quantized_wrapper():
    from imagedt.tensorflow.network.lenet import lenet

    features, _ = lenet(tf.zeros((1, 28, 28, 1)), num_classes=0, is_training=False)
    assert features.shape == (1, 1024)
    with pytest.raises(ValueError):
        lenet(tf.zeros((1, 28, 28, 1)), drop_keep_prob=0)
    model = object.__new__(Lite_Wrapper)
    model.input_details = [{"dtype": np.uint8, "shape": [1, 2, 2, 3]}]
    with pytest.raises(ValueError):
        model._check_model_type()
    model.input_details = [{"dtype": np.float32, "shape": [2, 3]}]
    with pytest.raises(ValueError):
        model._check_model_type()


def test_zero_based_labels_and_all_declared_tool_exports():
    # Test value: protects=last-class one-hot targets and stable export resolution;
    # failure=one-based label overflow/stale registry; oracle=identity matrix and resolved names; seam=none.
    from imagedt.converters.tfrecords import get_label_dict
    from imagedt.tensorflow import tools

    mapping = get_label_dict({"a": [], "b": []})
    assert mapping == {"a": 0, "b": 1}
    np.testing.assert_array_equal(
        tf.one_hot(list(mapping.values()), 2).numpy(), np.eye(2)
    )
    for name in tools.__all__:
        assert getattr(tools, name) is not None


def test_failed_wrapper_closes_session(tmp_path, monkeypatch):
    # Test value: protects=owned Session release during constructor failure;
    # failure=tensor validation raises after Session opens; oracle=actual closed Session; seam=none.
    path = tmp_path / "tiny.pb"
    make_graph(path)
    closed = []
    original = TFmodel_Wrapper.close

    def close(wrapper):
        original(wrapper)
        closed.append(wrapper.sess._closed)

    monkeypatch.setattr(TFmodel_Wrapper, "close", close)
    with pytest.raises(ValueError):
        TFmodel_Wrapper(path, "missing", "softmax")
    assert closed == [True]


def test_corrupt_record_images_have_split_specific_manifests(tmp_path, monkeypatch):
    # Test value: protects=corrupt input containment and split diagnostics;
    # failure=validation errors leak into training report; oracle=manifest names + parsed records; seam=none.
    from imagedt.tensorflow.tools import Records

    monkeypatch.setattr(Records.random, "shuffle", lambda items: None)
    files = []
    for name in ["bad-train", "good-train", "bad-validation", "good-validation"]:
        path = tmp_path / (name + ".jpg")
        if name.startswith("bad"):
            path.write_bytes(b"corrupt")
        else:
            cv2.imwrite(str(path), np.zeros((3, 4, 3), np.uint8))
        files.append(str(path) + "\t0")
    index = tmp_path / "index.txt"
    index.write_text("\n".join(files))
    output = tmp_path / "records"
    RecordWriter.write_cls_records(index, output, 0.5)
    assert (
        output / "test_error_images.txt"
    ).read_text().strip() == "bad-validation.jpg"
    assert (output / "train_error_images.txt").read_text().strip() == "bad-train.jpg"
    for prefix in ["train_", "validation_"]:
        path = next(output.glob(prefix + "*.tfrecord"))
        assert len(list(tf.compat.v1.io.tf_record_iterator(str(path)))) == 1


def test_arcface_logits_match_margin_and_fallback_formula():
    # Test value: protects=target-only ArcFace margin and large-angle fallback;
    # failure=wrong mask/sign/threshold; oracle=independent NumPy formula; seam=none.
    import math

    from imagedt.tensorflow.optim.loss import arcface_loss

    embedding = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], np.float32)
    labels = np.array([1, 1, 0], np.int32)
    expected = embedding.copy()
    for row, label in enumerate(labels):
        cosine = embedding[row, label]
        expected[row, label] = (
            cosine * math.cos(0.5)
            - math.sqrt(max(0.0, 1 - cosine * cosine)) * math.sin(0.5)
            if cosine > math.cos(math.pi - 0.5)
            else cosine - math.sin(0.5) * 0.5
        )
    graph = tf.Graph()
    with graph.as_default(), tf.compat.v1.Session(graph=graph) as session:
        logits = arcface_loss(
            tf.constant(embedding),
            tf.constant(labels),
            2,
            w_init=tf.constant_initializer(np.eye(2)),
            s=2.0,
            m=0.5,
        )
        session.run(tf.compat.v1.global_variables_initializer())
        np.testing.assert_allclose(
            session.run(logits), expected * 2, rtol=1e-6, atol=1e-6
        )
