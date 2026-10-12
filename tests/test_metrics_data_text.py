import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from imagedt.file import Anno_OP, read_csv
from imagedt.tensorflow.tools.DataInterface import DataSet, DataSets
from imagedt.tools.metrics import detect_eval
from imagedt.tools.metrics.f1_score import (
    _f1_score,
    _total_f1_score,
    calculate_f1_score,
    gernerate_infos_dict,
    load_label_Dict,
)
from imagedt.tools.text_generator.background_generator import BackgroundGenerator
from imagedt.tools.text_generator.computer_text_generator import ComputerTextGenerator
from imagedt.tools.text_generator.data_generator import FakeTextDataGenerator
from imagedt.tools.text_generator.distorsion_generator import DistorsionGenerator


@pytest.mark.parametrize(
    "boxes,classes,expected",
    [
        ([[1, 1, 10, 10]], ["cat"], 1.0),
        ([[1, 1, 10, 10]] * 2, ["cat"] * 2, 2 / 3),
        ([], [], 0.0),
        ([[20, 20, 30, 30]], ["cat"], 0.0),
        ([[1, 1, 10, 10]], ["dog"], 0.0),
    ],
)
def test_voc_matching(tmp_path, boxes, classes, expected):
    gt, det = tmp_path / "gt", tmp_path / "det"
    gt.mkdir()
    det.mkdir()
    Anno_OP.write_xml(gt / "a.xml", [[1, 1, 10, 10]], [1.0], ["cat"])
    Anno_OP.write_xml(det / "a.xml", boxes, [0.9] * len(boxes), classes)
    assert detect_eval.voc_eval(det, gt) == pytest.approx(expected)


def test_voc_empty_unknown_low_conf_and_difficult(tmp_path):
    gt, det = tmp_path / "gt", tmp_path / "det"
    gt.mkdir()
    det.mkdir()
    assert detect_eval.voc_eval(det, gt) == 0
    Anno_OP.write_xml(gt / "a.xml", [], [], [])
    Anno_OP.write_xml(det / "a.xml", [[1, 1, 4, 4]], [0.9], ["cat"])
    assert detect_eval.voc_eval(det, gt) == 0
    Anno_OP.write_xml(gt / "b.xml", [[1, 1, 4, 4]], [1.0], ["cat"])
    Anno_OP.write_xml(det / "b.xml", [[1, 1, 4, 4]], [0.1], ["cat"])
    assert detect_eval.voc_eval(det, gt) == 0
    assert detect_eval.voc_eval(det, gt, set_gt_labels=["unused"]) == 0
    with pytest.raises(ValueError):
        detect_eval.voc_eval(det, gt, det_file_type="bad")
    with pytest.raises(ValueError):
        detect_eval.voc_eval(det, gt, conf=2)
    annotation = Anno_OP.read_xml(gt / "b.xml")
    from lxml import etree

    etree.SubElement(annotation.find("object"), "difficult").text = "1"
    Anno_OP.write_anno_object(annotation, gt / "b.xml")
    assert detect_eval.parse_rec(gt / "b.xml")["cat"][0]["difficult"] == 1


def test_voc_txt(tmp_path):
    gt, det = tmp_path / "gt", tmp_path / "det"
    gt.mkdir()
    det.mkdir()
    Anno_OP.write_xml(gt / "a.xml", [[1, 1, 10, 10]], [1.0], ["cat"])
    (det / "detect_cat.txt").write_text("a  0.9 1 1 10 10\n")
    assert detect_eval.voc_eval(det, gt, det_file_type="txt") == 1


def test_f1_rows_and_alignment(tmp_path):
    labels = ["101\t1\t苹果", "102\t2\t梨"]
    mapping = load_label_Dict(labels)
    info = gernerate_infos_dict(["a,1,.9", "b,1,.8"], ["a,1", "b,2"], mapping)
    rows = _f1_score(info, mapping)
    assert rows[0][-1] == "F1-score"
    assert rows[1][-1] == pytest.approx(0.6667)
    assert info["2"]["object_id"] == "102"
    assert _total_f1_score(info) == (0.5, 0.5)
    assert _total_f1_score({}) == (0.0, 0.0)
    with pytest.raises(ValueError):
        gernerate_infos_dict(["a,1,.9"], [], mapping)
    predict, truth, label = (
        tmp_path / name for name in ["pre.txt", "gt.txt", "labels.txt"]
    )
    predict.write_text("a,1,.9\nb,2,.9\n")
    truth.write_text("b,2\na,1\n")
    label.write_text("\n".join(labels), encoding="utf-8")
    calculate_f1_score(predict, truth, label)
    assert [row[-1] for row in read_csv(tmp_path / "image_classify_infos.csv")[1:]] == [
        "1",
        "1",
    ]
    truth.write_text("other,2\na,1\n")
    with pytest.raises(ValueError):
        calculate_f1_score(predict, truth, label)


def test_roc_vectorized(tmp_path):
    from imagedt.tools.metrics.roc_curve import roc

    path = tmp_path / "result.csv"
    path.write_text(
        "Image,Name,Label,Predict,Confidence,Match\na,x,1,1,0.9,1\nb,y,2,1,0.4,0\n"
    )
    roc(path)
    rows = read_csv(tmp_path / "thres-fpr-tpr.csv")
    assert len(rows) == 1001
    assert list(map(float, rows[500])) == [0.0, 1.0, 0.5]
    assert (tmp_path / "roc.png").exists()
    path.write_text("")
    with pytest.raises(ValueError):
        roc(path)


@pytest.mark.parametrize("is_list", [False, True])
def test_dataset_batches(is_list):
    data = DataSet()
    x, y = np.arange(5), np.arange(5) * 2
    if is_list:
        x, y = list(x), list(y)
    data.set_x(x)
    data.set_y(y)
    assert data.batch_count(2) == 2
    assert data.batch_count(2, False) == 3
    batches = [data.next_batch(2, False) for _ in range(3)]
    assert [len(x) for x, y in batches] == [2, 2, 1]
    assert sorted(int(v) for x, y in batches for v in x) == list(range(5))
    for x, y in batches:
        np.testing.assert_array_equal(np.asarray(x) * 2, y)
    assert len(data.next_batch(2)[0]) == 2
    data.reset_step()
    assert len(data.next_batch(2)[0]) == 2
    with pytest.raises(ValueError):
        data.next_batch(6)
    with pytest.raises(ValueError):
        data.batch_count(0)
    empty = DataSet()
    assert empty.count == 0
    with pytest.raises(ValueError):
        empty.next_batch(1)
    ds = DataSets()
    assert ds.train is not ds.test and ds.validate is not ds.train
    with pytest.raises(ValueError):
        _ = ds.x_shape
    with pytest.raises(ValueError):
        _ = ds.y_shape
    ds.set_x_shape((2, 3))
    ds.set_y_shape((1,))
    assert ds.x_shape == (2, 3) and ds.y_shape == (1,)


@pytest.fixture
def font():
    # Matplotlib ships a redistributable test font on all CI platforms.
    import matplotlib

    return str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")


@pytest.mark.parametrize("orientation", [0, 1])
def test_text_render(font, orientation):
    image = ComputerTextGenerator.generate(
        "Hello 123", font, "#000000,#222222", 24, orientation, 1.0
    )
    assert image.mode == "RGBA" and min(image.size) > 0
    assert np.array(image)[..., 3].max() > 0
    with pytest.raises(ValueError):
        ComputerTextGenerator.generate("", font, "#000000", 24, orientation, 1)
    with pytest.raises(ValueError):
        ComputerTextGenerator.generate("hi", font, "#000000", 24, 2, 1)


@pytest.mark.parametrize("background", [0, 1, 2])
def test_fake_text(font, background, tmp_path):
    image, text = FakeTextDataGenerator.generate(
        0,
        "1234",
        font,
        tmp_path,
        40,
        "png",
        0,
        False,
        0,
        False,
        background,
        0,
        0,
        False,
        1,
        100,
        1,
        "#000000",
        0,
        1,
        save=True,
    )
    assert image.size == (100, 40) and text == "1234"
    assert (tmp_path / "0_1234.png").exists()


@pytest.mark.parametrize(
    "vertical,horizontal", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("kind", ["sin", "cos", "random"])
def test_distortions(kind, vertical, horizontal):
    image = Image.new("RGBA", (30, 20), "black")
    distorted = getattr(DistorsionGenerator, kind)(image, vertical, horizontal)
    assert distorted.width >= image.width and distorted.height >= image.height
    assert distorted.mode == "RGBA"


def test_backgrounds_and_generator(font, tmp_path, monkeypatch):
    assert BackgroundGenerator.quasicrystal(1, 1).size == (1, 1)
    assert BackgroundGenerator.plain_white(3, 4).size == (4, 3)
    assert BackgroundGenerator.gaussian_noise(3, 4).mode == "RGB"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pictures").mkdir()
    Image.new("RGB", (2, 3)).save(tmp_path / "pictures" / "tiny.png")
    assert BackgroundGenerator.picture(20, 30).size == (30, 20)
    from imagedt.tools.text_generator.text_generator import Text_Generator

    fonts = tmp_path / "fonts"
    fonts.mkdir()
    shutil.copy(font, fonts / "font.ttf")
    with Text_Generator(fonts, langu="en") as generator:
        assert len(generator.gen_price_realtime_datas(2)) == 2
        assert len(generator.gen_price_realtime_datas(1)) == 1
        generator.set_params(["hi"])
        assert len(generator.generate_texts()) == 1
        with pytest.raises(ValueError):
            generator.gen_price_realtime_datas(-1)


def test_plot_empty_and_counts(tmp_path):
    from imagedt.tools.plot_curve import plot_tools

    plot_tools.set_xy_values({})
    assert plot_tools.x == [] and plot_tools.y == []
    data = tmp_path / "data"
    data.mkdir()
    (data / "a").mkdir()
    (data / "a" / "test.jpg").touch()
    assert plot_tools.load_dataset_infos(data) == {"a": 1}
    plot_tools.plot_datas(data)
    assert (tmp_path / "datasets.png").exists()


@pytest.mark.parametrize(
    "orientation,distortion,alignment,name_format",
    [(0, 1, 0, 0), (0, 2, 2, 2), (1, 3, 1, 1), (0, 0, 1, 99)],
)
def test_generator_option_branches(
    font, tmp_path, orientation, distortion, alignment, name_format
):
    image, text = FakeTextDataGenerator.generate(
        0,
        "abc",
        font,
        tmp_path,
        40,
        "png",
        2,
        True,
        1,
        True,
        1,
        distortion,
        2,
        False,
        name_format,
        100,
        alignment,
        "#000000",
        orientation,
        1,
        save=True,
    )
    assert min(image.size) > 0 and text == "abc"
    with pytest.raises(ValueError):
        FakeTextDataGenerator.generate(
            0,
            "abc",
            font,
            tmp_path,
            5,
            "png",
            0,
            False,
            0,
            False,
            1,
            0,
            0,
            False,
            1,
            100,
            1,
            "#000000",
            0,
            1,
        )
    with pytest.raises(NotImplementedError):
        FakeTextDataGenerator.generate(
            0,
            "abc",
            font,
            tmp_path,
            40,
            "png",
            0,
            False,
            0,
            False,
            1,
            0,
            0,
            True,
            1,
            100,
            1,
            "#000000",
            0,
            1,
        )


def test_voc_ignores_difficult_and_uses_strict_iou(tmp_path):
    # Test value: protects=declared VOC difficult/IoU semantics;
    # failure=ignored GT counts or boundary changes; oracle=analytic F1; seam=none.
    gt, det = tmp_path / "gt", tmp_path / "det"
    gt.mkdir()
    det.mkdir()
    Anno_OP.write_xml(
        gt / "a.xml", [[0, 0, 1, 1], [10, 10, 11, 11]], [1, 1], ["cat", "cat"]
    )
    root = Anno_OP.read_xml(gt / "a.xml")
    from lxml import etree

    etree.SubElement(root.findall("object")[1], "difficult").text = "1"
    Anno_OP.write_anno_object(root, gt / "a.xml")
    Anno_OP.write_xml(
        det / "a.xml", [[0, 0, 1, 1], [10, 10, 11, 11]], [1, 1], ["cat", "cat"]
    )
    assert detect_eval.voc_eval(det, gt) == 1.0
    # Inclusive areas are 4 and 2; IoU is exactly one half, which must not match.
    Anno_OP.write_xml(det / "a.xml", [[0, 0, 0, 1]], [1], ["cat"])
    assert detect_eval.voc_eval(det, gt, ovthresh=0.5) == 0.0
