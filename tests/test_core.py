import hashlib
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

import imagedt
from imagedt.dir import LoopFile, get_dir_images, loop
from imagedt.file import Anno_OP, FilesOp, file_write
from imagedt.image import process


def test_import_is_lightweight():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import imagedt, sys; assert not {'tensorflow', 'matplotlib', 'cv2', 'xlwt'} & sys.modules.keys(); assert imagedt.__version__ == '0.6.0rc1'",
        ],
        check=True,
    )
    assert imagedt.FilesOp is FilesOp
    assert imagedt.Anno_OP is Anno_OP
    assert imagedt.noise_padd is process.noise_padd
    assert "write_txt" in dir(imagedt)
    with pytest.raises(AttributeError):
        _ = imagedt.no_such_attribute


@pytest.mark.parametrize(
    "data", ["你好", ["你好", "world"], [["a", 2]], {"名字": "值"}]
)
def test_write_txt(tmp_path, data):
    path = tmp_path / "text.txt"
    file_write.write_txt(data, path)
    assert path.read_text(encoding="utf-8")


def test_csv_json_unicode(tmp_path):
    rows = [["中文", 'comma,quote"\nline'], ["1", "2"]]
    path = tmp_path / "data.csv"
    file_write.write_csv(rows, path, is_excel=True)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert file_write.read_csv(path) == rows
    file_write.write_json({"中文": [1, 2]}, tmp_path / "data.json")
    assert file_write.read_json(tmp_path / "data.json") == {"中文": [1, 2]}
    assert file_write.decode_strs("中文".encode()) == "中文"
    assert file_write.decode_strs("中文") == "中文"
    file_write.writeExcel([[["中文"]]], str(tmp_path / "test.xls"))
    assert (tmp_path / "test.xls").stat().st_size > 0


def test_loop_filters_and_symlink_cycle(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ["a.jpg", "noextension", "b.png", "skip.jpg"]:
        (tmp_path / name).touch()
    (tmp_path / "sub" / "deep.jpg").touch()
    (tmp_path / "sub" / "cycle").symlink_to(tmp_path, target_is_directory=True)
    assert len(loop(tmp_path)) == 4
    assert get_dir_images(tmp_path) == 4
    assert len(loop(tmp_path, None)) == 5
    paths = LoopFile(tmp_path, [".jpg"], ["skip"], ["deep.jpg"]).start(
        lambda path: path
    )
    assert [Path(path).name for path in paths] == ["a.jpg"]
    first, second = LoopFile(tmp_path), LoopFile(tmp_path)
    assert first.file_extend == second.file_extend == ()


@pytest.mark.parametrize(
    "shape", [(1, 1, 3), (1, 100, 3), (100, 1, 3), (11, 16, 3), (16, 11, 3)]
)
@pytest.mark.parametrize("edge", [1, 7, 32])
def test_noise_padding(shape, edge):
    result = process.noise_padd(np.zeros(shape, np.uint8), edge)
    assert result.shape == (edge, edge, 3)
    assert result.dtype == np.uint8


@pytest.mark.parametrize("shape", [(1, 100, 3), (100, 1, 3), (11, 16, 3)])
def test_rectangular_padding(shape):
    result = process.padd_pixel(np.zeros(shape, np.uint8), 20, 12)
    assert result.shape == (12, 20, 3)
    assert np.all((result == 0) | ((result >= 230) & (result < 240)))


@pytest.mark.parametrize(
    "edge,lo,hi", [(0, 0, 255), (2, 255, 255), (2, -1, 255), (2, 0, 257)]
)
def test_invalid_padding(edge, lo, hi):
    with pytest.raises(ValueError):
        process.noise_padd(np.zeros((2, 2, 3), np.uint8), edge, lo, hi)


def test_cleanup_waits_for_all_workers(tmp_path):
    for i in range(7):
        (tmp_path / f"{i}.jpg").write_bytes(b"bad")
    valid = tmp_path / "valid.png"
    cv2.imwrite(str(valid), np.zeros((3, 4, 3), np.uint8))
    process.clear_dir(tmp_path, debug=False, multi_process=3)
    assert list(tmp_path.iterdir()) == [valid]
    with pytest.raises(ValueError):
        process.clear_dir(tmp_path, multi_process=0)


def test_resize_preprocessing_hash_and_draw(tmp_path):
    image = np.ones((12, 24, 3), np.uint8) * 127
    assert process.resize_with_scale(image, 12).shape == (6, 12, 3)
    assert process.resize_with_scale(image, 30) is image
    with pytest.raises(ValueError):
        process.resize_with_scale(image, 0)
    assert process.swap_chanel_to_RGB(image).shape == image.shape
    assert process.our_processing(image).shape == (1, 12, 24, 3)
    np.testing.assert_allclose(
        process.inception_preprocesing(image), (127 / 255 - 0.5) * 2, rtol=1e-5
    )
    np.testing.assert_allclose(
        process.vgg_preprocesing(image)[0, 0], [3.32, 10.22, 23.06], atol=1e-4
    )
    path = tmp_path / "image.png"
    cv2.imwrite(str(path), image)
    assert process.md5(path) == hashlib.md5(path.read_bytes()).hexdigest()
    process.resize_dir_images_with_scale(tmp_path, 8)
    assert cv2.imread(str(path)).shape[:2] == (4, 8)
    assert process.draw_cvrect(image, [[1, 1, 4, 4]]) is image
    contour = process.points_to_contour([[1, 1], [4, 1], [4, 4], [1, 4]])
    assert process.draw_contours(image, [contour]) is image
    assert process.points_to_contours([[1, 1], [4, 1]]).shape == (1, 2, 1, 2)
    assert process.polly_points2contours([[1, 1, 4, 1, 4, 4, 1, 4]]).shape == (
        1,
        4,
        1,
        2,
    )
    assert process.get_box([0, 0, 4, 8]) == (0, 4)
    process.put_text(image, "a", (0, 2))


def test_alpha_to_color():
    image = Image.fromarray(np.array([[[10, 20, 30, 0], [10, 20, 30, 128]]], np.uint8))
    output = np.array(imagedt.alpha_to_color(image))
    assert output.tolist() == [[[255, 255, 255, 255], [10, 20, 30, 255]]]


def test_md5_rename_binary_and_collision(tmp_path):
    path = tmp_path / "first.jpg"
    path.write_bytes(b"\xff\x00image")
    digest = process.md5(path)
    process.rename_imfiles_with_md5(tmp_path)
    assert (tmp_path / f"{digest}.jpg").exists()
    (tmp_path / "second.jpg").write_bytes(b"\xff\x00image")
    with pytest.raises(FileExistsError):
        process.rename_imfiles_with_md5(tmp_path)


def test_file_operations(tmp_path):
    FilesOp.check_dir_exist(tmp_path / "nested" / "deep")
    assert FilesOp.fillter_file_type(["a.b.jpg", "a.xml"]) == ["a.b", "a"]
    assert FilesOp.fillter_dirname(["/tmp/image.jpg,1"]) == [["image.jpg", "1"]]
    assert FilesOp.rename_with_uuid("/a/b.c.jpg", "uid") == "/a/uid.jpg"
    with pytest.raises(ValueError):
        FilesOp.rename_with_uuid("noext", "uid")
    with pytest.raises(OSError):
        FilesOp.check_file_exist(tmp_path / "missing")
    with pytest.raises(ValueError):
        FilesOp.assert_in_op("x", [])
    with pytest.raises(ValueError):
        FilesOp.assert_not_in_op("x", ["x"])
    with pytest.raises(ValueError):
        FilesOp.assert_instance_op("x")
    FilesOp.assert_in_op(["x", "y"], ["x", "y"])
    FilesOp.assert_instance_op([])
    path = tmp_path / "nested" / "deep" / "image.jpg"
    path.touch()
    FilesOp.rename_class_dir(tmp_path)
    assert (path.parent.parent / "otherdeep" / "image.jpg").exists()


def test_annotations_roundtrip_and_reset(tmp_path):
    xml = tmp_path / "anno.xml"
    Anno_OP.write_xml(
        xml, [[1, 2, 3, 4], [5, 6, 7, 8]], [0.9, 0.1], ["7", "8"], thresh=0.5
    )
    assert Anno_OP.parse_lab_xml(xml) == [[1.0, 2.0, 3.0, 4.0]]
    objects = Anno_OP.get_label_object(xml)
    assert objects[0]["name"] == "7"
    assert Anno_OP.get_object_bndbox(objects[0]) == [1.0, 2.0, 3.0, 4.0]
    Anno_OP.reset_xml_cls(tmp_path, name="9", resetid_list=[7])
    assert Anno_OP.get_label_object(xml)[0]["name"] == "9"
    Anno_OP.filter_xml_classes(tmp_path)
    assert Anno_OP.get_label_object(xml)[0]["name"] == "0"
    Anno_OP.wirte_xml_file("<Annotation/>", xml)
    assert Anno_OP.parse_lab_xml(xml) == []


def test_polygon_annotation(tmp_path):
    missing, flags = Anno_OP.load_annoataion(tmp_path / "missing")
    assert missing.shape == (0, 4, 2) and flags.dtype == bool
    path = tmp_path / "polys" / "anno.txt"
    path.parent.mkdir()
    path.write_text("\ufeff1,2,3,2,3,4,1,4,###\n", encoding="utf-8")
    polygons, flags = Anno_OP.load_annoataion(path)
    assert polygons.shape == (1, 4, 2) and flags.tolist() == [True]
    Anno_OP.convert_poly_to_Xml(str(path))
    assert (
        Anno_OP.get_label_object(tmp_path / "Annotations_xml" / "anno.xml")[0]["name"]
        == "2"
    )


def test_decorator(capsys):
    @imagedt.time_cost
    def add(a, b):
        return a + b

    assert add(2, 3) == 5 and add.__name__ == "add"
    assert "time-cost" in capsys.readouterr().out
