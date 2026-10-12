from pathlib import Path

import cv2
import numpy as np
import pytest
from lxml import etree
from PIL import Image

from imagedt.file import Anno_OP, FilesOp
from imagedt.image import process


def test_network_socket_success_and_failure(monkeypatch):
    from imagedt.tools import com_tools

    class Sock:
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def settimeout(self, seconds):
            assert seconds == 2

        def connect(self, address):
            pass

        def getsockname(self):
            return ("127.0.0.1", 0)

    sock = Sock()
    monkeypatch.setattr(com_tools.socket, "socket", lambda *args: sock)
    assert com_tools.getip() == "127.0.0.1" and sock.closed

    def fail(*args):
        raise OSError("offline")

    monkeypatch.setattr(com_tools.socket, "socket", fail)
    assert com_tools.getip() == "x.x.x.x"


def test_pair_file_operations(tmp_path):
    images = tmp_path / "JPEGImages"
    images.mkdir()
    annotations = tmp_path / "Annotations"
    annotations.mkdir()
    for name in ["a.b", "c"]:
        (images / (name + ".jpg")).write_bytes(b"jpg")
        (annotations / (name + ".xml")).write_text("<Annotation/>")
    assert FilesOp.check_data_pairs(tmp_path) == "have same files"
    pairs = FilesOp.get_jpg_xml_pairs(tmp_path)
    assert len(pairs) == 2 and Path(pairs[0][0]).name == "a.b.jpg"
    (annotations / "unpaired.xml").write_text("<Annotation/>")
    FilesOp.check_data_pairs(tmp_path)
    assert not (annotations / "unpaired.xml").exists()
    FilesOp.rename_file_with_uuid(tmp_path)
    assert {p.stem for p in images.iterdir()} == {p.stem for p in annotations.iterdir()}
    FilesOp.rename_single_file_uuid(images)
    assert len(list(images.iterdir())) == 2
    assert len(FilesOp.loop_dir(tmp_path)) == 4
    FilesOp.del_broken_image(tmp_path)
    assert not list(images.iterdir())


def test_pair_rename_collision_and_rollback(tmp_path, monkeypatch):
    first, second = tmp_path / "a.jpg", tmp_path / "a.xml"
    first.touch()
    second.touch()
    (tmp_path / "uid.xml").touch()
    with pytest.raises(FileExistsError):
        FilesOp.rename_data_pairs([str(first), str(second)], "uid")
    assert first.exists() and second.exists()
    (tmp_path / "uid.xml").unlink()
    import os

    real_rename = os.rename

    def fail_second(source, target):
        if str(source).endswith("a.xml"):
            raise OSError("test failure")
        real_rename(source, target)

    monkeypatch.setattr(os, "rename", fail_second)
    with pytest.raises(OSError):
        FilesOp.rename_data_pairs([str(first), str(second)], "uid")
    assert first.exists() and second.exists()


def test_annotation_optional_fields_and_boundaries(tmp_path):
    path = tmp_path / "a.xml"
    Anno_OP.write_xml(path, [[1, 2, 3, 4]], [0.9], "7")
    annotation = Anno_OP.read_xml(path)
    obj = annotation.find("object")
    for key, value in [("price", "2"), ("classify_confidence", ".9"), ("blur", "0")]:
        etree.SubElement(obj, key).text = value
    prices = etree.SubElement(obj, "prices")
    etree.SubElement(prices, "price").text = "3"
    Anno_OP.write_anno_object(annotation, path)
    objects = Anno_OP.get_label_object(path)
    assert objects[0]["prices"] == ["3"] and objects[0]["price"] == "2"
    Anno_OP.filter_xml_classes(tmp_path, resetid_list=[7])
    assert Anno_OP.get_label_object(path)[0]["name"] == "7"
    Anno_OP.reset_xml_cls(tmp_path, resetid_list=[8])
    assert Anno_OP.get_label_object(path) == []
    with pytest.raises(ValueError):
        Anno_OP.reset_xml_cls(tmp_path, resetid_list="x")
    with pytest.raises(ValueError):
        Anno_OP.filter_xml_classes(tmp_path, resetid_list="x")
    with pytest.raises(ValueError):
        Anno_OP.write_xml(path, [[0, 0, 1, 1]], [], ["x"])
    text = tmp_path / "a.txt"
    text.write_text("1,2,3,4\n")
    assert Anno_OP.read_txt_bndbox(text) == [["1", "2", "3", "4"]]
    assert Anno_OP.parse_poly_locations(
        [[3.1, 2.2], [1.2, 4.5], [2, 1.1], [4.1, 3]]
    ) == (1, 1, 5, 5)
    Anno_OP.wirte_xml_file(b"<Annotation/>", path)
    poly = tmp_path / "poly"
    poly.mkdir()
    (poly / "a.txt").write_text("1,1,2,1,2,2,1,2,hello\n")
    Anno_OP.convert_poly_to_Xml(str(poly / "a.txt"))
    assert (
        Anno_OP.get_label_object(tmp_path / "Annotations_xml" / "a.xml")[0]["name"]
        == "1"
    )
    empty = tmp_path / "empty.txt"
    empty.touch()
    assert Anno_OP.load_annoataion(empty)[0].shape == (0, 4, 2)


def test_image_helpers_and_errors(tmp_path, monkeypatch):
    invalid = tmp_path / "bad.jpg"
    invalid.write_bytes(b"broken")
    process.remove_broken_image(tmp_path)
    assert not invalid.exists()
    valid = tmp_path / "good.png"
    cv2.imwrite(str(valid), np.zeros((3, 4, 3), np.uint8))
    process.remove_broken_image(tmp_path)
    assert valid.exists()
    invalid.write_bytes(b"broken")
    with pytest.raises(ValueError):
        process.resize_dir_images_with_scale(tmp_path)
    with pytest.raises(ValueError):
        process.noise_padd(np.zeros((3, 3), np.uint8))
    assert process.get_box([2, 3, 10, 20], "top") == (2, 3)
    assert process.get_box([2, 3, 10, 20], "bot") == (2, 15)
    assert process.get_box([2, 3, 10, 20], "botup") == (2, 15)
    for name in ["imshow", "waitKey", "destroyAllWindows"]:
        monkeypatch.setattr(cv2, name, lambda *args: None)
    process.cv_show(np.zeros((3, 3, 3), np.uint8))
    process.put_cvtext(
        np.zeros((20, 20, 3), np.uint8), "A", (1, 1), font_type=cv2.FONT_HERSHEY_SIMPLEX
    )


def test_handwriting_clear_error():
    from imagedt.tools.text_generator.handwritten_text_generator import (
        HandwrittenTextGenerator,
    )

    with pytest.raises(NotImplementedError, match="legacy"):
        HandwrittenTextGenerator.generate("hello")


def test_pdf_real_render(tmp_path):
    pdfium = pytest.importorskip("pypdfium2")
    from imagedt.tools.pdf import pdf2image

    source = tmp_path / "source.pdf"
    document = pdfium.PdfDocument.new()
    for _ in range(2):
        page = document.new_page(100, 120)
        page.close()
    document.save(str(source))
    document.close()
    pdf2image(source, dpi=72)
    with Image.open(tmp_path / "source-0.jpg") as image:
        assert image.size == (100, 120)
    with Image.open(tmp_path / "source-1.jpg") as image:
        assert image.size == (100, 120)
    with pytest.raises(ValueError):
        pdf2image(source, dpi=0)
    document = pdfium.PdfDocument.new()
    page = document.new_page(30, 40)
    page.close()
    document.save(str(source))
    document.close()
    pdf2image(source, save_out=tmp_path / "single.png", dpi=144)
    with Image.open(tmp_path / "single.png") as image:
        assert image.size == (60, 80)


def test_pdf_and_excel_dependency_diagnostics(monkeypatch):
    from imagedt import _backends

    def fail(name):
        raise ModuleNotFoundError("missing submodule", name="internal_dependency")

    monkeypatch.setattr(_backends, "import_module", fail)
    with pytest.raises(ModuleNotFoundError, match="submodule"):
        _backends.require("outer", "outer")


@pytest.mark.parametrize("failure", ["missing", "collision", "second_rename"])
def test_md5_pair_failure_preserves_originals(tmp_path, monkeypatch, failure):
    # Test value: protects=image/XML integrity on a partial rename;
    # failure=missing/colliding annotation or filesystem failure; oracle=original bytes; seam=none.
    images = tmp_path / "JPEGImages"
    images.mkdir()
    annotations = tmp_path / "Annotations"
    annotations.mkdir()
    image = images / "sample.jpg"
    image.write_bytes(b"image")
    xml = annotations / "sample.xml"
    if failure != "missing":
        xml.write_bytes(b"original annotation")
    target = annotations / (process.md5(image) + ".xml")
    if failure == "collision":
        target.write_bytes(b"existing annotation")
    if failure == "second_rename":
        import os

        rename = os.rename

        def fail(source, destination):
            if str(source) == str(xml):
                raise OSError("injected XML failure")
            rename(source, destination)

        monkeypatch.setattr(os, "rename", fail)
    with pytest.raises(OSError):
        process.rename_imfiles_with_md5(tmp_path, rename_xml=True)
    assert image.read_bytes() == b"image"
    if failure != "missing":
        assert xml.read_bytes() == b"original annotation"
    if failure == "collision":
        assert target.read_bytes() == b"existing annotation"
