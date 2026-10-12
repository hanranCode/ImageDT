import os
import random
import re
import sys
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np

import imagedt
from imagedt._backends import tensorflow_v1

from ...file.file_operate import FilesOp
from ...file.parse_annotation import Anno_OP
from . import dataset_util

tf = tensorflow_v1()


class ImageReader:
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        pass

    def read_image_dims(self, sess, image_data):
        return self.decode_jpeg(sess, image_data).shape[:2]

    def decode_jpeg(self, sess, image_data):
        image = cv2.imdecode(
            np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ValueError("Cannot decode image")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class Record_Writer(ImageReader):
    """docstring for RecordWriter"""

    def __init__(self):
        super().__init__()
        self.error_images = []

    def image_to_tfexample(self, image_data, image_format, height, width, class_id):
        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image/encoded": dataset_util.bytes_feature(image_data),
                    "image/format": dataset_util.bytes_feature(image_format),
                    "image/class/label": dataset_util.int64_feature(class_id),
                    "image/height": dataset_util.int64_feature(height),
                    "image/width": dataset_util.int64_feature(width),
                }
            )
        )

    def convert_to_tfrecord(self, f_lines, svae_dir, dataset_type="train"):
        """Converts a file to TFRecords."""
        os.makedirs(svae_dir, exist_ok=True)
        print("Generating TFRecords......")
        with tf.Session():
            piece_count = 50000
            num_pieces = (len(f_lines) + piece_count - 1) // piece_count
            for num_piece in range(num_pieces):
                start_id, end_id = (
                    num_piece * piece_count,
                    min(len(f_lines), (num_piece + 1) * piece_count),
                )

                output_file = os.path.join(
                    svae_dir, dataset_type + str(num_piece + 1).zfill(6) + ".tfrecord"
                )
                with tf.python_io.TFRecordWriter(output_file) as record_writer:
                    for index in range(start_id, end_id):
                        try:
                            img = cv2.imread(f_lines[index][0])
                            height, width, chanel = img.shape
                            encoded_image = cv2.imencode(".png", img)[1].tobytes()
                            # encoded_image = tf.gfile.FastGFile(f_lines[index][0], 'rb').read()
                        except Exception:
                            traceback.print_exc(file=sys.stdout)
                            print(
                                f"error image file {os.path.basename(f_lines[index][0])}..."
                            )
                            self.error_images.append(
                                os.path.basename(f_lines[index][0])
                            )
                            continue

                        # tf example format: NCHW
                        image_format = b"png"
                        example = self.image_to_tfexample(
                            encoded_image,
                            image_format,
                            height,
                            width,
                            int(f_lines[index][1]),
                        )
                        record_writer.write(example.SerializeToString())
                        print(
                            "finished: ",
                            index + 1,
                            "/",
                            len(f_lines),
                            f"; image height: {height}, width: {width}",
                        )

                    sys.stdout.write("\n")
                    sys.stdout.flush()

    def map_int(self, m_list):
        return np.array(list(map(int, list(map(float, m_list)))))

    def create_tf_example(self, sess, jpg_path, anno_infos):
        with tf.gfile.GFile(jpg_path, "rb") as fid:
            encoded_image = fid.read()
        # get infos
        chanel = 3  # now set 3
        height, width = self.read_image_dims(sess, encoded_image)
        image_type = os.path.basename(jpg_path).split(".")[-1]
        image_format = image_type.encode()

        # convert infos
        anno_infos = np.asarray(anno_infos).reshape(-1, 6)
        xmins = self.map_int(anno_infos[:, 0]) / float(
            width
        )  # List of normalized left x coordinates in bounding box (1 per box)
        ymins = self.map_int(anno_infos[:, 1]) / float(
            height
        )  # List of normalized top y coordinates in bounding box (1 per box)
        xmaxs = self.map_int(anno_infos[:, 2]) / float(
            width
        )  # List of normalized right x coordinates in bounding box # (1 per box)
        ymaxs = self.map_int(anno_infos[:, 3]) / float(
            height
        )  # List of normalized bottom y coordinates in bounding box # (1 per box)
        classes = self.map_int(
            anno_infos[:, 4]
        )  # List of integer class id of bounding box (1 per box)
        classes_text = anno_infos[
            :, 5
        ]  # List of string class name of bounding box (1 per box)

        assert (
            len(xmins)
            == len(ymins)
            == len(xmaxs)
            == len(ymaxs)
            == len(classes)
            == len(classes_text)
        )

        filename = (
            jpg_path.encode()
        )  # Filename of the image. Empty if image is not from file
        tf_example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image/height": dataset_util.int64_feature(height),
                    "image/width": dataset_util.int64_feature(width),
                    "image/channels": dataset_util.int64_feature(chanel),
                    "image/filename": dataset_util.bytes_feature(filename),
                    "image/source_id": dataset_util.bytes_feature(filename),
                    "image/encoded": dataset_util.bytes_feature(encoded_image),
                    "image/format": dataset_util.bytes_feature(image_format),
                    "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
                    "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
                    "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
                    "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
                    "image/object/class/text": dataset_util.bytes_list_feature(
                        classes_text
                    ),
                    "image/object/class/label": dataset_util.int64_list_feature(
                        classes
                    ),
                }
            )
        )
        return tf_example

    def write_cls_records(self, train_images_file, save_dir, test_ratio=0):
        if not 0 <= test_ratio <= 1:
            raise ValueError("test_ratio must be in [0, 1]")
        os.makedirs(save_dir, exist_ok=True)
        self.error_images = []
        with open(train_images_file, encoding="utf-8") as f:
            all_files = [item.strip().split("\t") for item in f.readlines()]
            # save images file in tfrecord dir
            imagedt.file.write_csv(all_files, os.path.join(save_dir, "image_files.txt"))

        # shuffle all the files
        random.shuffle(all_files)
        sample_count = len(all_files)
        # Generate before replacing the owned output family, so failed writes
        # leave the previous dataset usable. Publishing multiple files is not atomic.
        test_count = int(sample_count * test_ratio)
        with tempfile.TemporaryDirectory(dir=save_dir) as staging:
            self.convert_to_tfrecord(
                all_files[sample_count - test_count :],
                staging,
                dataset_type="validation_",
            )
            imagedt.file.write_txt(
                self.error_images, os.path.join(staging, "test_error_images.txt")
            )
            self.error_images = []
            self.convert_to_tfrecord(
                all_files[: sample_count - test_count], staging, dataset_type="train_"
            )
            imagedt.file.write_txt(
                self.error_images, os.path.join(staging, "train_error_images.txt")
            )
            generated = {path.name for path in Path(staging).iterdir()}
            for path in Path(staging).iterdir():
                os.replace(path, Path(save_dir) / path.name)
            for path in Path(save_dir).iterdir():
                if (
                    re.fullmatch(r"(?:train|validation)_[0-9]{6}\.tfrecord", path.name)
                    and path.name not in generated
                ):
                    path.unlink()

    def converte_anno_info(self, data_dir, det_cls_name="3488"):
        data_pairs = FilesOp.get_jpg_xml_pairs(data_dir)
        examples = {}
        for data_pair in data_pairs:
            jpg_path, xml_path = data_pair
            anno_infos = Anno_OP.parse_lab_xml(xml_path)
            # add default class and class_name
            anno_infos = [item + [str(det_cls_name)] * 2 for item in anno_infos]
            examples[jpg_path] = anno_infos
        print("Read image pairs: ", len(examples))
        return examples

    def write_detect_records(
        self, data_dir, save_dir, record_name="traning_detect.tfrecord"
    ):
        examples = self.converte_anno_info(data_dir)
        record_name = os.path.join(save_dir, record_name)
        os.makedirs(save_dir, exist_ok=True)
        with tf.python_io.TFRecordWriter(record_name) as writer:
            for key_path in examples:
                example = self.create_tf_example(None, key_path, examples[key_path])
                writer.write(example.SerializeToString())

        print(f"All class: 3488, class_name 3488, records save dir {save_dir}")


RecordWriter = Record_Writer()


def write_pbmodel_summery(tf_pbmodel, log_dir):
    graph = tf.Graph()
    with graph.as_default(), tf.gfile.GFile(tf_pbmodel, "rb") as stream:
        definition = tf.GraphDef()
        definition.ParseFromString(stream.read())
        tf.import_graph_def(definition, name="")
    with graph.as_default(), tf.summary.FileWriter(log_dir, graph=graph) as writer:
        writer.flush()
