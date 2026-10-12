import os
from collections import OrderedDict

import imagedt as lake
from imagedt._backends import tensorflow_v1
from imagedt.tensorflow.tools.dataset_util import (
    bytes_feature as bytes_feature,
)
from imagedt.tensorflow.tools.dataset_util import (
    float_feature as float_feature,
)
from imagedt.tensorflow.tools.dataset_util import (
    int64_feature as int64_feature,
)

# import imagedt
from imagedt.tensorflow.tools.Records import ImageReader as ImageReader
from imagedt.tensorflow.tools.Records import RecordWriter

tf = tensorflow_v1()


def get_files_dir_dict(all_files):
    tem_dict = dict()
    for f_name in all_files:
        dir_name = os.path.basename(os.path.dirname(f_name))
        # if dir_name.startswith('other') or not f_name.endswith('.jpg'):
        #     continue
        tem_dict.setdefault(dir_name, []).append(f_name)
        # [tem_dict.setdefault(os.path.basename(os.path.dirname(f_name)), []).append(f_name) for f_name in datas]
    return tem_dict


def get_image_pairs(data_dir):
    all_files = lake.dir.loop(data_dir, [".jpg", ".png"])
    return all_files


def get_label_dict(dir_dict):
    return OrderedDict((label, index) for index, label in enumerate(dir_dict))


def write_laebl_lines(root_dir, all_files, label_map):
    label_lines = []
    for cls in label_map:
        label_lines.append([cls, label_map[cls]])
    lake.file.write_csv(label_lines, os.path.join(root_dir, "labels.txt"))

    f_lines = []
    for f_name in all_files:
        cls = os.path.basename(os.path.dirname(f_name))
        f_lines.append([f_name, label_map[cls]])
    lake.file.write_csv(f_lines, os.path.join(root_dir, "image_files.txt"))

    return f_lines


def image_to_tfexample(image_data, image_format, height, width, class_id):
    return RecordWriter.image_to_tfexample(
        image_data, image_format, height, width, class_id
    )


def convert_to_tfrecord(f_lines, root_dir):
    return RecordWriter.convert_to_tfrecord(
        f_lines, root_dir, dataset_type="flowers_train_liby_"
    )
