# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import os
import cv2
import sys
import lake
import random
import numpy as np
from collections import OrderedDict
import tensorflow as tf

# import imagedt
from ..image.process import noise_padd

def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
  """Returns a TF-Feature of floats.

  Args:
    values: A scalar of list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))




class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


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
    all_files = lake.dir.loop(data_dir, ['.jpg', '.png'])
    return all_files


def get_label_dict(dir_dict):
    return OrderedDict((label, index+1) for index, label in enumerate(dir_dict))


def write_laebl_lines(root_dir, all_files, label_map):
    label_lines = []
    for cls in label_map:
        label_lines.append([cls, label_map[cls]])
    lake.file.write_csv(label_lines, os.path.join(root_dir, 'labels.txt'))

    f_lines = []
    for f_name in all_files:
        cls = os.path.basename(os.path.dirname(f_name))
        f_lines.append([f_name, label_map[cls]])
    lake.file.write_csv(f_lines, os.path.join(root_dir, 'image_files.txt'))

    return f_lines


def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(
        feature={'image/encoded': bytes_feature(image_data),
                 'image/format': bytes_feature(image_format),
                 'image/class/label': int64_feature(class_id),
                 'image/height': int64_feature(height),
                 'image/width': int64_feature(width), }))


def convert_to_tfrecord(f_lines, root_dir):
    """Converts a file to TFRecords."""
    print('Generating TFRecords' )
    with tf.Graph().as_default():
        image_reader = ImageReader()
        with tf.Session('') as sess:
            piece_count = 15000
            num_pieces = int(len(f_lines) / piece_count + 1)

            for num_piece in range(num_pieces):
                start_id, end_id = num_piece*piece_count, min(len(f_lines), (num_piece+1)*piece_count)

                output_file = os.path.join(root_dir, 'flowers_train_liby_' + str(num_piece+1).zfill(6) + '.tfrecord')
                with tf.python_io.TFRecordWriter(output_file) as record_writer:
                    for index in range(start_id, end_id):
                        try:
                            cvmat = cv2.imread(f_lines[index][0])
                            h, w, _ = cvmat.shape
                        except Exception as e:
                            print(e)
                            print("error image file {0}...".format(f_lines[index][0]))
                            continue
                        pdmat = noise_padd(cvmat, edge_size=max(h, w))
                        encoded_image = cv2.imencode('.jpg', pdmat)[1].tostring()

                        # encoded_image = tf.gfile.FastGFile(f_lines[index][0], 'rb').read()
                        # print("train image label: {0}".format(map_dict[cls_name]))
                        height, width = image_reader.read_image_dims(sess, encoded_image)
                        # tf exxmple format: NCHW
                        example = image_to_tfexample(encoded_image, b'jpg', height, width, f_lines[index][1])
                        record_writer.write(example.SerializeToString())

                        print("finished: ", index + 1, '/', len(f_lines), "; image height: {0}, width: {1}".format(height, width))

                    sys.stdout.write('\n')
                    sys.stdout.flush()


if __name__ == '__main__':
    # root_dir = '/data/dataset/liby_offline/train_renet50/train_datas'
    # all_files_c = get_image_pairs(root_dir)

    train_image_file = '/data/dataset/liby_offline/train_renet50/models/train.txt'
    with open(train_image_file, 'r') as f:
        all_files = [item.strip().split(' ')[0] for item in f.readlines()]

    dir_dict = get_files_dir_dict(all_files)
    label_map = get_label_dict(dir_dict)

    check_0 = all_files[0]
    # shuffle all the files
    random.shuffle(all_files)
    assert check_0 != all_files[0]

    svae_dir = '/data/dataset/liby_offline/train_renet50/save_tf_records'
    f_lines = write_laebl_lines(svae_dir, all_files, label_map)

    convert_to_tfrecord(f_lines, svae_dir)

