# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import random
import numpy as np
import traceback
import tensorflow as tf

from . import dataset_util
from ...file.file_operate import FilesOp
from ...file.parse_annotation import Anno_OP
from .data_provider import tf_noise_padd
import imagedt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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


class Record_Writer(ImageReader):
  """docstring for RecordWriter"""
  def __init__(self):
    super(Record_Writer, self).__init__()
    self.error_images = []

  def image_to_tfexample(self, image_data, image_format, height, width, class_id):
      return tf.train.Example(features=tf.train.Features(
          feature={'image/encoded': dataset_util.bytes_feature(image_data),
                   'image/format': dataset_util.bytes_feature(image_format),
                   'image/class/label': dataset_util.int64_feature(class_id),
                   'image/height': dataset_util.int64_feature(height),
                   'image/width': dataset_util.int64_feature(width), }))

  def convert_to_tfrecord(self, f_lines, svae_dir, dataset_type='train'):
    """Converts a file to TFRecords."""
    print('Generating TFRecords......' )
    with tf.Session() as sess:
      piece_count = 50000
      num_pieces = int(len(f_lines) / piece_count + 1)
      for num_piece in range(num_pieces):
        start_id, end_id = num_piece*piece_count, min(len(f_lines), (num_piece+1)*piece_count)

        output_file = os.path.join(svae_dir, dataset_type + str(num_piece+1).zfill(6) + '.tfrecord')
        with tf.python_io.TFRecordWriter(output_file) as record_writer:
          for index in range(start_id, end_id):
            try:
              img = cv2.imread(f_lines[index][0])
              height, width, chanel = img.shape
              encoded_image = cv2.imencode('.png', img)[1].tostring()
              # encoded_image = tf.gfile.FastGFile(f_lines[index][0], 'rb').read()
            except Exception as e:
              traceback.print_exc(file=sys.stdout)
              print("error image file {0}...".format(os.path.basename(f_lines[index][0])))
              self.error_images.append(os.path.basename(f_lines[index][0]))
              continue

            # tf example format: NCHW
            image_format = os.path.basename(f_lines[index][0]).split('.')[-1].encode()
            example = self.image_to_tfexample(encoded_image, image_format, height, width, int(f_lines[index][1]))
            record_writer.write(example.SerializeToString())
            print("finished: ", index + 1, '/', len(f_lines), "; image height: {0}, width: {1}".format(height, width))

          sys.stdout.write('\n')
          sys.stdout.flush()

  def map_int(self, m_list):
    return np.array(map(int, map(float, m_list)))

  def create_tf_example(self, sess, jpg_path, anno_infos):
    with tf.gfile.GFile(jpg_path, 'rb') as fid:
        encoded_image = fid.read()
    # get infos
    chanel = 3 # now set 3
    height, width = self.read_image_dims(sess, encoded_image)
    image_type = os.path.basename(jpg_path).split('.')[-1]
    image_format = image_type.encode()

    # convert infos
    anno_infos = np.array(anno_infos)
    xmins = self.map_int(anno_infos[:, 0]) / float(width)  # List of normalized left x coordinates in bounding box (1 per box)
    ymins = self.map_int(anno_infos[:, 1]) / float(height)  # List of normalized top y coordinates in bounding box (1 per box)
    xmaxs = self.map_int(anno_infos[:, 2]) / float(width)  # List of normalized right x coordinates in bounding box # (1 per box)
    ymaxs = self.map_int(anno_infos[:, 3]) / float(height)  # List of normalized bottom y coordinates in bounding box # (1 per box)
    classes = self.map_int(anno_infos[:, 4])  # List of integer class id of bounding box (1 per box)
    classes_text = anno_infos[:, 5]  # List of string class name of bounding box (1 per box)

    assert len(xmins) == len(ymins) == len(xmaxs) == len(ymaxs) == len(classes) == len(classes_text)

    filename = jpg_path.encode()  # Filename of the image. Empty if image is not from file
    tf_example = tf.train.Example(features=tf.train.Features(
        feature={'image/height': dataset_util.int64_feature(height),
                 'image/width': dataset_util.int64_feature(width),
                 'image/channels': dataset_util.int64_feature(chanel),
                'image/filename': dataset_util.bytes_feature(filename),
                'image/source_id': dataset_util.bytes_feature(filename),
                'image/encoded': dataset_util.bytes_feature(encoded_image),
                'image/format': dataset_util.bytes_feature(image_format),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes), }))
    return tf_example

  def write_cls_records(self, train_images_file, save_dir, test_ratio=0):
    with open(train_images_file, 'r') as f:
      all_files = [item.strip().split('\t') for item in f.readlines()]
      # save images file in tfrecord dir
      imagedt.file.write_csv(all_files, os.path.join(save_dir,'image_files.txt'))

    # shuffle all the files
    random.shuffle(all_files)
    sample_count = len(all_files)
    # read file, write to tfrecords
    test_count = int(sample_count * test_ratio)
    if test_count:
      test_samples = all_files[-test_count:]
      self.convert_to_tfrecord(test_samples, save_dir, dataset_type='validation_')
      # save error images
      imagedt.file.write_txt(self.error_images, os.path.join(save_dir,'test_error_images.txt'))

    train_samples = all_files[:sample_count-test_count]
    self.convert_to_tfrecord(all_files, save_dir, dataset_type='train_')
    # save error images
    imagedt.file.write_txt(self.error_images, os.path.join(save_dir,'train_error_images.txt'))

  def converte_anno_info(self, data_dir, det_cls_name='3488'):
    data_pairs = FilesOp.get_jpg_xml_pairs(data_dir)
    examples = {}
    for index, data_pair in enumerate(data_pairs):
        jpg_path, xml_path = data_pair
        anno_infos = Anno_OP.parse_lab_xml(xml_path)
        # add default class and class_name
        anno_infos = [item+['3488']*2 for item in anno_infos]
        examples[jpg_path] = anno_infos
    print("Read image pairs: ", len(examples))
    return examples

  def write_detect_records(self, data_dir, save_dir, record_name='traning_detect.tfrecord'):
    examples = self.converte_anno_info(data_dir)
    record_name = os.path.join(save_dir, record_name)
    writer = tf.python_io.TFRecordWriter(record_name)
    with tf.Session() as sess:
      for index, key_path in enumerate(examples):
          tf_example = self.create_tf_example(sess, key_path, examples[key_path])
          writer.write(tf_example.SerializeToString())
          print("write record files: {0}/{1} ".format(index+1, len(examples)))
      writer.close()

    print("All class: 3488, class_name 3488, records save dir {}".format(save_dir))


RecordWriter = Record_Writer()


from tensorflow.python.platform import gfile
def write_pbmodel_summery(tf_pbmodel, log_dir):
    with tf.Session() as sess:
        model_filename =tf_pbmodel
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)
    train_writer = tf.summary.FileWriter(log_dir)
    train_writer.add_graph(sess.graph)