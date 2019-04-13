# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf

import imagedt
from .DataInterface import *
from ...image.process import noise_padd

slim = tf.contrib.slim


def dynamically_loaded_data(image_paths, labels, height=224, width=224):
  labels = map(int, labels)
  # max(x. 128): prevent memory bomb
  images = np.zeros([max(len(image_paths), 128), height, width, 3], np.float32)
  for index, image_path in enumerate(image_paths):
    cvmat = cv2.imread(image_path)
    h, w, c = cvmat.shape
    if h != 224 or w != 224:
      cvmat =  noise_padd(cvmat, edge_size=224,start_pixel_value=0)
    images[index] = np.array(cvmat)
  return images, labels


def resize_image_keep_aspect(image, max_edge=224):
  # Take width/height
  initial_height = tf.shape(image)[0]
  initial_width = tf.shape(image)[1]

  # Take the greater value, and use it for the ratio
  max_value = tf.maximum(initial_width, initial_height)
  ratio = tf.to_float(max_value) / tf.constant(max_edge, dtype=tf.float32)

  new_width = tf.to_int32(tf.to_float(initial_width) / ratio)
  new_height = tf.to_int32(tf.to_float(initial_height) / ratio)

  def set_w():
    new_width = max_edge
    return new_width, new_height

  def set_h():
    new_height = max_edge
    return new_width, new_height

  new_width, new_height = tf.cond(tf.greater(new_width, new_height), set_w, set_h)

  return tf.image.resize_images(image, [max_edge, new_width])


def tf_noise_padd(images, max_edge=224, start_pixel=0, end_pixel=255):
  # resize image with scale
  images = resize_image_keep_aspect(images, max_edge)

  height = tf.shape(images)[0]
  width = tf.shape(images)[1]
  channels = 3

  # # height > width
  def case_height_width():
    left_pad_size = tf.div(tf.subtract(max_edge, width), 2)
    right_pad_size = tf.subtract(tf.subtract(max_edge, width), left_pad_size)
    noise_left = tf.random_uniform((height, left_pad_size, channels), minval=start_pixel, 
      maxval=end_pixel,dtype=tf.float32)
    noise_right = tf.random_uniform((height, right_pad_size, channels), minval=start_pixel, 
      maxval=end_pixel, dtype=tf.float32)
    # noise_left = tf.ones((height, left_pad_size, channels)) * start_pixel
    # noise_right = tf.ones((height, right_pad_size, channels)) * start_pixel
    merge = tf.concat([noise_left, images, noise_right], axis=1)
    return merge

  # width > height
  def case_width_height():
    top_padd_size = tf.div(tf.subtract(max_edge, height), 2)
    bottom_padd_size = tf.subtract(tf.subtract(max_edge, height), top_padd_size)
    noise_top = tf.random_uniform((top_padd_size, width, channels), minval=start_pixel, 
      maxval=end_pixel, dtype=tf.float32)
    noise_bottom = tf.random_uniform((bottom_padd_size, width, channels), minval=start_pixel, 
      maxval=end_pixel, dtype=tf.float32)
    # noise_top = tf.ones((top_padd_size, width, channels)) * start_pixel
    # noise_bottom = tf.ones((bottom_padd_size, width, channels)) * start_pixel
    merge = tf.concat([noise_top, images, noise_bottom], axis=0)
    return merge

  padd_noise_op = tf.cond(tf.greater(height, width), case_height_width, case_width_height)
  return padd_noise_op


def tf_JitterCut(images, jitter=0.05):
  width, height ,channels = images.shape
  ratio = tf.random_uniform(1, minval=0, maxval=jitter, dtype=tf.float32)
  new_h = height*(1-ratio)
  new_w = width*(1-ratio)

  start_x = tf.random_uniform(0, minval=0, maxval=width-new_w)
  start_y = tf.random_uniform(0, minval=0, maxval=height-new_h)

  return images[start_y:new_h, start_x:new_w]

  # cv::Mat JitterCut(cv::Mat &src, float &jitter){
  #     cv::Mat cv_img = src.clone();
  #     uint64 timeseed =(double)cv::getTickCount();
  #     cv::RNG rng(timeseed);
  #     unsigned int height = cv_img.rows;
  #     unsigned int width = cv_img.cols;
  #     float ratio = rng.uniform(0., jitter);
  #     unsigned int new_h = height*(1-ratio);
  #     unsigned int new_w = width*(1-ratio);
  #     unsigned int start_x = rng.uniform(0, width-new_w);
  #     unsigned int start_y = rng.uniform(0, height-new_h);
  #     cv::Rect roi(start_x,start_y,new_w,new_h);
  #     cv_img = cv_img(roi);
  #     return cv_img;
  #  }


class Data_Provider(object):
  """docstring for Data_Provider"""
  def __init__(self, data_dir, num_classes):
    super(Data_Provider, self).__init__()
    self.data_dir = data_dir
    self.num_classes = num_classes
    self._init_dataset_infos()
    self._init_reader()

  @property
  def _get_tfrecords(self, file_format=['.tfrecord']):
      return imagedt.dir.loop(self.data_dir, file_format)

  def _search_laebls_file(self):
    sear_files = imagedt.dir.loop(self.data_dir, ['.txt'])
    sear_label_file = [item for item in sear_files if ('label' in item) 
      and item.endswith('.txt')]
    if sear_label_file:
      lines = imagedt.file.readlines(sear_label_file[0])
      self.num_classes = len(lines)

  def _init_dataset_infos(self):
    self._search_laebls_file()

  @property
  def _log_dataset_infos(self):
    print('##'*20)
    print ("Dataset infos:\ndata_dir: {0} \nnum_classes: {1}".format(self.data_dir, self.num_classes))
    print('##'*20)

  def _init_reader(self):
    # read all tfrecord files
    self.reader = tf.TFRecordReader()
    record_files = self._get_tfrecords
    self.filequeue = tf.train.string_input_producer(record_files)


  def read_from_tfrecord(self, batch_size=32, image_shape=(224, 224, 3)):
    self.image_shape = image_shape
    _, fetch_tensors = self.reader.read(self.filequeue)
    load_features = tf.parse_single_example(
        fetch_tensors,features={
            'image/height': tf.FixedLenFeature([], dtype=tf.int64),
            'image/width': tf.FixedLenFeature([], dtype=tf.int64),
            'image/class/label': tf.FixedLenFeature([], dtype=tf.int64),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string),
        }
    )
    height = tf.cast(load_features['image/height'], tf.int32)
    width = tf.cast(load_features['image/width'], tf.int32)
    label = tf.cast(load_features['image/class/label'], tf.int64)
    image = tf.image.decode_jpeg(load_features['image/encoded'], channels=image_shape[2])
    image = tf_noise_padd(image, max_edge=image_shape[0], start_pixel=255)
    image = tf.reshape(image, image_shape, name=None)
    # make a batch
    label = slim.one_hot_encoding(label, self.num_classes)
    image_batch, label_batch = tf.train.shuffle_batch([image, label], 
                        batch_size=batch_size, 
                        capacity=500,
                        min_after_dequeue=100,
                        num_threads=4)
    return image_batch, label_batch