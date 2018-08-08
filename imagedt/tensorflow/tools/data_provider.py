# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf

from .DataInterface import *
from ...image.process import noise_padd

class DataProvider(object):
  """docstring for DataProvider
  train file: image [space] label 
  """
  def __init__(self, train_file, test_percent=0):
    super(DataProvider, self).__init__()
    self.train_file = train_file
    self.test_percent = test_percent
    self._check_infos()
    self.load_image_label_pairs()

  def _check_infos(self):
    assert self.test_percent >= 0
    assert self.test_percent <= 1

  def load_image_label_pairs(self):
    print("loading data......")
    with open(self.train_file, 'r') as f:
      lines = f.readlines()
    split_str = ' ' if ' ' in lines[0] else ','
    self.train_infos = np.array([line.strip().split(split_str) for line in lines])

  @property
  def get_data_provider(self):
    count = len(self.train_infos)
    count_test = int(count * self.test_percent)
    count_train = count - count_test

    datas = DataSets()
    datas.train.set_x(self.train_infos[:, 0][0:count_train])
    datas.train.set_y(self.train_infos[:, 1][0:count_train])
    datas.test.set_x(self.train_infos[:, 0][count_train:])
    datas.test.set_y(self.train_infos[:, 1][count_train:])
    return datas


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


def resize_image_keep_aspect(image, lo_dim=224):
  # Take width/height
  initial_width = tf.shape(image)[0]
  initial_height = tf.shape(image)[1]

  # Take the greater value, and use it for the ratio
  max_value = tf.maximum(initial_width, initial_height)
  ratio = tf.to_float(max_value) / tf.constant(lo_dim, dtype=tf.float32)

  new_width = tf.to_int32(tf.to_float(initial_width) / ratio)
  new_height = tf.to_int32(tf.to_float(initial_height) / ratio)

  return tf.image.resize_images(image, [new_width, new_height])


def tf_noise_padd(images, max_edge=224, start_pixel=0):
  # resize image with scale
  images = resize_image_keep_aspect(images, max_edge)

  height = tf.shape(images)[0]
  width = tf.shape(images)[1]
  channels = 3

  # # height > width
  def case_height_width():
    left_pad_size = tf.div(tf.subtract(max_edge, width), 2)
    right_pad_size = tf.subtract(tf.subtract(max_edge, width), left_pad_size)

    noise_left = tf.random_uniform((height, left_pad_size, channels), minval=start_pixel, maxval=255,dtype=tf.float32)
    noise_right = tf.random_uniform((height, right_pad_size, channels), minval=start_pixel, maxval=255, dtype=tf.float32)

    merge = tf.concat([noise_left, images, noise_right], axis=1)
    return merge

  # width > height
  def case_width_height():
    top_padd_size = tf.div(tf.subtract(max_edge, height), 2)
    bottom_padd_size = tf.subtract(tf.subtract(max_edge, height), top_padd_size)

    noise_top = tf.random_uniform((top_padd_size, width, channels), minval=start_pixel, maxval=255,dtype=tf.float32)
    noise_bottom = tf.random_uniform((bottom_padd_size, width, channels), minval=start_pixel, maxval=255, dtype=tf.float32)

    merge = tf.concat([noise_top, images, noise_bottom], axis=0)
    return merge

  padd_noise_op = tf.cond(tf.greater(height, width), case_height_width, case_width_height)
  return padd_noise_op

