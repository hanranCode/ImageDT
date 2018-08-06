# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from .DataInterface import *


class DataProvider(object):
  """docstring for DataProvider
  train file: image [space] label 
  """
  def __init__(self, train_file, test_percent=0):
    super(DataProvider, self).__init__()
    self.train_file = train_file
    self.test_percent = test_percent
    self._check_infos()
    self.load_image_label_pairs(slef.train_file)

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
