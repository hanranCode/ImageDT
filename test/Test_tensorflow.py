# coding: utf-8
import sys
sys.path.append('./')
import numpy as np
import nose.tools as ntools
from imagedt.tensorflow.tools import DataInterface



class Test_Own_Tensorflow(object):
    def __init__(self):
      super(Test_Own_Tensorflow, self).__init__()
      self.data_instance = DataInterface.DataSets()
      self.gen_datasets()

    def gen_datasets(self):
      self.data_instance.train.set_x(np.zeros([32,10]))
      self.data_instance.train.set_y(np.zeros([32]))
      self.data_instance.test.set_x(np.ones([32, 10]))
      self.data_instance.test.set_y(np.ones([32]))

    def test_get_next_batch_datas(self):
      train_datas, train_labels = self.data_instance.train.next_batch(8, isFixed=False)
      ntools.assert_equal(train_datas[0][0], 0)
      ntools.assert_equal(train_labels[0], 0)

      test_datas, test_labels = self.data_instance.test.next_batch(8, isFixed=False)
      ntools.assert_equal(test_datas[0][0], 1)
      ntools.assert_equal(test_labels[0], 1)


if __name__ == '__main__':
  owntf_init = Test_Own_Tensorflow()
  owntf_init.test_get_next_batch_datas()
