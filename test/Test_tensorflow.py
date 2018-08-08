# coding: utf-8
import sys
sys.path.append('./')
import os
os.environ['CUDA_VISIBLE_DIVICES'] = '1'
import cv2
import numpy as np
import tensorflow as tf
import nose.tools as ntools

from imagedt.tensorflow.tools import DataInterface
from imagedt.tensorflow.tools import data_provider
from imagedt.tensorflow.tools import RecordWriter
import imagedt

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

    def test_data_provider(self):
      # train_file line: image [space] label
      train_file = '/data/dataset/pg_one/PG/PG_data0726v2/train_datas/train_datasets_original/train.txt'
      Data_Provider = data_provider.DataProvider(train_file, test_percent=0.01)
      datas = Data_Provider.get_data_provider

      images, labels = datas.train.next_batch(32)
      images, labels = data_provider.dynamically_loaded_data(images, labels)

    def write_class_tfrecords(self):
      # write_cls_records(train_image_file, labels_file, save_dir)
      # train.train_images_file: image.name label[split by ' ']
      # labels_file: class_name    label[split by '\t']
      train_images_file = '/data/dataset/liby_offline/train_renet50/balance_trainset/train.txt'
      labels_file = '/data/dataset/liby_offline/train_renet50/balance_trainset/labels.txt'
      save_dir = '/ssd_data/liby/cls_eval_records/'
      RecordWriter.write_cls_records(train_images_file, labels_file, save_dir)

    def write_detect_tfrecords(self):
      # data_dir: contains {Jpg, Annotations}
      data_dir = './test/sources/data_dir/'
      save_dir = '/ssd_data/tmp'
      RecordWriter.write_detect_records(data_dir, save_dir)

    def noise_padd_op(self):
      img_file = imagedt.dir.loop('./test/sources/data_dir/', ['.jpg'])
      cvmat = cv2.imread(img_file[0])

      with tf.Session() as sess:
        noise_image = imagedt.tensorflow.tools.data_provider.tf_noise_padd(cvmat)
        image = sess.run(noise_image)

      cv2.imwrite('/ssd_data/tmp/origin.jpg', cvmat)
      cv2.imwrite('/ssd_data/tmp/noise_image.jpg', image)

if __name__ == '__main__':
  owntf_init = Test_Own_Tensorflow()
  owntf_init.noise_padd_op()
