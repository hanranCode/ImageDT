# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image


def visual_records(tfrecords_path):  
    # make an input queue from the tfrecord file  
    filename_queue = tf.train.string_input_producer([tfrecords_path])  
    #创建一个reader来读取TFRecord文件  
    reader = tf.TFRecordReader()  
    #从文件中独处一个样例。也可以使用read_up_to函数一次性读取多个样例  
    _, serialized_example = reader.read(filename_queue)  
    #解析每一个元素。如果需要解析多个样例，可以用parse_example函数  
    img_features = tf.parse_single_example(  
        serialized_example,  
        features={  
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string),  
        })  
    #tf.decode_raw可以将字符串解析成图像对应的像素数组  
    # image = tf.decode_raw(img_features['image/encoded'], tf.uint8)  
    image = tf.image.decode_jpeg(img_features['image/encoded'], channels=3)
    label = tf.cast(img_features['image/class/label'], tf.int32)
    height = tf.cast(img_features['image/height'], tf.int32)
    width = tf.cast(img_features['image/width'], tf.int32)

    with tf.Session() as sess:
      # 启动多线程
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      for i in range(2):
        # 获取一张图片和其对应的类型
        images, labels, height, width = sess.run([image, label, height, width])
        Image.fromarray(np.reshape(images, [height, width, 3])).show()
        exit()
