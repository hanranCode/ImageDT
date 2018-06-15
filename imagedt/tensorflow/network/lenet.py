# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim 

def lenet(images, num_classes=10, is_training=True, drop_keep_prob=0.5,
          prediction_fn=slim.softmax, scope='LeNet'):
  """
  Args: 
    images: A batch of `Tensors` of size [batch_size, height, width, channels].
    num_classes: classes of the training dataset
    drop_keep_prob: set drop out prob
  """
  end_points = {}

  with tf.variable_scope(scope, 'LeNet', [images]):
    net = end_points['conv1'] = slim.conv2d(images, 32, [5, 5], scope='conv1')
    net = end_points['pool1'] = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    net = end_points['conv2'] = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = end_points['pool2'] = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    net = slim.flatten(net)
    end_points['Flatten'] = net

    net = end_points['fc3'] = slim.fully_connected(net, 1024, scope='fc3')

    if not num_classes:
      return net, end_points

    net = end_points['dropout3'] = slim.dropout(net, drop_keep_prob, is_training=is_training,
          scope='dropout3')

    logits = end_points['Logits'] = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='fc4')

    net = end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

    return logits, end_points

lenet.default_image_size = 28



