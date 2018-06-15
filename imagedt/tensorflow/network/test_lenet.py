# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for slim.nets.vgg."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from . import lenet

slim = tf.contrib.slim


class LeNetTest(tf.test.TestCase):
  print("init test: LeNetTest...")

  def testBuild(self):
    batch_size = 5
    height, width = 28, 28
    num_classes = 10
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = lenet(inputs, num_classes)
      self.assertEquals(logits.op.name, 'LeNet/fc4/BiasAdd')
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])

  def testFullyConvolutional(self):
    batch_size = 1
    height, width = 28, 28
    num_classes = 10
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = lenet(inputs, num_classes)
      self.assertEquals(logits.op.name, 'LeNet/fc4/BiasAdd')
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])

  def testGlobalPool(self):
    batch_size = 1
    height, width = 28, 28
    num_classes = 10
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = lenet(inputs, num_classes)
      self.assertEquals(logits.op.name, 'LeNet/fc4/BiasAdd')
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])

  def testEndPoints(self):
    batch_size = 5
    height, width = 28, 28
    num_classes = 10
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      _, end_points = lenet(inputs, num_classes)
      expected_names = ['conv1', 
                        'pool1',
                        'conv2', 
                        'pool2', 
                        'Flatten',
                        'fc3',
                        'dropout3', 
                        'Logits', 
                        'Predictions']

      self.assertSetEqual(set(end_points.keys()), set(expected_names))

  def testEvaluation(self):
    batch_size = 2
    height = width = lenet.default_image_size  # set train imgae size
    num_classes = 10
    with self.test_session():
      eval_inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = lenet(eval_inputs, is_training=False)
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])
      predictions = tf.argmax(logits, 1)
      self.assertListEqual(predictions.get_shape().as_list(), [batch_size])

  def testForward(self): 
    batch_size = 1
    height = width = lenet.default_image_size  # set train imgae size
    with self.test_session() as sess:
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = lenet(inputs)
      sess.run(tf.global_variables_initializer())
      output = sess.run(logits)
      self.assertTrue(output.any())

  def TODO_train(self):
    imgae_dir = './test/images'
    images_path = os.listdir(imgae_dir)

    for image_path in images_path:
      cv2.imread(os.path.join(image_dir, image_path))



if __name__ == '__main__':
  tf.test.main()
