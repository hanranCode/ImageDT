# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf


class TFmodel_Wrapper(object):
  """docstring for TFmodel_Wrapper"""
  def __init__(self, pbmodel_path, input_nodename='input', output_nodename='softmax',):
    super(TFmodel_Wrapper, self).__init__()
    self.pbmodel_path = pbmodel_path
    self.input_node = input_nodename
    self.output_node = output_nodename
    self._load_model()
    self._check_node_name()
    self._set_output_node()

  def _load_model(self):
    # easy way! load as default graph
    print("load tfmodel {0}".format(self.pbmodel_path))
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(self.pbmodel_path, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
    self.sess = tf.Session()

  def _check_node_name(self):
    # checking input and output node name, notice, notice node name error
    self.ops = tf.get_default_graph().get_operations()
    all_tensor_names = [output.name for op in self.ops for output in op.outputs]
    for node_name in [self.input_node, self.output_node]:
      if not isinstance(node_name, str):
          raise ValueError("Node name: {0} should be string, like: 'input' 'data' 'softmax'.".format(node_name))
      if node_name+':0' not in all_tensor_names:
        print("##### Print some node name : {0}~{1}. #####".format(all_tensor_names[:2], all_tensor_names[-2:]))
        raise ValueError("node_name: {0} not in graph, please check input or output node name!".format(node_name))

  def _set_output_node(self):
    # Get handles to input and output tensors
    self.tensor_dict = {}
    intput_tensor = self.input_node + ':0'
    output_tensor = self.output_node + ':0'
    self.image_tensor = tf.get_default_graph().get_tensor_by_name(intput_tensor)
    self.tensor_dict[self.output_node] = tf.get_default_graph().get_tensor_by_name(output_tensor)

  def predict(self, image):
    # Run inference
    output_dict = self.sess.run(self.tensor_dict, 
                    feed_dict={self.image_tensor: np.expand_dims(image, axis=0)})
    # get top 1 class and confidence
    predict_cls = output_dict[self.output_node][0].argsort()[::-1][0]
    conf = output_dict[self.output_node][0][predict_cls]
    return predict_cls, conf



    # tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
    #                                     log_device_placement=True,
    #                                     ))