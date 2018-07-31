# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper


class Lite_Wrapper(object):
  """docstring for Lite_Wrapper"""
  def __init__(self, model_path):
    super(Lite_Wrapper, self).__init__()
    self.model_path = model_path
    self.load_model()
    self._check_model_type()

  def load_model(self):
    # load tflite model
    self.interpreter = interpreter_wrapper.Interpreter(model_path=self.model_path)
    self.interpreter.allocate_tensors()
    self.input_details = self.interpreter.get_input_details()
    self.output_details = self.interpreter.get_output_details()

  def _check_model_type(self):
    # only support test float tflite model for now
    try:
      self.input_details[0]['dtype'] == type(np.float32(1.0))
      self.height = input_details[0]['shape'][1]
      self.width = input_details[0]['shape'][2]
    except Exception as e:
      print ("Only support test float tflite model for now, Please check model type!")

  def predict(self, input_data):
    """
    input data : should be cv mat array (axis=0)
    """
    self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
    self.interpreter.invoke()
    predictions = np.squeeze(self.interpreter.get_tensor(self.output_details[0]['index']))
    # get top 1 prediction
    pre_class = predictions.argsort()[::-1][0]
    conf = predictions[pre_class]
    return pre_class, conf
