#!/usr/bin/env python
# coding: utf-8
import gc
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# from nose.tools import *
import sys
sys.path.append('./')

from imagedt.converters import caffe_tfmodel
from imagedt.converters import caffe_coreml
from imagedt.converters import tfmodel_lite
from imagedt.decorator import time_cost

class Test_Model_Converte(object):
    def __init__(self):
        super(Test_Model_Converte, self).__init__()

    @time_cost
    def caffe_tfmodel_convert(self, proto_def_path, caffemodel_path, output_model_name='freeze_graph_model.pb'):
        caffe_tfmodel.convert(proto_def_path, caffemodel_path)

    def test_caffe_tfmodel_convert(self):
        proto_def_path = '/data/dataset/liby_offline/train_renet50/models/deploy.prototxt'
        caffemodel_path = '/data/dataset/liby_offline/train_renet50/models/resnet50_snapshot_iter_60000.caffemodel'
        self.caffe_tfmodel_convert(proto_def_path, caffemodel_path)

    @time_cost
    def caffe_coreml_convert(self, model_path, prototxt_path, label_path):
        caffe_coreml.convert(model_path, prototxt_path, label_path, red=-123, green=-117, blue=-104, 
                scale=1.0, bgr=False, output_model_name='sku_cls_model_noise.mlmodel')

    def test_caffe_coreml_convert(self):
        prototxt_path = '/data/dataset/liby_offline/train_renet50/models/deploy.prototxt'
        model_path = '/data/dataset/liby_offline/train_renet50/models/resnet50_snapshot_iter_60000.caffemodel'
        label_path = '/data/dataset/liby_offline/train_renet50/models/sku_labels.txt'
        self.caffe_coreml_convert(model_path, prototxt_path, label_path)

    def tfmodel_lite_convert(self, graph_def_file, output_file):
        tfmodel_lite.convert(graph_def_file, output_file, output_format='TFLITE', inference_type='FLOAT', inference_input_type='FLOAT',
            input_array='data', input_shape='1,224,224,3', output_arrays='softmax')

    @time_cost
    def test_tfmodel_lite_convert(self):
        # TODO fix bug: pipeline excu program
        graph_def_file = '/data/project/idt-jackfruit-mobile-detect/data/ssd_mobile_512_liby_1/shel_detect_model.ckpt.pb'
        output_file = '/data/project/idt-jackfruit-mobile-detect/data/ssd_mobile_512_liby_1/shel_detect_model.tflite'
        self.tfmodel_lite_convert(graph_def_file, output_file)


if __name__ == '__main__':
    Converte_init = Test_Model_Converte()
    Converte_init.test_caffe_coreml_convert()
    # Converte_init.test_caffe_tfmodel_convert()
    # Converte_init.test_tfmodel_lite_convert()
