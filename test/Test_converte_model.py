#!/usr/bin/env python
# coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# from nose.tools import *
import sys
sys.path.append('/data/project/ImageDT')

from imagedt.converters import caffe_tfmodel
from imagedt.converters import caffe_coreml
from imagedt.converters import tfmodel_lite


class Test_Model_Converte(object):
    def __init__(self):
        super(Test_Model_Converte, self).__init__()

    def caffe_tfmodel_convert(self, proto_def_path, caffemodel_path):
        caffe_tfmodel.convert(proto_def_path, caffemodel_path)

    def test_caffe_tfmodel_convert(self):
        proto_def_path = '/data/dataset/liby_offline/train_renet50/models/deploy.prototxt'
        caffemodel_path = '/data/dataset/liby_offline/train_renet50/models/resnet50_snapshot_iter_100000.caffemodel'
        self.caffe_tfmodel_convert(proto_def_path, caffemodel_path)

    def caffe_coreml_convert(self, model_path, prototxt_path, label_path):
        caffe_coreml.convert(model_path, prototxt_path, label_path, red=-123, green=-117, blue=-104, 
                scale=1.0, bgr=False, output_model_name='sku_cls_model_noise.mlmodel')

    def test_caffe_coreml_convert(self):
        prototxt_path = '/data/dataset/liby_offline/train_renet50/models/deploy.prototxt'
        model_path = '/data/dataset/liby_offline/train_renet50/models/resnet50_snapshot_iter_100000.caffemodel'
        label_path = '/data/dataset/liby_offline/train_renet50/models/sku_labels.txt'
        self.caffe_coreml_convert(model_path, prototxt_path, label_path)

    def tfmodel_lite_convert(self, graph_def_file, output_file):
        tfmodel_lite.convert(graph_def_file, output_file, output_format='TFLITE', inference_type='FLOAT', inference_input_type='FLOAT',
            input_array='data', input_shape='1,224,224,3', output_arrays='softmax')

    def test_tfmodel_lite_convert(self):
        graph_def_file = '/data/dataset/liby_offline/train_renet50/models/standalonehybrid.pb'
        output_file = '/data/dataset/liby_offline/train_renet50/models/py_sku_classify.tflite'
        self.tfmodel_lite_convert(graph_def_file, output_file)


if __name__ == '__main__':
    Converte_init = Test_Model_Converte()
    # Converte_init.test_caffe_tfmodel_convert()
    # Converte_init.test_caffe_coreml_convert()
    Converte_init.test_tfmodel_lite_convert()

