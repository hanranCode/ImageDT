# coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from nose.tools import *

import imagedt
from imagedt.converters import caffe_tfmodel
from imagedt.converters import caffe_coreml



class Test_Model_Converte(object):
    def __init__(self):
        super(Test_Model_Converte, self).__init__()

    def caffe_tfmodel_convert(self, proto_def_path, caffemodel_path):
        caffe_tfmodel.convert(proto_def_path, caffemodel_path)

    def test_caffe_tfmodel_convert(self):
        proto_def_path = '/data/models/erroe_detect/deploy.prototxt'
        caffemodel_path = '/data/models/erroe_detect/resnet50_snapshot_iter_4000.caffemodel'
        self.caffe_tfmodel_convert(proto_def_path, caffemodel_path)

    def caffe_coreml_convert(self, model_path, prototxt_path, label_path):
        caffe_coreml.convert(model_path, prototxt_path, label_path, red=-123, green=-117, blue=-104, 
                scale=1.0, bgr=False, output_model_name='sku_cls_model_noise.mlmodel')

    def test_caffe_coreml_convert(self):
        model_path = '/data/models/erroe_detect/resnet50_snapshot_iter_4000.caffemodel'
        prototxt_path = '/data/models/erroe_detect/deploy.prototxt'
        label_path = '/data/models/erroe_detect/sku_labels.txt'

        self.caffe_coreml_convert(model_path, prototxt_path, label_path)



if __name__ == '__main__':
    Converte_init = Test_Model_Converte()
    Converte_init.test_caffe_tfmodel_convert()
    Converte_init.test_caffe_coreml_convert()
