#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.append('./')



from imagedt.file import parse_annotation, file_operate
from imagedt.tensorflow.tools import data_converters
from imagedt.dir import dir_loop
from imagedt.image import process
from imagedt.file import detect_eval
from imagedt.file import file_write


class Test_File_operators(object):
    def __init__(self):
        super(Test_File_operators, self).__init__()
        self.parse_tools = parse_annotation.Anno_OP()
        self.file_op = file_operate.FilesOp()
        self.data_dir = '/data/dataset/shelf_datasets/liby_merge_shlef'

    def parse_annotations(self, xml_path):
        self.parse_tools.parse_lab_xml(xml_path)

    def test_pars_annotations(self):
        xml_poaths = dir_loop.loop(self.data_dir, ['.xml'])
        self.parse_annotations(xml_poaths[0])

    def test_converte_detect_records(self):
        save_path = '/data/dataset/shelf_datasets/liby_merge_shlef/training.tfrecord'
        data_converters.converte_detect_records(self.data_dir, save_path)

    def test_file_operators(self):
        self.file_op.rename_file_with_uuid(self.data_dir)
        self.file_op.del_broken_image(self.data_dir)
        self.file_op.check_data_pairs(self.data_dir)

    def test_file_operate_one(self):
        self.file_op.rename_class_dir('/data/dataset/liby_offline/train_renet50/zip', 'other_diao_')

    def remove_brokken(self):
        process.remove_broken_image('/data/dataset/liby_offline/train_renet50/zip')

    def rename_xml_cls_name(self):
        self.parse_tools.reset_xml_cls('/data/dataset/shelf_datasets/test_detect_save/Annotations')

    def detect_eval_map(self):
        detpath = '/data/dataset/shelf_datasets/test_jpg/Annotations'
        annopath = '/data/dataset/shelf_datasets/test_jpg/Annotations2'

        rec, prec, ap = detect_eval.voc_eval(detpath, annopath, '3488',
                                                     ovthresh=0.5,
                                                     use_07_metric=False)
        return rec, prec, ap


if __name__ == '__main__':
    File_operate_init = Test_File_operators()
    # File_operate_init.test_pars_annotations()
    # File_operate_init.test_file_operators()
    # File_operate_init.test_converte_detect_records()
    # File_operate_init.test_file_operate_one()
    # File_operate_init.rename_xml_cls_name()
    rec, prec, ap = File_operate_init.detect_eval_map()

    print "mean ap: {0}".format(ap)
