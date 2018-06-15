#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('./')

from imagedt.file import parse_annotation, file_operate
from imagedt.tensorflow.tools import data_converters
from imagedt.dir import dir_loop

class Test_File_operators(object):
    def __init__(self):
        super(Test_File_operators, self).__init__()
        self.parse_tools = parse_annotation.Anno_OP()
        self.file_op = file_operate.FilesOp()
        self.data_dir = 'test/sources/data_dir'

    def parse_annotations(self, xml_path):
        self.parse_tools.parse_lab_xml(xml_path)

    def test_pars_annotations(self):
        xml_poaths = dir_loop.loop(self.data_dir, ['.xml'])
        self.parse_annotations(xml_poaths[0])

    def test_converte_detect_records(self):
        save_path = './test/sources/training.tfrecord'
        data_converters.converte_detect_records(self.data_dir, save_path)

    def test_file_operators(self):
        self.file_op.rename_file_with_uuid(self.data_dir)
        self.file_op.del_broken_image(self.data_dir)
        self.file_op.check_data_pairs(self.data_dir)


if __name__ == '__main__':
    File_operate_init = Test_File_operators()
    File_operate_init.test_pars_annotations()
    File_operate_init.test_converte_detect_records()
    File_operate_init.test_file_operators()
