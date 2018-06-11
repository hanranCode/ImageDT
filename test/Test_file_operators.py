#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('/data/project/ImageDT')

from imagedt.file import parse_annotation


class Test_File_operators(object):
    def __init__(self):
        super(Test_File_operators, self).__init__()
        parse_tools = parse_annotation.Anno_OP()

    def parse_annotations(self, xml_path):
        pass

    def test_pars_annotations(self):
        self.parse_annotations()


if __name__ == '__main__':
    File_operate_init = Test_File_operators()