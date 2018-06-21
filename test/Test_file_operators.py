#!/usr/bin/env python
# coding: utf-8

import os
import sys
import nose.tools  as ntools

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
        self.test_data_dir = './test/sources/data_dir'
        self.test_train_datas_dir = './test/sources/train_datas_dir'

    def _create_brk_jpg(self):
        create_brk_jpg = os.path.join(self.test_data_dir, 'broken.jpg')
        with open(create_brk_jpg, 'w') as f:
            f.write("'It's boring")
        return create_brk_jpg

    def parse_annotations(self, xml_path):
        self.parse_tools.parse_lab_xml(xml_path)

    def test_pars_annotations(self):
        xml_poaths = dir_loop.loop(self.test_data_dir, ['.xml'])
        self.parse_annotations(xml_poaths[0])

    def test_converte_detect_records(self):
        save_path = './training.tfrecord'
        data_converters.converte_detect_records(self.test_data_dir, save_path)

    ###################### test_file_operators ######################
    def test_rename_file_with_uuid(self):
        self.file_op.rename_file_with_uuid(self.test_data_dir)

    def test_del_broken_image(self):
        # test case: creating a new broken image file , it should be delete
        create_brk_jpg = self._create_brk_jpg()
        ntools.assert_true(os.path.isfile(create_brk_jpg))
        # check broken images
        self.file_op.del_broken_image(self.test_data_dir)
        # deleted file
        ntools.assert_true(os.path.isfile(create_brk_jpg))

    def test_check_data_pairs(self):
        self.file_op.check_data_pairs(self.test_data_dir)

    def test_file_operate_one(self):
        self.file_op.rename_class_dir(self.test_train_datas_dir, before_cls_str='class1')
        classes = os.listdir(self.test_train_datas_dir)[0]
        ntools.assert_false(classes.startswith('other_'))
        self.file_op.rename_class_dir(self.test_train_datas_dir, before_cls_str='other_diao_')
        classes = os.listdir(self.test_train_datas_dir)[0]
        ntools.assert_true(classes.startswith('other'))
    ###################### test_file_operators ######################

    def test_process_remove_brokken(self):
        create_brk_jpg = self._create_brk_jpg()
        ntools.assert_true(os.path.isfile(create_brk_jpg))
        process.remove_broken_image(self.test_data_dir)
        # deleted file
        ntools.assert_false(os.path.isfile(create_brk_jpg))

    def rename_xml_cls_name(self):
        self.parse_tools.reset_xml_cls(self.test_data_dir, name='3488', desc=u'圆柱圆台形')

    def test_write_xml_with_boxandscore(self):
        # write_xml(bndboxs, scores, xmlname, thresh=0.1, classes=None)
        # bndboxs : [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]
        # scores: [0.8, 0.9, score3, score4, score5]
        # len(bndboxs) == len(scores)
        # xmlname: save xml path,eg:"save_dir/xxxxx.xml"
        bndboxs = [[1,1, 101, 101], [11,11, 1101, 1101]]
        scores = [0.9, 0.1]
        xmlname = os.path.join(os.path.dirname(self.test_train_datas_dir), 'text_write.xml')
        self.parse_tools.write_xml(bndboxs, scores, xmlname, thresh=0.1, classes='3488')


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
    # File_operate_init.test_file_operate_one()

    # File_operate_init.converte_detect_records()
    # File_operate_init.rename_xml_cls_name()
    File_operate_init.test_write_xml_with_boxandscore()

    rec, prec, ap = File_operate_init.detect_eval_map()
    print "mean ap: {0}".format(ap)

