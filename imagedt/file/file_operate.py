# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import os
import cv2
import uuid

from ..dir.dir_loop import loop

DIR_TYPR = ['JPEGImages', 'Annotations']


class FilesOp(object):
    def __init__(self):
        super(FilesOp, self).__init__()

    def get_abs_path(self, file_dir, files_list):
        return [os.path.join(file_dir, f_item) for f_item in files_list]

    def check_dir_exist(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
            print("mkidr {0}".format(path))

    def check_file_exist(self, fpath):
        if not os.path.exists(fpath):
            raise IOError("{0} not exist".format(fpath))

    def fillter_dirname(self, file_paths):
        lines = list()
        for file_path in file_paths:
            imtes = file_path.strip().split(',')
            bsname = os.path.basename(imtes[0])
            lines.append([bsname, imtes[1]])
        return lines

    def fillter_file_type(self, file_paths):
        return [''.join(file_path.split('.')[0]) for file_path in file_paths]

    def rename_with_uuid(self, fname, uid):
        dir_name = os.path.dirname(fname)
        bs_name = os.path.basename(fname)
        try:
            self.assert_in_op('.', bs_name)
        except Exception as e:
            print(e) 
            raise ValueError("file {0} basename error ".format(bs_name))
        return os.path.join(dir_name, uid+'.'+bs_name.split('.')[-1])

    def rename_data_pairs(self, file_pair, uid):
        for f_path in file_pair:
            new_fpath = self.rename_with_uuid(f_path, uid)
            os.rename(f_path, new_fpath)

    def assert_not_in_op(self, member, container):
        try:
            assert member not in container
        except Exception as e:
            print(e) 
            raise ValueError('Value {0} contains {1}'.format(container, member))

    def assert_in_op(self, member, container):
        try:
            if isinstance(member, type(list())):
                for assert_str in member:
                    assert assert_str in container
            else:
                assert member in container
        except Exception as e:
            print(e) 
            raise ValueError("Value {0} didn't contain {1}".format(container, member))

    def assert_instance_op(self, inst, tar_inst=list()):
        try:
            assert type(inst) == type(tar_inst)
        except Exception as e:
            print(e)
            raise ValueError("input Values should be list...")

    def del_broken_image(self, root_dir):
        image_files = os.listdir(os.path.join(root_dir, DIR_TYPR[0]))
        for index, img in enumerate(image_files):
            img_name = os.path.join(root_dir, 'JPEGImages', img)
            imat = cv2.imread(img_name)
            if imat is None:
                os.remove(img_name)
                print("remove broken image: {0}".format(img))
            print("check_data_pairs finished: {0} / {1}".format(index + 1, len(image_files)))

    def check_data_pairs(self, root_dir, anno_type='.xml'):
        # root dir should contain 'JPEGImages' and 'Annotations' dirs
        self.assert_in_op(DIR_TYPR, os.listdir(root_dir))

        images_files = self.fillter_file_type(os.listdir(os.path.join(root_dir, DIR_TYPR[0])))
        allxml_files = self.fillter_file_type(os.listdir(os.path.join(root_dir, DIR_TYPR[1])))

        num_images = len(images_files)
        num_xmlfil = len(allxml_files)
        if num_images == num_xmlfil and len(set(images_files+allxml_files)) == num_images:
            print("check_data_pairs finished: effective data pairs {0} . ".format(len(images_files)))
            return "have same files"

        for index, file_name in enumerate(allxml_files):
            if file_name not in images_files:
                print("annotation file {0} not in JPG".format(file_name))
                os.remove(os.path.join(root_dir, DIR_TYPR[1], file_name+anno_type))
            print("check_data_pairs finished: {0} / {1}".format(index+1, num_xmlfil))

    def get_jpg_xml_pairs(self, root_dir, anno_type='.xml'):
        """return: image and xml pairs abs path [image_path, xml_path]......"""
        file_pairs = list()
        allxml_dir = os.path.join(root_dir, DIR_TYPR[1])
        images_dir = os.path.join(root_dir, DIR_TYPR[0])
        xml_files = self.get_abs_path(allxml_dir, os.listdir(allxml_dir))
        for xml_path in xml_files:
            imgae_bsname = os.path.basename(xml_path).replace(anno_type, '.jpg')
            image_path = os.path.join(images_dir, imgae_bsname)
            file_pairs.append([image_path, xml_path])
        return file_pairs

    def rename_file_with_uuid(self, root_dir):
        """
        root_dir contains: JPEGImages and Annotations
        """
        self.check_data_pairs(root_dir)
        file_pairs = self.get_jpg_xml_pairs(root_dir)
        for index, file_pair in enumerate(file_pairs):
            uid = str(uuid.uuid1())
            self.rename_data_pairs(file_pair, uid)
            print("finished: {0} / {1}.".format(index+1, len(file_pairs)))

    def rename_single_file_uuid(self, root_dir):
        f_names = self.get_abs_path(root_dir, os.listdir(root_dir))
        for index, f_name in enumerate(f_names):
            f_uuid = str(uuid.uuid1())
            new_fpath = self.rename_with_uuid(f_name, f_uuid)
            os.rename(f_name, new_fpath)
            print("finished {0}/{1}".format(index+1, len(f_names)))

    def loop_dir(self, root_dir):
        dep_files = list()
        dirs = self.get_abs_path(root_dir, os.listdir(root_dir))
        for item in dirs:
            dep_files += self.get_abs_path(item, os.listdir(item))
        return dep_files

    def rename_class_dir(self, root_dir, before_cls_str='other'):
        all_files = loop(root_dir, ['.jpg'])
        all_dirs = dict([(item, 1) for item in all_files])

        for class_dir in all_dirs:
            class_dir_name = os.path.dirname(class_dir)
            set_name = class_dir_name.replace(os.path.basename(class_dir_name), before_cls_str + os.path.basename(class_dir_name))

            if not class_dir.startswith(before_cls_str):
                try:
                    os.renames(class_dir_name, set_name)
                    print("rename {0} as {1}".format(os.path.basename(class_dir_name), os.path.basename(set_name)))
                except Exception as e:
                    print(class_dir_name)

FilesOp = FilesOp()