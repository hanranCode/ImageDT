# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import os
import cv2
import uuid
import numpy as np
import tensorflow as tf
from lxml import etree

from . import dataset_util
from ...dir import dir_loop


# 过滤了瓶盖
class_dict = {'3488': 1, '3477': 2, '3487': 3, '3486': 4, '3649': 5, '3648': 6, '3647': 7,
              '3650': 8, '3651': 9, '3652': 10, '3653': 11, '3172': 12}


def parse_lab_xml(xml):
    """
    :param xml: xml path ,infos: ['name', 'desc', 'bndbox':[xmin, ymin ,xmax, ymax]]
    :return: cls sequence int [xmin, ymin, xmax, ymax]
    """
    annotation = etree.parse(xml).getroot()
    bndx_infos = []
    for index, tag in enumerate(annotation.iter('object')):
        label = [class_dict.get(tag.find('name').text, None)]
        # if label[0] is None:
        #     label 
        text = [tag.find('name').text]
        bndbox = [float(tag.find('bndbox').find(coord).text) for coord in ['xmin', 'ymin', 'xmax', 'ymax']]
        # bndx_infos.append(bndbox+label+text)
        bndx_infos.append(bndbox+['3488']+['3488'])
    return bndx_infos


def get_img_anno_pair(data_dir):
    """
    :param data_dir: include: JPEGImages and Annotations dir
    :return: data_pairs: [(jpg_path, xml_path), ...]
    """
    data_pairs = []
    image_path = os.path.join(data_dir, 'JPEGImages')

    images = dir_loop.loop(image_path)
    for image in images:
        baesname = os.path.basename(image)
        xml_name = baesname.split('.')[0]+'.xml'
        xml_path = os.path.join(data_dir, 'Annotations', xml_name)
        if os.path.isfile(xml_path):
            data_pairs.append([image, xml_path])
    return data_pairs


def converte_anno_info(data_dir):
    data_pairs = get_img_anno_pair(data_dir)
    examples = {}
    for index, data_pair in enumerate(data_pairs):
        jpg_path, xml_path = data_pair
        anno_infos = parse_lab_xml(xml_path)
        examples[jpg_path] = anno_infos

    print("get image pairs: ", len(examples))
    return examples


def map_int(m_list):
    return np.array(map(int, map(float, m_list)))


def create_tf_example(jpg_path, anno_infos):
    # Bosch
    height, width, chanel = cv2.imread(jpg_path).shape
    filename = jpg_path.encode()  # Filename of the image. Empty if image is not from file

    with tf.gfile.GFile(jpg_path, 'rb') as fid:
        encoded_image = fid.read()

    image_type = os.path.basename(jpg_path).split('.')[-1]
    image_format = image_type.encode()

    anno_infos = np.array(anno_infos)

    xmins = map_int(anno_infos[:, 0]) / float(width)  # List of normalized left x coordinates in bounding box (1 per box)
    ymins = map_int(anno_infos[:, 1]) / float(height)  # List of normalized top y coordinates in bounding box (1 per box)
    xmaxs = map_int(anno_infos[:, 2]) / float(width)  # List of normalized right x coordinates in bounding box # (1 per box)
    ymaxs = map_int(anno_infos[:, 3]) / float(height)  # List of normalized bottom y coordinates in bounding box # (1 per box)
    classes = map_int(anno_infos[:, 4])  # List of integer class id of bounding box (1 per box)
    classes_text = anno_infos[:, 5]  # List of string class name of bounding box (1 per box)

    assert len(xmins) == len(ymins) == len(xmaxs) == len(ymaxs) == len(classes) == len(classes_text)

    tf_example = tf.train.Example(features=tf.train.Features(
        feature={'image/height': dataset_util.int64_feature(height),
                 'image/width': dataset_util.int64_feature(width),
                 'image/channels': dataset_util.int64_feature(chanel),
                'image/filename': dataset_util.bytes_feature(filename),
                'image/source_id': dataset_util.bytes_feature(filename),
                'image/encoded': dataset_util.bytes_feature(encoded_image),
                'image/format': dataset_util.bytes_feature(image_format),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes), }))
    return tf_example


def converte_detect_records(data_dir, save_path):
    examples = converte_anno_info(data_dir)
    writer = tf.python_io.TFRecordWriter(save_path)

    # TODO(user): Write code to read in your dataset to examples variable

    # print("Loaded ", len(examples.values()), "examples")
    for index, key_path in enumerate(examples):
        tf_example = create_tf_example(key_path, examples[key_path])
        writer.write(tf_example.SerializeToString())
        print("write record: ", index+1, '/', len(examples))
    writer.close()
    
