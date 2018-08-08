# coding: utf-8
import os
import csv
import math
import numpy as np
from lxml import etree


class Anno_OP(object):
    def __init__(self):
        super(Anno_OP, self).__init__()

    def read_xml(self, fxml):
        """
        :param fxml: xml file path.
        :return: annotation is object: annotation.iter('object').
        """
        annotation = etree.parse(fxml).getroot()
        return annotation

    def parse_bndx_object(self, bndbox_object):
        """return bndx list ['xmin', 'ymin', 'xmax', 'ymax'] by float"""
        rectangle = [bndbox_object.find(coord).text for coord in ['xmin', 'ymin', 'xmax', 'ymax']]
        return map(float, rectangle)

    def read_txt_bndbox(self, fpath):
        with open(fpath, 'r') as f:
            lines = f.readlines()
        return [line.strip().split(',') for line in lines]

    def parse_lab_xml(self, fxml):
        annotation = self.read_xml(fxml)
        return [self.parse_bndx_object(iter_object.find('bndbox')) for iter_object in annotation.iter('object')]

    def load_annoataion(self, poly_file):
        '''
        load annotation from the text file
        :param p:
        :return:
        '''
        text_polys = []
        text_tags = []
        if not os.path.exists(poly_file):
            return np.array(text_polys, dtype=np.float32)
        with open(poly_file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                label = line[-1]
                # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
                text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                if label == '*' or label == '###':
                    text_tags.append(True)
                else:
                    text_tags.append(False)
            return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)

    def parse_poly_locations(self, loca_info):
        xmin, ymin = map(math.floor, loca_info[0])
        xmax, ymax = map(math.ceil, loca_info[2])
        return xmin, ymin, xmax, ymax

    def convert_poly_to_Xml(self, poly_file):
        loca_infos, has_fonts = self.load_annoataion(poly_file)

        xml_dir = os.path.join(os.path.dirname(os.path.dirname(poly_file)), 'Annotations_xml')
        xml_name = os.path.join(xml_dir, os.path.basename(poly_file).replace('.txt', '.xml'))

        if not os.path.exists(xml_dir):
            os.mkdir(xml_dir)

        annotation = etree.Element("Annotation")
        for index, loca_info in enumerate(loca_infos):
            vocObject = etree.SubElement(annotation, "object")
            name = etree.SubElement(vocObject, "name")

            if has_fonts[index] == True:
                # fonts are blur
                name.text = '2'
            else:
                name.text = '1'

            desc = etree.SubElement(vocObject, "desc")
            desc.text = unicode('文字文本', 'utf-8')
            xmins, ymins, xmaxs, ymaxs = map(str, self.parse_poly_locations(loca_info))

            bndbox = etree.SubElement(vocObject, "bndbox")
            xmin = etree.SubElement(bndbox, "xmin")
            xmin.text = xmins
            ymin = etree.SubElement(bndbox, "ymin")
            ymin.text = ymins
            xmax = etree.SubElement(bndbox, "xmax")
            xmax.text = xmaxs
            ymax = etree.SubElement(bndbox, "ymax")
            ymax.text = ymaxs

        xml = etree.tostring(annotation, pretty_print=True, encoding='UTF-8')
        with open(xml_name, "w") as xmlfile:
            xmlfile.write(xml)

    def reset_xml_cls(self, xml_dir, name='3488', desc=u'圆柱圆台形' ):
        for index, f_path in enumerate(os.listdir(xml_dir)):
            xml_path = os.path.join(xml_dir, f_path)
            an_objects = self.read_xml(xml_path)
            for tag in an_objects.iter('object'):
                tag.find('name').text = name
                if tag.find('desc') is not None:
                     tag.find('desc').text = desc
            xml = etree.tostring(an_objects, pretty_print=True, encoding='UTF-8')
            with open(xml_path, "w") as xmlfile:
                xmlfile.write(xml)
            print("finished reset {0}/{1}".format(index+1, len(os.listdir(xml_dir))))

    def write_xml(self, bndboxs, scores, xmlname, thresh=0.1, classes='3488'):

        annotation = etree.Element("Annotation")
        for index, loca_info in enumerate(bndboxs):
            if scores[index] < thresh:
                continue
            vocObject = etree.SubElement(annotation, "object")
            name = etree.SubElement(vocObject, "name")
            name.text = classes

            desc = etree.SubElement(vocObject, "desc")
            desc.text = unicode('object', 'utf-8')

            confi = etree.SubElement(vocObject, "confidence")
            confi.text = str(scores[index])

            # xmin ,ymin, xmax, ymax
            xmins, ymins, xmaxs, ymaxs = map(str, loca_info)
            # ymin, xmin, ymax, xmax
            # ymins, xmins, ymaxs, xmaxs = map(str, loca_info)

            bndbox = etree.SubElement(vocObject, "bndbox")
            xmin = etree.SubElement(bndbox, "xmin")
            xmin.text = xmins
            ymin = etree.SubElement(bndbox, "ymin")
            ymin.text = ymins
            xmax = etree.SubElement(bndbox, "xmax")
            xmax.text = xmaxs
            ymax = etree.SubElement(bndbox, "ymax")
            ymax.text = ymaxs

        xml = etree.tostring(annotation, pretty_print=True, encoding='UTF-8')
        with open(xmlname, "w") as xmlfile:
            xmlfile.write(xml)

Anno_OP = Anno_OP()