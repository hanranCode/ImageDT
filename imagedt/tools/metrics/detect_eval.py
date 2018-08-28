# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import os
import cPickle
import numpy as np
import xml.etree.ElementTree as ET
from collections import OrderedDict

from imagedt.file import write_txt
from imagedt.dir.dir_loop import loop


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = {}
    for obj in tree.findall('object'):
        obj_struct = {}
        name = obj.find('name').text

        if name not in objects:
            objects[name] = []
        # obj_struct['pose'] = obj.find('pose').text
        # obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['name'] = name
        obj_struct['difficult'] = 0

        obj_struct['confidence'] = obj.find('confidence').text if obj.find('confidence') is not None else 0.99
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [float(bbox.find('xmin').text),
                              float(bbox.find('ymin').text),
                              float(bbox.find('xmax').text),
                              float(bbox.find('ymax').text)]
        objects[obj_struct['name']].append(obj_struct)
    return objects


def parse_detect(detannos):
    xmls_path = loop(detannos, ['.xml'])
    print('########## Reading {0} annotation files of DT.##########'.format(len(xmls_path)))
    infos = []
    for xml_path in xmls_path:
        info = parse_rec(xml_path)
        info = [item for cls_key in info for item in info[cls_key]]
        basename = os.path.basename(xml_path)
        infos.append([os.path.splitext(basename)[0], 
            [[item['confidence']]+[item['name']]+item['bbox'] for item in info]])

    return [[item[0], item[1]] for item in infos]


def load_detect_lines(detannos, conf):
    txts_path = loop(detannos, ['.txt'])
    print('########## Reading {0} annotation files of DT.##########'.format(len(txts_path)))
    infos = []
    for txt_path in txts_path:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        true_class = os.path.basename(txt_path).split('.')[0].split('_')[-1]
        # 图片名[无后缀] 置信度 x1 y1 x2 y2
        # target: image_name, conf, det_class, [xmin, ymin, xmax, ymax]
        for line in lines:
            line = line.strip().split(' ')
            if float(line[1]) < conf:
                continue
            infos.append(line[0:2]+[true_class]+line[2:])
    return infos

def get_xmls_basename(xml_path):
    return [os.path.splitext(os.path.basename(item))[0] for item in loop(xml_path, ['.xml'])]


def voc_eval(detpath, annopath, ovthresh=0.5, det_file_type='xml', conf=0.5, set_gt_labels=None):
    ################### read detect results##################
    xmlnames = get_xmls_basename(annopath)
    # load gt
    recs = {}
    for i, imagename in enumerate(xmlnames):
        recs[imagename] = parse_rec(os.path.join(annopath, '{0}.xml').format(imagename))
    print('########## Reading {0} annotation files of GT.##########'.format(len(xmlnames)))

    # extract gt objects
    gt_class_bbox = {}
    class_recs = {}
    npos = 0
    for imagename in xmlnames:
        if det_file_type == 'xml':
            R = [obj for objs in recs[imagename].values() for obj in objs]
            #print(len(R))
        else:
            R = [obj for objs in recs[imagename].values() for obj in objs if int(obj['name']) !=0]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        classes = np.array([x['name'] for x in R])
        for x in R:
            label = x['name']
            if label not in gt_class_bbox.keys():
                gt_class_bbox[label] = 1
            else:
                gt_class_bbox[label] += 1
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det,
                                 'classes':classes}
    print("Ground Truth bbox num: {}\nsum: {}".format(gt_class_bbox, sum(gt_class_bbox.values())))

    ################### read detect results ##################
    if det_file_type == 'xml':
        lines = parse_detect(detpath)
        # splitlines : image_name, conf, det_class, [xmin, ymin, xmax, ymax]
        #splitlines = [[item[0]]+x for item in lines for x in item[1]]
        splitlines = []
        for item in lines:
            for x in item[1]:
                #print(x)
                #print("conf: {}\tthresh: {}".format(x[0], conf))
                if float(x[0]) < conf:
                    #print(x)
                    x[1] = '0'
                #print(x)
                splitlines.append([item[0]]+x)
    else:
        splitlines = load_detect_lines(detpath, conf=conf)

    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    det_classes = np.array([x[2] for x in splitlines])
    #
    det_class_bbox = {}
    for x in splitlines:
        label = x[2]
        if label not in det_class_bbox.keys():
            det_class_bbox[label] = 1
        else:
            det_class_bbox[label] += 1
    print("Predict bbox num: {}\nsum: {}".format(det_class_bbox,sum(det_class_bbox.values())))
    #
    BB = np.array([[float(z) for z in x[3:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    # sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind]
    det_classes = det_classes[sorted_ind]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    cls_tp = np.zeros(nd)
    cls_fp = np.zeros(nd)
    other_tp = 0

    if set_gt_labels is None:
        gt_labels = ['9265','9304','9320','9282','9334','9273','9252','9294']
    else:
        gt_labels = set([ite for item in class_recs.values() for ite in item['classes']])

    tp = {}
    for label in gt_labels:
        if label not in tp.keys():
            tp[label]=0 
    fp = {}
    for label in gt_labels:
        if label not in fp.keys():
            fp[label]=0 
    tn = {}
    for label in gt_labels:
        if label not in tn.keys():
            tn[label]=0 
    fn = {}
    for label in gt_labels:
        if label not in fn.keys():
            fn[label]=0 

    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                R['det'][jmax] = 1
                if int(det_classes[d]) == int(R['classes'][jmax]):
                    for label in gt_labels:
                        if label == det_classes[d]:
                            tp[label] += 1
                        else:
                            tn[label] += 1
                else:
                    for label in gt_labels:
                        if label == det_classes[d]:
                            fp[label] += 1
                        elif label == R['classes'][jmax]:
                            fn[label] += 1
                        else:
                            tn[label] += 1
            else:
                for label in gt_labels:
                    #if label != R['classes'][jmax]:
                    if label != det_classes[d]:
                        tn[label] += 1
                    else:
                        fp[label] += 1
        else:
            for label in gt_labels:
                if label != R['classes'][jmax]:
                    tn[label] += 1
                else:
                    fn[label] += 1
                
    for label in gt_labels:
        miss_gt = gt_class_bbox[label] - (tp[label] + fn[label])
        if miss_gt > 0:
            fn[label] += miss_gt

    print("ground truth: {0}, detect results {1}".format(npos, nd))
    precision = {}
    recall = {}
    f1_score = {}
    for label in gt_labels:
        if label == '0':
            continue
        print("Label: {}".format(label))
        print("\ttp: {}\tfp: {}\ttn: {}\tfn: {}\ttotal: {}".format(tp[label],fp[label],tn[label],fn[label], sum([tp[label],fp[label],tn[label],fn[label]])))
        precision[label] = tp[label] * 1.0 / np.maximum( (tp[label] + fp[label]), np.finfo(np.float64).eps)
        recall[label] = tp[label] * 1.0 / np.maximum( (tp[label] + fn[label]), np.finfo(np.float64).eps)
        f1_score[label] = round(2*(precision[label]*recall[label])/np.maximum((precision[label]+recall[label]), np.finfo(np.float64).eps), 4)
        print ("\t#### recall: {0}, precision: {1} ####".format(round(recall[label], 4), round(precision[label], 4)))
        print ("\t#### f1-score: {0} ##########".format(f1_score[label]))
    
    mean_f1 = np.mean(f1_score.values())
    print("Mean F1: {}\n".format(round(mean_f1, 4)))
    return mean_f1
