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
        infos.append([os.path.splitext(basename)[0], [[item['confidence']]+[item['name']]+item['bbox'] for item in info]])

    return [[item[0], item[1]] for item in infos]


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_xmls_basename(xml_path):
    return [os.path.splitext(os.path.basename(item))[0] for item in loop(xml_path, ['.xml'])]


def voc_eval(detpath,
             annopath,
             ovthresh=0.5,
             use_07_metric=False,
             metric_type='f1_score'):

    ################### read detect results##################
    xmlnames = get_xmls_basename(annopath)
    # load gt
    recs = {}
    gt_cls = []
    for i, imagename in enumerate(xmlnames):
        recs[imagename] = parse_rec(os.path.join(annopath, '{0}.xml').format(imagename))
        gt_cls.append(recs[imagename].keys())
    print('########## Reading {0} annotation files of GT.##########'.format(len(xmlnames)))

    ################### read detect results##################
    # splitlines = [x.strip().split(' ') for x in lines]
    lines = parse_detect(detpath)
    splitlines = [[item[0]]+x for item in lines for x in item[1]]
    # add dict classes

    classes_dict = {}
    [classes_dict.setdefault(line[2], []).append(line) for line in splitlines]

    # extract gt objects for this class
    maps = {}
    f1_scores = {}
    gt_clses = set([item for items in gt_cls for item in items])
    for gt_cls in gt_clses:
        if gt_cls not in classes_dict:
            f1_scores[gt_cls] = 0
            maps[gt_cls] = 0
            continue
        class_recs = {}
        npos = 0
        for imagename in xmlnames:
            # R = [obj for obj in recs[imagename] if obj['name'] == classname]
            R = [obj for key_cls in recs[imagename] for obj in recs[imagename][key_cls] if key_cls == gt_cls]

            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            classes = np.array([x['name'] for x in R])
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det,
                                     'classes':classes}


        # test detection results
        splitlines = classes_dict[gt_cls]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        det_classes = np.array([x[2] for x in splitlines])
        BB = np.array([[float(z) for z in x[3:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        cls_tp = np.zeros(nd)
        cls_fp = np.zeros(nd)

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
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
                    # cls correct
                    assert type(det_classes[d]) == type(R['classes'][jmax])
                    if det_classes[d] == R['classes'][jmax]:
                        cls_tp[d] = 1.
                    else:
                        cls_fp[d] = 1.
            else:
                fp[d] = 1.
                cls_fp[d] = 1.

        # import pdb
        # pdb.set_trace()
        # compute precision recall
        if metric_type == 'f1_score':
            cls_fp = np.sum(cls_fp)
            cls_tp = np.sum(cls_tp)
            rec = np.sum(tp) / float(np.sum(tp)+np.sum(fp))
            prec = cls_tp / np.maximum(cls_tp + cls_fp, np.finfo(np.float64).eps)
            f1_mean_score = 2*(rec*prec)/np.maximum(rec+prec, np.finfo(np.float64).eps)
            f1_scores[gt_cls] = f1_mean_score
            # return f1_mean_score
        else:
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = voc_ap(rec, prec, use_07_metric)
            # return ap
            maps[gt_cls] = ap
    if metric_type == 'f1_score':
        return f1_scores
    else:
        return maps
