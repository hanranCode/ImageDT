# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from collections import OrderedDict

import imagedt

def load_label_Dict(lines):
  # line: onject_id, class_id, sku_name
  return dict([(line.split('\t')[1], line.split('\t')) for line in lines])


def write_predict_details(predict_infos, label_infos, label_map_dict):
  # save test image details, init list
  jpg_details = [['Image_name', 'True_class_name', 'Label_id', 'Predict_id', 'Confidence', 'Match']]
  for ind in range(len(predict_infos)):
    # GT infos
    true_class = label_infos[ind].split(',')[1]
    true_objectId, _, true_class_name = label_map_dict[true_class]
    # test infos
    file_name, pre_class, pred_conf = predict_infos[ind].split(',')
    pred_object_id, _, _ = label_map_dict[pre_class]
    # set key infos
    match = 1 if int(true_class) == int(pre_class) else 0
    file_name = os.path.basename(file_name)
    jpg_details.append([file_name, true_class_name, true_objectId, pred_object_id, pred_conf, match])
  return jpg_details


def _total_f1_score(cls_dict):
  cor = sum([cls_dict[cls_id]['correct'] for cls_id in cls_dict])
  sum_count = sum([cls_dict[cls_id]['cls_pre_count'] for cls_id in cls_dict])
  total_precision =  round(float(cor)/sum_count, 4) 

  tp_count = sum([cls_dict[cls_id]['TP'] for cls_id in cls_dict])
  sum_count = sum([cls_dict[cls_id]['count'] for cls_id in cls_dict])
  total_recall = round(float(cor)/sum_count, 4) 
  return total_recall, total_precision


def _f1_score(cls_dict, label_map_dict):
  #index_item = map(int, set([item[0] for item in label_map_dict.values() if int(item[0]) != -1]))
  index_item = set([item[0] for item in label_map_dict.values()])
  results = [['SKU_name', 'Object_id', 'Sample_count','TP', 'FN', 'FP', 'Recall', 'Precision', 'F1-score']]
  for cls_id in cls_dict:
    object_id = cls_dict[cls_id]['object_id']
    count = cls_dict[cls_id]['count']
    TP = cls_dict[cls_id]['TP']
    FN = cls_dict[cls_id]['FN']
    FP = cls_dict[cls_id]['FP']

    cls_dict[cls_id]['cls_pre_count'] = TP+FP

    recall = round(float(TP)/np.maximum((TP+FN), np.finfo(np.float64).eps), 4)
    precision = round(float(TP)/np.maximum((TP+FP), np.finfo(np.float64).eps), 4)
    f1_score = round(2*recall*precision/np.maximum((recall+precision), np.finfo(np.float64).eps), 4)

    # col_list = ['SKU_name', 'Object_id', 'Sample_count','TP', 'FN', 'FP', 'Recall', 'Precision']
    sku_name =  label_map_dict[cls_id][2]
    if sku_name.startswith('other'):
      continue
    value_list = [sku_name, object_id, count, TP, FN, FP, recall, precision, f1_score]
    results.append(value_list)

  results = sorted(results, key=lambda x: x[-1])[::-1]
  return results


def gernerate_infos_dict(predicts, true_labels, label_map_dict):
  # predict infos data struct
  cls_dict = OrderedDict()

  for ind in range(len(predicts)):
    true_class = true_labels[ind].split(',')[1]
    true_class_name = label_map_dict[true_class][2]
    true_objectId = label_map_dict[true_class][0]

    file_name, pre_class, pred_conf = predicts[ind].split(',')
    pred_cls_name = label_map_dict[pre_class][2]
    pred_object_id = label_map_dict[pre_class][0]

    if true_class not in cls_dict:
      cls_dict[true_class] = {}
      cls_dict[true_class]['TP'] = 0
      cls_dict[true_class]['FN'] = 0
      cls_dict[true_class]['FP'] = 0
      cls_dict[true_class]['count'] = 0
      cls_dict[true_class]['object_id'] = 0
      cls_dict[true_class]['precision'] = 0
      cls_dict[true_class]['recall'] = 0
      cls_dict[true_class]['correct'] = 0
      cls_dict[true_class]['cls_pre_count'] = 0

    if pre_class not in cls_dict:
      cls_dict[pre_class] = {}
      cls_dict[pre_class]['TP'] = 0
      cls_dict[pre_class]['FN'] = 0
      cls_dict[pre_class]['FP'] = 0
      cls_dict[pre_class]['count'] = 0
      cls_dict[pre_class]['object_id'] = 0
      cls_dict[pre_class]['precision'] = 0
      cls_dict[pre_class]['recall'] = 0
      cls_dict[pre_class]['correct'] = 0
      cls_dict[pre_class]['cls_pre_count'] = 0

    if int(pre_class) == int(true_class):
      cls_dict[true_class]['TP'] += 1
      cls_dict[true_class]['correct'] += 1
      cls_dict[true_class]['object_id'] = true_objectId
    else:
      cls_dict[true_class]['FN'] += 1
      cls_dict[pre_class]['FP'] += 1
      cls_dict[pre_class]['object_id'] = pred_object_id

    cls_dict[true_class]['count'] += 1
  return cls_dict


def calculate_f1_score(predict_file, gt_file, labelt_file):
  """
  predict_file one line: filename, predict_cls, predict_conf]
  gt_file one line: filename, label_cls
  labelt_file one line: object_id, cls_id, sku_name
  """
  labelt_lines = imagedt.file.readlines(labelt_file)
  predict_lines = imagedt.file.readlines(predict_file)
  true_lines = imagedt.file.readlines(gt_file)

  try:
    assert len(predict_lines) == len(true_lines)
  except Exception as e:
    raise ValueError("len(predict infos) != len(groundtrue infos, please check datas.")

  try:
    assert '\t' in labelt_lines[0]
  except Exception as e:
    raise ValueError("check label_t file format, split by '/t'")

  # init classifier
  label_map_dict = load_label_Dict(labelt_lines)
  cls_dict = gernerate_infos_dict(predict_lines, true_lines, label_map_dict)

  # save test image infos
  jpg_details = write_predict_details(predict_lines, true_lines, label_map_dict)
  jpg_infos_name = os.path.join(os.path.dirname(predict_file), 'image_classify_infos.csv')
  imagedt.file.write_csv(jpg_details, jpg_infos_name)
  print("Saved test image infos, save dir:{0}".format(jpg_infos_name))

  # save f1 score
  f1_score_pandas = _f1_score(cls_dict, label_map_dict)
  exlce_name = os.path.join(os.path.dirname(predict_file), 'image_recall_precision.csv')
  imagedt.file.write_csv(f1_score_pandas, exlce_name)
  print("Saved f1 score, save dir:{0}".format(exlce_name))
