# coding: utf-8

import imagedt

# gt_file line:[image name, predict class] 219549_0.jpg,723
# predict_file line: [image name, predict class, confidence] 219549_0.jpg,723,0.999
# labelt_file line: [object_id, cls_id, cls_name] :6973 0 维他奶维他柠檬茶250ml盒装_6973


gt_file = '/ssd_data/danon/meeting_shelfs/test_outinfos/label_infos.txt'
predict_file = '/ssd_data/danon/meeting_shelfs/test_outinfos/test_infos.txt'
labelt_file = '/ssd_data/danon/cls_records/labels_t.txt'

imagedt.tools.metrics.calculate_f1_score(predict_file, gt_file, labelt_file)