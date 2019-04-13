# coding: utf-8
import os
import numpy as np
import matplotlib.pyplot as plt

import imagedt


def roc(test_file, show_fig=False):
    lines = imagedt.file.readlines(test_file)

    # line: Image_name,True_class_name,Label_id,Predict_id,Confidence,Match
    try:
        int(lines[0].split(',')[-1])
        start_line_num = 0
    except ValueError:
        start_line_num = 1

    results = np.array([[line.split(',')[-2], line.split(',')[-1]] for line in lines[start_line_num:]])
    prob_arr = np.array(results[:, 0], dtype=float)
    match_arr = np.array(results[:, 1], dtype=int)

    # imagedt.tools.set_pdb()
    save_lines, thresvalue = [], []
    tpr, fnr = [], []
    for index, thres in enumerate(np.arange(0,1.001,0.001)):
      tp, fn, fp, tn = 0, 0, 0, 0
      for res_ind, pre_prob in enumerate(prob_arr):
        if float(prob_arr[res_ind]) >= thres:
          if int(match_arr[res_ind]):
            tp += 1
          else:
            fp += 1
        else:
          if int(match_arr[res_ind]):
            fn += 1
          else:
            tn += 1

      y = float(tp) / np.maximum(float(tp + fn), np.finfo(np.float64).eps)
      x = float(fp) / np.maximum(float(fp + tn), np.finfo(np.float64).eps)
      thresvalue.append(thres)
      tpr.append(y)
      fnr.append(x)
      save_lines.append([str(x), str(y), str(thres)])

    save_dir = os.path.dirname(test_file)


    imagedt.file.write_csv(save_lines, os.path.join(save_dir, 'thres-fpr-tpr.csv'))
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    plt.grid()
    plt.xlabel('FPR')
    plt.ylabel('TPR/THRES')
    plt.title('ROC-THRES')
    ax.plot(fnr, tpr, color='red')
    ax.plot(fnr, thresvalue, color='green')
    plt.legend(('roc', 'fpr-thres'),loc='best')

    plt.savefig(os.path.join(save_dir, 'roc.png'), dpi=300)
    print ("ROC curve figure save: {0}".format(save_dir))
    if show_fig:
        plt.show()
