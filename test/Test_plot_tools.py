# coding: utf-8
import sys
sys.path.append('./')

import imagedt
from imagedt.tools import plot_tools
from imagedt.tools import pdf


"""
  Add: imagedt

  1、git clone: ttps://github.com/hanranCode/ImageDT.git
     git checkout dev

  2、Add path
    vi ~/.bashrc
    export PYTHONPATH="/path_to_ImageDT:$PYTHONPATH"

    import imagedt
"""


class Test_PlotTools(object):
    def __init__(self):
        super(Test_PlotTools, self).__init__()

    def plot_datas_distribution(self, data_dir, title):
      plot_tools.plot_datas(data_dir, title=title)

      # 目标类统计
      Tese_label = []

      import os
      other_files = 0
      less_10_classes = 0
      less_10_others = 0
      taget_files = 0
      all_classes = os.listdir(data_dir)
      others_cls = 0
      target_cls = 0

      for class_name in all_classes:
        class_files = imagedt.dir.loop(os.path.join(data_dir, class_name), ['.png', '.jpg'])
        file_num = len(class_files)
        if class_name.startswith('other'):
          other_files += file_num
          others_cls += 1
        else:
          taget_files += file_num
          target_cls += 1
        if file_num < 10:
          less_10_classes +=1
          if class_name.startswith('other'):
            less_10_others += 1

      print "总images: ", other_files+taget_files
      print "类别数: ", len(all_classes)
      print "平均每类 images: ", (other_files+taget_files)/len(all_classes)
      print "others 类别: ", others_cls, ' ', other_files, 'images'
      print "目标类 类别: ", target_cls, ' ', taget_files, 'images'
      print "少于10类的类别数: ", less_10_classes ,'(others {0} 类)'.format(less_10_others)


    def test_pdf2image(sefl, pdf_path):
      pdf.pdf2image(pdf_path, dpi=400)


if __name__ == '__main__':
  plot_init = Test_PlotTools()

  data_dir = '/data2/李锦记/数据集/ljj_traindatas_191128_v2_check'
  plot_init.plot_datas_distribution(data_dir, 'lijinji_train_20191128')

  # pdf_path = '/data/dataset/ocr/20181023/invoice/授权经销合范本.pdf'
  # plot_init.test_pdf2image(pdf_path)