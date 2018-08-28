# coding: utf-8
import sys
sys.path.append('./')


from imagedt.tools import plot_tools



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


if __name__ == '__main__':
  data_dir = '/data/dataset/danong/traindatas'

  plot_init = Test_PlotTools()
  plot_init.plot_datas_distribution(data_dir, 'danon_all_drinks')
