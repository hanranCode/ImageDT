#coding: utf-8

import os
import imagedt
import numpy as np

from matplotlib import pyplot as plt



class PlotCurve(object):
  """docstring for PlotCurev"""
  def __init__(self):
    super(PlotCurve, self).__init__()
    self.params = {
        'axes.labelsize': '35',
        'xtick.labelsize':'10',
        'ytick.labelsize':'15',
        'lines.linewidth':2 ,
        'legend.fontsize': '8',
        'figure.figsize': '24, 9', }
    plt.rcParams.update(self.params)

  def load_dataset_infos(self, data_dir):
    class_names = os.listdir(data_dir)
    pos_samples_dict = {}
    for class_name in class_names:
      if os.path.isfile(os.path.join(data_dir, class_name)):
        continue
      if class_name.startswith('other') or class_name.startswith('0'):
        continue
      # pos_samples_dict[class_name] = len(imagedt.dir.loop(os.path.join(data_dir, class_name), ['.jpg', '.png']))
      pos_samples_dict[class_name] = len(imagedt.dir.loop(os.path.join(data_dir, class_name)))
      if len(imagedt.dir.loop(os.path.join(data_dir, class_name))) < 10:
        print(class_name)
    return pos_samples_dict

  def set_xy_values(self, sample_dict):
    pos_list = [[class_name, sample_dict[class_name]] for class_name in sample_dict]
    pos_list = np.array(sorted(pos_list, key=lambda x: int(x[1]))[::-1])
    self.x = range(len(pos_list[:, 0]))
    self.y = map(int, pos_list[:, 1])

  def add_bar_values(self, rects):
    for index, rect in enumerate(rects):
      height = rect.get_height()
      plt.text(rect.get_x() + rect.get_width() / 1.5, height, self.y[index], ha='center', va='bottom',fontsize=3)

  def add_avg_yline(self):
    line_arange = np.arange(0, len(self.x), len(self.x)/5.)
    y_ite = [100, 200, 300, 400, 500]
    for index, ite in enumerate(line_arange):
      plt.axhline(y=int(y_ite[index]), color='blue')
      plt.axvline(x=int(ite), color='blue')

  def plot_datas(self, data_dir, add_bar_values=True, title=None, curve_type='bar'):
    print("loading data infos......")
    sample_dict = self.load_dataset_infos(data_dir)
    # get x,y values
    self.set_xy_values(sample_dict)
    # plot bar
    if curve_type == 'bar':
      rects = plt.bar(range(len(self.x)), self.y, 1, align='edge', color='rgby')
      if add_bar_values:
        self.add_bar_values(rects)
        self.add_avg_yline()
    else:
      plt.plot(self.x, self.y)

    plt.ylim(ymin=0, ymax=max(self.y))
    add_str = '_'+str(len(self.x))+' classes'
    title = os.path.basename(data_dir)+add_str if title is None else title+add_str
    plt.title(title)

    plt.xticks(range(len(self.x)), self.x, rotation=270, fontsize=3)
    plt.ylabel("sample count")

    save_figure_path = os.path.join(os.path.dirname(data_dir), 'datasets.png')
    print("saving figure, path {0}......".format(save_figure_path))
    plt.savefig(save_figure_path, dpi=300)


plot_tools = PlotCurve()

