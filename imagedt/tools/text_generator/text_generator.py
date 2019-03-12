# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import time
import random
import argparse
import numpy as np
from tqdm import tqdm

if sys.hexversion < 0x3000000:
	from multiprocessing.dummy import Pool
else:
	from multiprocessing import Pool

from .data_generator import FakeTextDataGenerator


class Text_Generator(object):
	"""docstring for Text_Generator"""
	def __init__(self, font_path, langu='cn'):
		super(Text_Generator, self).__init__()
		self.langu = langu
		self.font_path = font_path
		self.fonts = self._load_fonts
		self._init_multi_threads
		self.set_params()

	@property
	def _load_fonts(self):
		"""
		    Load all fonts in the fonts directories
		"""
		# if self.langu == 'cn':
		# 	cn_font_dir = os.path.join(self.font_dir, 'Chinese_font')
		# 	return [os.path.join(cn_font_dir, font) for font in os.listdir(cn_font_dir)]
		# else:
		# 	en_font_dir = os.path.join(self.font_dir, 'English_font')
		# 	return [os.path.join(en_font_dir, font) for font in os.listdir(en_font_dir)]
		return [os.path.join(self.font_path, font) for font in os.listdir(self.font_path)]

	def set_params(self, tests=None):
		if tests is not None:
			count = len(tests)
			text = tests
		else:
			count = 64
			text = [u'测试快得飞起'] *count

		index = [i for i in range(0, count)]
		font_path = self.fonts[:1] * count
		save_dir = ['/data/tmp/temp/price_tags_fake_dats'] * count
		size = [96] * count
		extension = ['png'] * count
		skewing_angle = [random.randint(0, 20) for _ in range(count)]
		# skewing_angle = [0] * count
		random_skew = [True] * count
		blur = [0] * count
		random_blur = [True] * count
		background_type = [0] * count
		distorsion_type = [0] * count
		distorsion_orientation = [0] * count
		is_handwritten = [False] * count
		name_format = [1] * count
		width = [200] * count
		alignment = [1] * count
		text_color = ['#282828'] * count
		orientation = [0] * count
		space_width = [1.0] * count
		save = [False] * count

		self.args = zip(index,
			text,
			font_path,
			save_dir,
			size,
			extension,
			skewing_angle,
			random_skew,
			blur,
			random_blur,
			background_type,
			distorsion_type,
			distorsion_orientation,
			is_handwritten,
			name_format,
			width,
			alignment,
			text_color,
			orientation,
			space_width,
			save)

	@property
	def _init_multi_threads(self, num_theards=4):
		self.pool = Pool(num_theards)

	def generate_texts(self):
		show_progress = False
		st_time = time.time()
		if show_progress:
			[items for items in tqdm(self.pool.imap_unordered(FakeTextDataGenerator.generate_from_tuple, self.args),
				total=count)]
		else:
			[items for items in self.pool.imap_unordered(FakeTextDataGenerator.generate_from_tuple, self.args)]
		self.pool.terminate()
		print ("time cost: {0:.4f}".format(time.time()-st_time))

	# import imagedt
	# @imagedt.decorator.time_cost
	def gen_price_realtime_datas(self, batch_size=64):
		items, texts = [], []
		for _ in xrange(batch_size):
			text = str(random.randint(1, 99999))
			if len(text) > 2:
				text = text[:-1] + '0'
			texts.append(text)

		self.set_params(texts)
		# for _ in tqdm(self.pool.imap_unordered(FakeTextDataGenerator.generate_from_tuple, self.args),
		# 		total=len(texts)):
		# 	pass
		items = [[np.array(items[0]), items[1]] for items in 
				self.pool.imap_unordered(FakeTextDataGenerator.generate_from_tuple, self.args)]
		# close
		# self.pool.terminate()
		return items
