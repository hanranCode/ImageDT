# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import os
from wand.image import Image


def pdf2image(pdf_path, save_out=None, dpi=300):
  with Image(filename=pdf_path, resolution=dpi) as img:
    if save_out is None:
      save_out = pdf_path[:-4] + '.jpg'
    img.save(filename=save_out)
