# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import os
import cv2
import numpy as np

from ..dir.dir_loop import loop

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.


def noise_padd(img, edge_size=224, start_pixel_value=0):
    """
    img: cvMat
    edge_size: image max edge

    return: cvMat [rectangle, height=width=edge_size]
    """
    h, w, _ = img.shape

    width_ratio = float(w) / edge_size
    height_ratio = float(h) / edge_size
    if width_ratio > height_ratio:
        resize_width = edge_size
        resize_height = int(round(h / width_ratio))
        if (edge_size - resize_height) % 2 == 1:
            resize_height += 1
    else:
        resize_height = edge_size
        resize_width = int(round(w / height_ratio))
        if (edge_size - resize_width) % 2 == 1:
            resize_width += 1
    img = cv2.resize(img, (int(resize_width), int(resize_height)), interpolation=cv2.INTER_LINEAR)

    channels = 3
    # fill ends of dimension that is too short with random noise
    if width_ratio > height_ratio:
        padding = (edge_size - resize_height) / 2
        noise_size = (padding, edge_size)
        if channels > 1:
            noise_size += (channels,)
        noise = np.random.randint(start_pixel_value, 256, noise_size).astype('uint8')
        # noise = np.zeros(noise_size).astype('uint8')
        img = np.concatenate((noise, img, noise), axis=0)
    else:
        padding = (edge_size - resize_width) / 2
        noise_size = (edge_size, padding)
        if channels > 1:
            noise_size += (channels,)
        noise = np.random.randint(start_pixel_value, 256, noise_size).astype('uint8')
        # noise = np.zeros(noise_size).astype('uint8')
        img = np.concatenate((noise, img, noise), axis=1)
    return img


def remove_broken_image(data_dir):
    image_files = loop(data_dir, IMG_EXTENSIONS)

    for image_file in image_files:
        try:
            img_mat = cv2.imread(image_file)

            if img_mat is None:
                os.remove(image_file)
                print('remove broken image {0}'.format(image_file))
        except:
            os.remove(image_file)
            print('remove broken file {0}'.format(image_file))


def resize_with_scale(cvmat, max_length):
    h, w, _ = cvmat.shape
    max_edge = max(h, w)
    if max_edge > max_length:
        ratio = float(max_edge) / max_length
        width, height = w/ratio, h/ratio
        cvmat = cv2.resize(cvmat, (int(width), int(height)))
    return cvmat


def padd_pixel(img, edge_size_w=200, edge_size_h=96):
    h, w, _ = img.shape

    width_ratio = float(w) / edge_size_w
    if width_ratio > 1:
        img = cv2.resize(img, (int(w/width_ratio), int(h/width_ratio)))
        h, w, _ = img.shape
        width_ratio = float(w) / edge_size_w

    height_ratio = float(h) / edge_size_h
    if height_ratio > 1:
        img = cv2.resize(img, (int(w/height_ratio), int(h/height_ratio)))
        h, w, _ = img.shape
        height_ratio = float(h) / edge_size_h

    if width_ratio > height_ratio:
        resize_width = edge_size_w
        resize_height = int(round(h / width_ratio))
        if (edge_size_h - resize_height) % 2 == 1:
            resize_height += 1
    else:
        resize_height = edge_size_h
        resize_width = int(round(w / height_ratio))
        if (edge_size_w - resize_width) % 2 == 1:
            resize_width += 1
    img = cv2.resize(img, (int(resize_width), int(resize_height)), interpolation=cv2.INTER_LINEAR)

    channels = 3
    if width_ratio > height_ratio:
        padding = (edge_size_h - resize_height) / 2
        noise_size = (padding, edge_size_w)
        if channels > 1:
            noise_size += (channels,)
        noise = np.random.randint(230, 240, noise_size).astype('uint8')
        img = np.concatenate((noise, img, noise), axis=0)
    else:
        padding = (edge_size_w - resize_width) / 2
        noise_size = (edge_size_h, padding)
        if channels > 1:
            noise_size += (channels,)
        noise = np.random.randint(230, 240, noise_size).astype('uint8')
        img = np.concatenate((noise, img, noise), axis=1)
    return img


def reduce_imagenet_mean(cv_mat):
  [B, G, R] = cv2.split(cv_mat)
  return cv2.merge([B-_B_MEAN, G-_G_MEAN, R-_R_MEAN])


def swap_chanel_to_RGB(cvmat):
  B, G, R = cv2.split(cvmat)
  return cv2.merge([R, G, B])


def vgg_preprocesing(cvmat):
  cvmat = np.array(cvmat, dtype=np.float32)
  cvmat -= [_R_MEAN,_G_MEAN,_B_MEAN]
  return cvmat

def inception_preprocesing(cvmat):
  cvmat = (cvmat / 255. - 0.5) * 2
  cvmat = np.array(cvmat, dtype=np.float32)
  return cvmat
