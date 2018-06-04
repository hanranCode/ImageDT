# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np



def noise_padd(img, edge_size=224):
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
        padding = int((edge_size - resize_height) / 2)
        noise_size = (padding, edge_size)
        if channels > 1:
            noise_size += (channels,)
        noise = np.random.randint(255, 256, noise_size).astype('uint8')
        # noise = np.zeros(noise_size).astype('uint8')
        img = np.concatenate((noise, img, noise), axis=0)
    else:
        padding = int((edge_size - resize_width) / 2)
        noise_size = (edge_size, padding)
        if channels > 1:
            noise_size += (channels,)
        noise = np.random.randint(255, 256, noise_size).astype('uint8')
        # noise = np.zeros(noise_size).astype('uint8')
        img = np.concatenate((noise, img, noise), axis=1)

    return img
