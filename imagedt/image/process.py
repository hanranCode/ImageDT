# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import os
import cv2
import sys
import hashlib
import numpy as np
from PIL import Image

from ..dir.dir_loop import loop

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.


def clear_dir(images_dir, debug = True):
    """清理目录下破损图片"""
    images_path = loop(images_dir, IMG_EXTENSIONS)
    count = len(images_path)
    for i, image_path in enumerate(images_path):
        if debug:
            sys.stdout.write('\r%d/%d' % (i, count))
            sys.stdout.flush()
        try:
            Image.open(image_path).convert('RGB')
        except Exception as e:
            print("Remove broken image: {0}".format(image_path))
            os.remove(image_path)


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
  # B, G, R = cv2.split(cvmat)
  # return cv2.merge([R, G, B])
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def vgg_preprocesing(cvmat):
  cvmat = np.array(cvmat, dtype=np.float32)
  cvmat -= [_R_MEAN,_G_MEAN,_B_MEAN]
  return cvmat


def inception_preprocesing(cvmat):
  cvmat = (cvmat / 255. - 0.5) * 2
  cvmat = np.array(cvmat, dtype=np.float32)
  return cvmat


def our_processing(cvmat):  
  cvmat = reduce_imagenet_mean(cvmat)
  cvmat = np.array(cvmat, dtype=np.float32)
  return np.expand_dims(cvmat, axis=0)

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_box(bndbox, loca_type='mid'):
    if loca_type == 'top':
        # xmid = bndbox[0]+(bndbox[2]-bndbox[0])/2.
        xmid = bndbox[0]
        ymid = bndbox[1]
    else:
        xmid = bndbox[0]+(bndbox[2]-bndbox[0])/2.
        ymid = bndbox[1]+(bndbox[3]-bndbox[1])/2.
    return (int(xmid), int(ymid))


def put_cvtext(img, text, posi, color=(0,0,255), fontScale=1, thickness=3, font_type=None):
    if font_type is None:
        font = cv2.FONT_HERSHEY_SIMPLEX
    else:
        font = font_type
    cv2.putText(img=img, text=text, org=posi, fontFace=font, fontScale=fontScale, 
        color=color, thickness=thickness)


def draw_cvrect(img, bndboxes, box_color=(0,0,255), border=1, font_color=(0,255,0), draw_strs=None, 
                    fontScale=3, compare_strs=None, draw_location='mid', font_type=None):
    for index, bndbox in enumerate(bndboxes):
        bndbox = map(int, bndbox)
        left_up = (bndbox[0], bndbox[1])
        right_bottom = (bndbox[2], bndbox[3])
        cv2.rectangle(img, left_up, right_bottom, box_color, border)
        mid_box = get_box(bndbox, loca_type=draw_location)
        if draw_strs is not None:
            show_str = draw_strs[index]
        else:
            show_str = str(index)
        put_cvtext(img, show_str, mid_box, color=font_color, fontScale=fontScale, thickness=3, font_type=font_type)
    return img


def cv_show(imgs, win_name='show_image'):
    if not isinstance(imgs, list):
        imgs = [imgs]
    for index, img in enumerate(imgs):
        cv2.imshow(win_name+str(index), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_dir_images_with_scale(image_dir, resize_max_edge=224):
    image_files = loop(image_dir, ['jpg', 'png'])
    for ite_ind, image_file in enumerate(image_files, 1):
        img = cv2.imread(image_file)
        if img is not None:
            if max(img.shape) > 224:
                img = resize_with_scale(img, 224)
                cv2.imwrite(img, image_file)
        else:
            shutil.move(image_file)
            print("delete broken image file {0}".format(image_file))
        print ("resize dir images max edge to {0}, finished {1}/{2}".format())


def points_to_contour(points):
    contours = [[list(p)]for p in points]
    return np.asarray(contours, dtype=np.int32)

def points_to_contours(points):
    return np.asarray([points_to_contour(points)])

def put_text(img, text, pos, scale=1, color=(255, 255, 255), thickness=1):
    pos = np.int32(pos)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img = img, text = text, org = tuple(pos), fontFace = font,  
        fontScale = scale,  color = color, thickness = thickness)


def draw_contours(img, contours, idx = -1, color = 1, border_width = 1):
    x1,y1,x2,y2,x3,y3,x4,y4 = contours[0].reshape(8, -1)
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=10, lineType=8, shift=0)
    for ind, box in enumerate(contours[0]):
        cx, cy = box[0]
        cv2.circle(img, (cx, cy), 10, (0,0,255), -1)
        put_text(img, str(ind+1), (cx ,cy))
    cv2.drawContours(img, contours, idx, color, border_width)
    return img

def polly_points2contours(points):
    points = [np.reshape(item, [4, 2]) for item in points]
    return points_to_contours(points)[0]
