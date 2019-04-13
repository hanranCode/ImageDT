# coding: utf-8
from PIL import Image
import numpy as np
import collections


def alpha_to_color(image, color=(255, 255, 255)):
    """Set all fully transparent pixels of an RGBA image to the specified color.
    This is a very simple solution that might leave over some ugly edges, due
    to semi-transparent areas. You should use alpha_composite_with color instead.
    Source: http://stackoverflow.com/a/9166671/284318
    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)
    """ 
    x = np.array(image)
    r, g, b, a = np.rollaxis(x, axis=-1)
    # print np.where(a != 0)
    r[a == 0] = color[0]
    g[a == 0] = color[1]
    b[a == 0] = color[2]
    a = np.ones_like(a) * 255
    x = np.dstack([r, g, b, a])
    return Image.fromarray(x, 'RGBA')
