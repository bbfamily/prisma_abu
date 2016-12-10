# -*- encoding:utf-8 -*-
"""
    prisma辅助模块
"""
from __future__ import division

from cStringIO import StringIO
import IPython.display
import numpy as np
import PIL.Image
from PIL import ImageDraw, ImageFont
import PrismaEnv
__author__ = 'BBFamily'


K_LOGO = PrismaEnv.K_LOGO
g_enable_print_log = PrismaEnv.g_enable_print_log
g_print_log_pic = PrismaEnv.g_print_log_pic


def print_logo(fn):
    if not g_enable_print_log or PrismaEnv.g_prisma_image_size > 640:
        return

    im = PIL.Image.open(fn).convert('RGBA')
    if g_print_log_pic:
        mark = PIL.Image.open("../logo.png")
        layer = PIL.Image.new('RGBA', im.size, (0, 0, 0, 0))
        layer.paste(mark, (im.size[0] - mark.size[0], im.size[1] - mark.size[1]))
        out = PIL.Image.composite(layer, im, layer)
    else:
        txt = PIL.Image.new('RGBA', (108, 36), (155, 42, 28, 128))
        fnt = ImageFont.truetype("../mode/Condensed_Bold.ttf", 20)
        d = ImageDraw.Draw(txt)
        d.text((15, 10), K_LOGO, font=fnt, fill=(255, 255, 255, 255))
        layer = PIL.Image.new('RGBA', im.size, (0, 0, 0, 0))
        layer.paste(txt, (im.size[0] - txt.size[0], im.size[1] - txt.size[1]))
        out = PIL.Image.composite(layer, im, layer)

    a = np.uint8(np.clip(out, 0, 255))
    with open(fn, 'w') as f:
        PIL.Image.fromarray(a).save(f, 'jpeg')


def show_array_ipython(a, fmt='jpeg'):
    """
    仅当使用ipython notebook时显示在notebook上
    Parameters
    ----------
    a
    fmt
    Returns
    -------
    """
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))


def show_array(a):
    """
    PIL显示
    :param a: np array
    :return:
    """
    a = np.uint8(np.clip(a, 0, 255))
    PIL.Image.fromarray(a).show()


def save_array_img(a, file_name, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    with open(file_name, 'w') as f:
        PIL.Image.fromarray(a).save(f, fmt)
    print_logo(file_name)


def preprocess_with_roll(img, mean_pixel):
    return np.float32(np.rollaxis(img, 2)[::-1]) - mean_pixel


def deprocess_with_stack(img, mean_pixel):
    return np.dstack((img + mean_pixel)[::-1])
