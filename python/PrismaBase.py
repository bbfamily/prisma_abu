# -*- encoding:utf-8 -*-
"""
    风格画基类，封装接口与预处理操作
"""
from __future__ import division
from abc import ABCMeta, abstractmethod
import six
import PIL.Image
from PIL import ImageEnhance
from PIL import ImageFilter

__author__ = 'BBFamily'

"""
    模块全局增强参数
"""
g_enhance_p = -1


class BasePrismaClass(six.with_metaclass(ABCMeta, object)):
    @abstractmethod
    def fit_guide_img(self, img_path, gd_path, resize=False, size=480, enhance=None, iter_n=10, **kwargs):
        pass

    @abstractmethod
    def fit_img(self, img_path, resize=False, size=480, enhance=None, iter_n=10, **kwargs):
        pass

    @abstractmethod
    def gd_features_make(self, *args, **kwargs):
        pass

    @abstractmethod
    def do_prisma(self, *args, **kwargs):
        pass

    def resize_img(self, r_img, base_width=480, keep_size=True, h_size=None):
        """
        :param r_img: 原始图像
        :param base_width: 基准宽度
        :param keep_size: 是否保持宽高比，false的话h将使用base_width
        :param h_size
        :return:
        """
        if h_size is None:
            if keep_size:
                w_percent = (base_width / float(r_img.size[0]))
                h_size = int((float(r_img.size[1]) * float(w_percent)))
            else:
                h_size = base_width
        r_img = r_img.resize((base_width, h_size), PIL.Image.ANTIALIAS)
        return r_img

    def handle_enhance(self, r_img, enhance, sharpness=8.8, brightness=1.8, contrast=2.6, color=7.6, contour=2.6):
        """
        预处理图像
        :param r_img: 原始图像
        :param enhance: 处理类型
        :param sharpness: 修改默认值sharpness
        :param brightness: 修改默认值brightness
        :param contrast: 修改默认值contrast
        :param color: 修改默认值color
        :param contour: 修改默认值contour
        :return:
        """
        if enhance == 'Sharpness':
            enhancer = ImageEnhance.Sharpness(r_img)
            sharpness = sharpness if g_enhance_p == -1 else g_enhance_p
            s_img = enhancer.enhance(sharpness)
            img = s_img
        elif enhance == 'Brightness':
            enhancer = ImageEnhance.Brightness(r_img)
            brightness = brightness if g_enhance_p == -1 else g_enhance_p
            b_img = enhancer.enhance(brightness)
            img = b_img
        elif enhance == 'Contrast':
            enhancer = ImageEnhance.Contrast(r_img)
            contrast = contrast if g_enhance_p == -1 else g_enhance_p
            t_img = enhancer.enhance(contrast)
            img = t_img
        elif enhance == 'Color':
            enhancer = ImageEnhance.Color(r_img)
            color = color if g_enhance_p == -1 else g_enhance_p
            c_img = enhancer.enhance(color)
            img = c_img
        elif enhance == 'CONTOUR':
            enhancer = ImageEnhance.Contrast(r_img)
            t_img = enhancer.enhance(contour)
            fc_img = t_img.filter(ImageFilter.CONTOUR)
            img = fc_img
        elif enhance == 'EDGES':
            ffe_img = r_img.filter(ImageFilter.FIND_EDGES)
            img = ffe_img
        elif enhance == 'EMBOSS':
            feb_img = r_img.filter(ImageFilter.EMBOSS)
            img = feb_img
        elif enhance == 'EEM':
            feem_img = r_img.filter(ImageFilter.EDGE_ENHANCE_MORE)
            img = feem_img
        elif enhance == 'EE':
            fee_img = r_img.filter(ImageFilter.EDGE_ENHANCE)
            img = fee_img
        else:
            img = r_img
        return img
