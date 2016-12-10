# -*- encoding:utf-8 -*-
from __future__ import division
from PrismaCaffe import CaffePrismaClass
import PrismaHelper
import PrismaEnv
import numpy as np
import PIL.Image
from skimage import filters
from skimage import segmentation
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from scipy.signal import convolve2d
from scipy import ndimage

import matplotlib.pylab as plt

__author__ = 'BBFamily'

"""
    当前运行环境是否ipython notebook
"""
g_show_in_ipython = PrismaEnv.g_show_in_ipython
g_all_mask_rb_rate = PrismaEnv.g_all_mask_rb_rate

g_mask_sp_array_factor = []
g_mask_sp_array = []
g_mask_sp_array_ext_factor = []
g_mask_sp_array_ext = []

g_prisma_image_size = PrismaEnv.g_prisma_image_size
g_prisma_image_keep_size = PrismaEnv.g_prisma_image_keep_size


class PrismaWorkerClass(object):
    def __init__(self):
        self.cp = CaffePrismaClass()
        self.show_func = PrismaHelper.show_array_ipython if g_show_in_ipython else PrismaHelper.show_array

    def show_features(self, gd_file):
        r_img = self.cp.resize_img(PIL.Image.open(gd_file), base_width=g_prisma_image_size, keep_size=False)
        l_img = np.float32(r_img.convert('L'))
        ll_img = np.float32(l_img / 255)

        coords = corner_peaks(corner_harris(ll_img), min_distance=5)
        coords_subpix = corner_subpix(ll_img, coords, window_size=25)

        plt.figure(figsize=(8, 8))
        plt.imshow(r_img, interpolation='nearest')
        plt.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15, mew=5)
        plt.plot(coords[:, 1], coords[:, 0], '.b', markersize=7)
        plt.axis('off')
        plt.show()

    def find_features(self, gd_file=None, r_img=None, l_img=None, loop_factor=1, show=False):
        if gd_file is not None:
            r_img = self.cp.resize_img(PIL.Image.open(gd_file), base_width=g_prisma_image_size, keep_size=False)
            l_img = np.float32(r_img.convert('L'))
            l_img = np.float32(l_img / 255)

        coords = corner_peaks(corner_harris(l_img), min_distance=5)

        r_img_copy = np.zeros_like(l_img)
        rd_img = np.float32(r_img)

        r_img_copy[coords[:, 0], coords[:, 1]] = 1

        f_loop = int(rd_img.shape[1] / 10 * loop_factor)
        for _ in np.arange(0, f_loop):
            """
                放大特征点，使用loop_factor来控制特征放大倍数
            """
            r_img_copy = ndimage.binary_dilation(r_img_copy).astype(r_img_copy.dtype)
        # r_img_copy_ret = r_img_copy * 255

        # if show:
        r_img_copy_d = [rd_img[:, :, d] * r_img_copy for d in range(3)]
        r_img_copy = np.stack(r_img_copy_d, axis=2)

        if show:
            self.show_func(r_img_copy)

        return r_img_copy, r_img

    # noinspection PyUnusedLocal
    def do_otsu(self, r_img, l_img, cb, dd=True):
        """
        :param r_img:
        :param l_img:
        :param cb:
        :param dd: 取正向还是反向，边缘里面的还是外面的
        :return:
        """
        mask = l_img > filters.threshold_otsu(l_img) if dd else l_img < filters.threshold_otsu(l_img)
        clean_border = mask
        if cb:
            clean_border = segmentation.clear_border(mask).astype(np.int)
        clean_border_img = np.float32(clean_border * 255)
        clean_border_img = np.uint8(np.clip(clean_border_img, 0, 255))
        return clean_border_img

    # noinspection PyUnusedLocal
    def do_features(self, r_img, l_img, cb, loop_factor=1.0):
        """
        :param r_img:
        :param l_img:
        :param cb:
        :param loop_factor: 特征放大因子
        :return:
        """
        mask, _ = self.find_features(r_img=r_img, l_img=l_img, loop_factor=loop_factor)
        mask = PIL.Image.fromarray(np.uint8(mask)).convert('L')
        clean_border = mask
        clean_border_img = np.uint8(np.clip(clean_border, 0, 255))
        return clean_border_img

    # noinspection PyUnusedLocal
    def do_stdmean(self, r_img, l_img, cb, std_factor=1.0):
        mean_img = l_img.mean()
        std_img = l_img.std()

        mask1 = l_img > mean_img + (std_img * std_factor)
        mask2 = l_img < mean_img - (std_img * std_factor)

        clean_border = mask1
        if cb:
            clean_border = segmentation.clear_border(mask1).astype(np.int)
        clean_border_img1 = np.float32(clean_border * 255)
        clean_border_img1 = np.uint8(np.clip(clean_border_img1, 0, 255))

        clean_border = mask2
        if cb:
            clean_border = segmentation.clear_border(mask2).astype(np.int)
        clean_border_img2 = np.float32(clean_border * 255)
        clean_border_img2 = np.uint8(np.clip(clean_border_img2, 0, 255))

        clean_border_img = clean_border_img1 + clean_border_img2
        clean_border_img = np.uint8(np.clip(clean_border_img, 0, 255))

        return clean_border_img

    def together_mask_func(self, r_img, l_img, cb, func_list):
        """
        将多个mask func组合与的形式，组合mask滤波器
        exp: tgt_mask_func = partial(together_mask_func, func_list=[do_otsu, mask_stdmean_func, mask_features_func])
        :param r_img
        :param l_img:
        :param cb:
        :param func_list:
        :return:
        """
        clean_border_img = None
        for func in func_list:
            if not callable(func):
                raise TypeError("together_mask_func must a func!!!")
            border_img = func(r_img, l_img, cb)
            if clean_border_img is None:
                clean_border_img = border_img
            else:
                """
                    叠加非0值，之后再np.clip(clean_border_img, 0, 255)
                """
                clean_border_img = clean_border_img + border_img
                # clean_border_img = np.where(clean_border_img > 255, 255, clean_border_img)
        clean_border_img = np.uint8(np.clip(clean_border_img, 0, 255))
        return clean_border_img

    def do_convd_filter(self, n1, n2, n3, rb_rate, r_img, guide_img, clean_border_img, convd_median_factor,
                        convd_big_factor):

        # 最小的卷积核目的是保留大体图像结构
        n = n1
        small_window = np.ones((n, n))
        small_window /= np.sum(small_window)
        clean_border_small = convolve2d(clean_border_img, small_window, mode="same", boundary="fill")

        # 中号的卷积核是为了保留图像的内嵌部分
        n = n2
        median_window = np.ones((n, n))
        median_window /= np.sum(median_window)
        clean_border_convd_median = convolve2d(clean_border_img, median_window, mode="same", boundary="fill")

        # 最大号的卷积核，只是为了去除散落的边缘，很多时候没有必要，影响速度和效果
        n = n3
        big_window = np.ones((n, n))
        big_window /= np.sum(big_window)
        clean_border_convd_big = convolve2d(clean_border_img, big_window, mode="same", boundary="fill")

        l_imgs = []
        for d in range(3):
            """
                针对rgb各个通道处理
            """
            rd_img = r_img[:, :, d]
            gd_img = guide_img[:, :, d]

            wn = []
            for _ in np.arange(0, rd_img.shape[1]):
                """
                    二项式概率分布
                """
                wn.append(np.random.binomial(1, rb_rate, rd_img.shape[0]))
            if rb_rate <> 1:
                """
                    针对rgb通道阶梯下降二项式概率
                """
                rb_rate -= 0.1
            w = np.stack(wn, axis=1)

            # 符合保留条件的使用原始图像，否则使用特征图像
            d_img = np.where(np.logical_or(
                np.logical_and(clean_border_convd_median > convd_median_factor * clean_border_convd_big.mean(), w == 1),
                np.logical_and(np.logical_and(clean_border_small > 0, w == 1),
                               clean_border_convd_big > convd_big_factor * clean_border_convd_big.mean(),
                               )), rd_img, gd_img)

            l_imgs.append(d_img)
        img_cvt = np.stack(l_imgs, axis=2).astype("uint8")
        return img_cvt

    def handle_mask_sp_array(self, clean_border_img):
        def do_mask_sp_array(cb_img, sp_array, mask_sp_array_factor, sp_value):
            for sp_mask, msf in zip(sp_array, mask_sp_array_factor):
                y = int(sp_mask[0])
                x = int(sp_mask[1])
                k_size = msf

                x_dunk = filter(lambda x_sub: 0 < x_sub < cb_img.shape[1], np.arange(x - k_size, x + k_size))
                y_dunk = filter(lambda y_sub: 0 < y_sub < cb_img.shape[0], np.arange(y - k_size, y + k_size))

                if not len(x_dunk) == len(y_dunk):
                    ml = min(len(x_dunk), len(y_dunk))
                    x_dunk = x_dunk[:ml - 1]
                    y_dunk = y_dunk[:ml - 1]

                xs, ys = np.meshgrid(x_dunk, y_dunk)

                cb_img[xs, ys] = sp_value
            return cb_img

        clean_border_img = do_mask_sp_array(clean_border_img, g_mask_sp_array, g_mask_sp_array_factor, 255)
        clean_border_img = do_mask_sp_array(clean_border_img, g_mask_sp_array_ext, g_mask_sp_array_ext_factor, 0)
        return clean_border_img

    def mix_mask_with_convd(self, do_mask_func, org_file=None, gd_file=None, nbk=None, enhance=None, n1=5, n2=25,
                            n3=120,
                            convd_median_factor=5.0, convd_big_factor=1.0, cb=False,
                            rb_rate=1, r_img=None, guide_img=None, all_mask=False, save=False, show=False):

        r_img_h = None
        if r_img is None:
            r_img = self.cp.resize_img(PIL.Image.open(org_file), base_width=g_prisma_image_size,
                                       keep_size=g_prisma_image_keep_size)
            if g_prisma_image_keep_size:
                """
                    如果keep size下面的guide要使用org的size
                """
                r_img_h = r_img.size[1]

        l_img = np.float32(r_img.convert('L'))
        l_img = np.float32(l_img / 255)
        r_img = np.float32(r_img)

        if show:
            self.show_func(np.float32(r_img))

        if not callable(do_mask_func):
            raise TypeError('mix_mask_with_convd must do_mask_func a func')

        clean_border_img = np.ones_like(l_img) * 255 if all_mask else do_mask_func(r_img=r_img, l_img=l_img, cb=cb)

        clean_border_img = self.handle_mask_sp_array(clean_border_img)

        if show:
            self.show_func(np.float32(clean_border_img))

        if guide_img is None:
            if gd_file is not None:
                guide_img = np.float32(self.cp.resize_img(PIL.Image.open(gd_file), base_width=g_prisma_image_size,
                                                          keep_size=g_prisma_image_keep_size, h_size=r_img_h))
            else:
                guide_img = np.zeros_like(r_img)

        if all_mask and rb_rate == 1:
            """
                如果只做浅层edges特征
                1. 强制将rgb三通道随机梯度渲染打开，不然没有意义
                2. 强制将n1 = n2 = n3 = 1，提升效率
            """
            rb_rate = g_all_mask_rb_rate
            n1 = n2 = n3 = 1

        img_cvt = self.do_convd_filter(n1, n2, n3, rb_rate, r_img, guide_img, clean_border_img,
                                       convd_median_factor=convd_median_factor, convd_big_factor=convd_big_factor)

        if nbk is not None:
            # 对转换出的图像进行一次简单浅层特征放大
            img_cvt = self.cp.fit_img(org_file, nbk=nbk, iter_n=10, enhance=enhance, img_np=img_cvt, save=save)

        if show:
            self.show_func(np.float32(img_cvt))
        return img_cvt
