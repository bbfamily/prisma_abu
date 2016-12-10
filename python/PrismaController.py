# -*- encoding:utf-8 -*-
from __future__ import division

import os
import glob
import PIL.Image
from traits.api import HasTraits, Button, Int, Range, Bool, Enum, Str, List, Float
from traitsui.api import View, Item, Group, VGroup, HGroup, ListStrEditor, UItem
import traitsui.file_dialog
from functools import partial
import datetime

import PrismaBase
import PrismaHelper
from PrismaWorker import PrismaWorkerClass
import PrismaWorker
import PrismaMaster
import PrismaEnv

__author__ = 'BBFamily'

"""
    使用参数批量艺术图片使用小数据集
"""
g_use_min_batch_set = PrismaEnv.g_use_min_batch_set
"""
    是否开启选择图片预览显示
"""
g_select_show = PrismaEnv.g_select_show


class PrismaControllClass(HasTraits):
    prisma_worker = PrismaWorkerClass()

    print_pool = List(Str)

    nbk_list = filter(lambda nbk: nbk[-8:-1] <> '_split_', prisma_worker.cp.net.blobs.keys()[1:-2])
    nbk_list.insert(0, None)
    nbk_enum = Enum(nbk_list)

    enhance_enum = Enum(
        [None, 'Sharpness', 'Brightness', 'Contrast', 'Color', 'CONTOUR', 'EDGES', 'EMBOSS', 'EEM', 'EE'])
    """
        增强效果的参数
    """
    enhance_p = float(-1)
    mask_enum = Enum(['otsu_func', 'stdmean_func', 'features_func', 'otsu_func + stdmean_func',
                      'otsu_func + features_func', 'stdmean_func + features_func',
                      'otsu_func + stdmean_func + features_func'])
    sample_enum = Enum(glob.glob('../sample/*.jpg'))
    """
        insert(0, None)可以没有引导特征
    """
    gd_list = glob.glob('../prisma_gd/*.jpg')
    gd_list.insert(0, None)
    guide_enum = Enum(gd_list)

    """
        暂时不连接这个
    """
    select_img = Button()

    prisma_img = Button()

    master_img = Button()

    force_iter_n = Int(PrismaEnv.g_force_iter_n)

    n1_convd = Int(5)
    n2_convd = Int(28)
    n3_convd = Int(1)

    convd_median_factor = Range(0, 5., 1.5)
    convd_big_factor = Range(0, 1., 0.0)
    rb_rate = Range(0, 1., 1.)

    stdmean_func_factor = Range(0, 2., 1.0)
    features_func_factor = Range(0, 2., 1.0)

    cb = Bool(False)
    dd = Bool(True)
    all_mask = Bool(False)
    all_mask_rb_rate = Float(PrismaWorker.g_all_mask_rb_rate)

    def print_info(self):
        self.print_pool = []

        self.print_pool.append(u'图像路径: {}'.format(self.org_file))
        self.print_pool.append(u'特征路径: {}'.format(self.gd_file))

        self.print_pool.append(u'特征放大层: {}'.format(self.nbk_enum))
        self.print_pool.append(u'预处理增强: {}'.format(self.enhance_enum))
        self.print_pool.append(u'预处理参数: {}'.format(self.enhance_p))

        self.print_pool.append(u'图像mask函数: {}'.format(self.mask_enum))

        self.print_pool.append(u'n1-小卷积核: {}'.format(self.n1_convd))
        self.print_pool.append(u'n2-中卷积核: {}'.format(self.n2_convd))
        self.print_pool.append(u'n3-大卷积核: {}'.format(self.n3_convd))

        self.print_pool.append(u'是否取otsu内部: {}'.format(self.dd))
        self.print_pool.append(u'只做浅层edges特征渲染: {}'.format(self.all_mask))
        self.print_pool.append(u'浅层edges渲染概率: {}'.format(self.all_mask_rb_rate))
        self.print_pool.append(u'执行clear_border: {}'.format(self.cb))
        self.print_pool.append(u'stdmean mask因子: {}'.format(self.stdmean_func_factor))
        self.print_pool.append(u'features mask因子: {}'.format(self.features_func_factor))
        self.print_pool.append(u'RGB随机渲染因子: {}'.format(self.rb_rate))
        self.print_pool.append(u'中卷积阀值因子: {}'.format(self.convd_median_factor))
        self.print_pool.append(u'大卷积阀值因子: {}'.format(self.convd_big_factor))

    def open_file(self, **traits):
        fn = traitsui.file_dialog.open_file(**traits)
        if len(fn) > 0:
            self.org_file = fn
            self.prisma_worker.cp.resize_img(PIL.Image.open(fn), base_width=320).show()

    def __init__(self):
        super(PrismaControllClass, self).__init__()
        self.org_file = self.sample_enum
        self.gd_file = None
        self.print_info()

    def _guide_enum_fired(self):
        self.gd_file = self.guide_enum
        self.print_info()

        if g_select_show:
            self.prisma_worker.cp.resize_img(PIL.Image.open(self.gd_file), base_width=320, keep_size=False).show()

    def _sample_enum_fired(self):
        self.org_file = self.sample_enum
        self.print_info()

        if g_select_show:
            self.prisma_worker.cp.resize_img(PIL.Image.open(self.org_file), base_width=320, keep_size=False).show()

    def _nbk_enum_fired(self):
        self.print_info()

    def _mask_enum_fired(self):
        self.print_info()

    def _n1_convd_fired(self):
        self.print_info()

    def _n2_convd_fired(self):
        self.print_info()

    def _n3_convd_fired(self):
        self.print_info()

    def _dd_fired(self):
        self.print_info()

    def _cb_fired(self):
        self.print_info()

    def _convd_median_factor_fired(self):
        self.print_info()

    def _convd_big_factor_fired(self):
        self.print_info()

    def _stdmean_func_factor_fired(self):
        self.print_info()

    def _features_func_factor_fired(self):
        self.print_info()

    def _all_mask_fired(self):
        self.print_info()

    def _all_mask_rb_rate_fired(self):
        PrismaWorker.g_all_mask_rb_rate = self.all_mask_rb_rate
        self.print_info()

    def _enhance_enum_fired(self):
        self.print_info()

    def _enhance_p_fired(self):
        self.print_info()

    def _rb_rate_fired(self):
        self.print_info()

    def _force_iter_n_fired(self):
        PrismaEnv.g_force_iter_n = self.force_iter_n

    def _select_img_fired(self):
        self.open_file()

    def _master_img_fired(self):
        nbk_list = ['conv2/3x3_reduce', 'conv2/3x3'] if g_use_min_batch_set else filter(
            lambda nbk: nbk[-8:-1] <> '_split_',
            self.prisma_worker.cp.net.blobs.keys()[1:-2])[:10]

        org_file_list = [self.org_file]

        gd_file_list = glob.glob('../prisma_gd/*.jpg')
        if g_use_min_batch_set:
            """
                小数据集只取 sm*.jpg
            """
            gd_file_list = filter(lambda fn: os.path.basename(fn).startswith('sm'), gd_file_list)
        # 无引导的完成在allmask中进行循环
        # gd_file_list.insert(0, None)

        PrismaBase.g_enhance_p = float(self.enhance_p)

        enhance_list = [None, 'Sharpness', 'Contrast'] if g_use_min_batch_set else [None, 'Sharpness', 'Contrast',
                                                                                    'CONTOUR']

        rb_rate_list = [1.0] if g_use_min_batch_set else [0.88, 1.0]

        # all_mask = self.all_mask
        all_mask = True

        save_dir = '../out/' + str(datetime.date.today())[:10]

        PrismaMaster.product_prisma(org_file_list, gd_file_list, nbk_list, enhance_list, rb_rate_list,
                                    self.mask_enum, self.n1_convd, self.n2_convd, self.n3_convd, self.dd,
                                    self.stdmean_func_factor, self.features_func_factor, self.convd_median_factor,
                                    self.convd_big_factor, self.cb, all_mask, save_dir)

    def _prisma_img_fired(self):

        # 使用partial统一mask函数接口形式
        mask_stdmean_func = partial(self.prisma_worker.do_stdmean, std_factor=self.stdmean_func_factor)
        mask_features_func = partial(self.prisma_worker.do_features, loop_factor=self.features_func_factor)
        mask_otsu_func = partial(self.prisma_worker.do_otsu, dd=self.dd)

        if self.mask_enum == 'otsu_func':
            mask_func = mask_otsu_func
        elif self.mask_enum == 'features_func':
            mask_func = mask_features_func
        elif self.mask_enum == 'stdmean_func':
            mask_func = mask_stdmean_func

        elif self.mask_enum == 'otsu_func + stdmean_func':
            mask_func = partial(self.prisma_worker.together_mask_func, func_list=[mask_otsu_func,
                                                                                  mask_stdmean_func])
        elif self.mask_enum == 'otsu_func + features_func':
            mask_func = partial(self.prisma_worker.together_mask_func, func_list=[mask_otsu_func,
                                                                                  mask_features_func])
        elif self.mask_enum == 'stdmean_func + features_func':
            mask_func = partial(self.prisma_worker.together_mask_func, func_list=[mask_stdmean_func,
                                                                                  mask_features_func])

        elif self.mask_enum == 'otsu_func + stdmean_func + features_func':
            mask_func = partial(self.prisma_worker.together_mask_func, func_list=[mask_otsu_func,
                                                                                  mask_stdmean_func,
                                                                                  mask_features_func])
        else:
            raise ValueError('_prisma_img_fired NO MATCH mask_enum!!!')

        """
            针对模块的增强参数外面设置就可以了
        """
        PrismaBase.g_enhance_p = float(self.enhance_p)

        if self.gd_file is None and self.nbk_enum is not None:
            """
                如果不需要特征渲染，这里的rb_rate被用来使用切换iter_n数量
            """
            iter_n = int((1 / self.rb_rate) * 10)
            prisma_img = self.prisma_worker.cp.fit_img(self.org_file, nbk=self.nbk_enum, iter_n=iter_n, resize=True,
                                                       enhance=self.enhance_enum, save=False)
        else:
            prisma_img = self.prisma_worker.mix_mask_with_convd(mask_func, self.org_file, self.gd_file, self.nbk_enum,
                                                                enhance=self.enhance_enum, rb_rate=self.rb_rate,
                                                                cb=self.cb,
                                                                n1=self.n1_convd, n2=self.n2_convd, n3=self.n3_convd,
                                                                convd_median_factor=self.convd_median_factor,
                                                                convd_big_factor=self.convd_big_factor,
                                                                all_mask=self.all_mask, save=False,
                                                                show=False)

        PrismaHelper.show_array(prisma_img)

        PrismaHelper.save_array_img(prisma_img, '../tmp.jpg')


    view = View(HGroup(
        VGroup(
            Group(
                Item('prisma_img', label=u'根据参数prisma图片', show_label=False),
                Item('master_img', label=u'使用参数批量艺术图片', show_label=False),
                Item('force_iter_n', label=u'force iter n', show_label=False),
                # Item('select_img', label=u'其它文件夹中图片', show_label=False),
                orientation='horizontal',
                label=u'命令执行',
                show_border=True
            ),
            Group(
                Item(name='sample_enum', label=u'选择图片'),
                Item(name='guide_enum', label=u'选择图片'),
                Item(name='nbk_enum', label=u'特征放大层'),

                Group(

                    Item(name='enhance_enum', label=u'预处理增强'),
                    Item(name='enhance_p', label=u'预处理参数'),
                    orientation='horizontal',
                    show_border=False
                ),
                Item(name='mask_enum', label=u'图像mask函数'),
                show_border=True
            ),
            Group(
                Item(name='n1_convd', label=u'n1-小卷积核'),
                Item(name='n2_convd', label=u"n2-中卷积核"),
                Item(name='n3_convd', label=u"n3-大卷积核"),
                label=u'卷积核大小',
                show_border=True
            ),
            Group(
                Item(name='dd', label=u"是否取otsu内部"),

                Item(name='all_mask', label=u"只做浅层edges特征渲染"),
                Item(name='all_mask_rb_rate', label=u"浅层edges渲染概率"),

                Item(name='cb', label=u"执行clear_border"),
                Item(name='convd_median_factor', label=u"中卷积阀值因子"),
                Item(name='convd_big_factor', label=u"大卷积阀值因子"),
                Item(name='rb_rate', label=u"RGB随机渲染因子"),
                Item(name='stdmean_func_factor', label=u"stdmean mask因子"),
                Item(name='features_func_factor', label=u"features mask因子"),
                label=u'阀值因子',
                show_border=True
            ),
        ),
        Group(
            UItem('print_pool', editor=ListStrEditor(auto_add=False)),
            label=u'参数信息',
            show_border=True
        ),
    ))


if __name__ == "__main__":
    # import PrismaWorker
    # PrismaWorker.g_show_in_ipython = False
    PrismaControllClass().configure_traits()
