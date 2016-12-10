# -*- encoding:utf-8 -*-
from __future__ import division
import os
import PrismaHelper
import PrismaEnv

import ShowMsg
import PrismaWorker
from PrismaWorker import PrismaWorkerClass
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
# from ProcessMonitor import add_process_wrapper
from itertools import product
import ZCommonUtil
from Decorator import warnings_filter
from functools import partial

__author__ = 'BBFamily'

g_process_cnt = PrismaEnv.g_process_cnt  # 并行进程数
g_show_msg_tip = PrismaEnv.g_show_msg_tip  # 只为多进程notebook环境下无法输出日志，使用tip信息
g_skip_no_guide_rb = PrismaEnv.g_skip_no_guide_rb  # 是否忽略无引导图的ra_rate转换iter_n


@warnings_filter
# @add_process_wrapper
def do_prisma_img(parm_product, mask_func_str, n1, n2, n3, dd_factor, std_factor, loop_factor, convd_median_factor,
                  convd_big_factor, cb, all_mask,
                  save_dir):
    org_file = parm_product[0]
    gd_file = parm_product[1]
    nbk = parm_product[2]
    enhance = parm_product[3]
    rb_rate = parm_product[4]

    if org_file is None:
        raise RuntimeError('do_prisma_img org_file is None!')

    gd_file_name = 'none_gd' if gd_file is None else os.path.basename(gd_file).split('.')[0]
    enhance_name = '' if enhance is None else enhance
    rb_rate_name = str(rb_rate)
    nbk_name = nbk.replace('/', '_')
    """
        外层/输入/引导/enhance_nbk_rb_rate
    """

    # gd_file_name.split('_allmask')[0]mask的自己生成目录
    fn = '{}/{}/{}/_{}_{}_{}_%s.jpg'.format(save_dir, os.path.basename(org_file).split('.')[0],
                                            gd_file_name.split('_allmask')[0],
                                            enhance_name,
                                            nbk_name, rb_rate_name)

    ZCommonUtil.ensure_dir(fn)
    if ZCommonUtil.file_exist(fn):
        return

    if g_skip_no_guide_rb and gd_file is None and rb_rate <> 1:
        """
            忽略无引导图的rb_rate转换iter_n
        """
        return

    """
        初始化worker
    """
    pw = PrismaWorkerClass()
    """
        使用partial统一mask函数接口形式
    """
    mask_stdmean_func = partial(pw.do_stdmean, std_factor=std_factor)
    mask_features_func = partial(pw.do_features, loop_factor=loop_factor)
    mask_otsu_func = partial(pw.do_otsu, dd=dd_factor)

    if gd_file is None:
        """
            如果不需要特征渲染，这里的rb_rate被用来使用切换iter_n数量
        """
        iter_n = int((1 / rb_rate) * 10)
        ret_img = pw.cp.fit_img(org_file, nbk=nbk, iter_n=iter_n, enhance=enhance, resize=True, save=False)
        PrismaHelper.save_array_img(ret_img, fn % 'mask')
    else:
        if mask_func_str is None:
            for mask_name, mask_func in zip(['otsu_func', 'features_func', 'stdmean_func'],
                                            [mask_otsu_func, mask_features_func, mask_stdmean_func]):
                ret_img = pw.mix_mask_with_convd(mask_func, org_file, gd_file, nbk, enhance=enhance, rb_rate=rb_rate,
                                                 cb=cb, n1=n1, n2=n2, n3=n3, convd_median_factor=convd_median_factor,
                                                 convd_big_factor=convd_big_factor, all_mask=all_mask, save=False,
                                                 show=False)
                PrismaHelper.save_array_img(ret_img, fn % mask_name)
        else:
            if mask_func_str == 'otsu_func':
                mask_func = mask_otsu_func
            elif mask_func_str == 'features_func':
                mask_func = mask_features_func
            elif mask_func_str == 'stdmean_func':
                mask_func = mask_stdmean_func

            elif mask_func_str == 'otsu_func + stdmean_func':
                mask_func = partial(pw.together_mask_func, func_list=[mask_otsu_func,
                                                                      mask_stdmean_func])
            elif mask_func_str == 'otsu_func + features_func':
                mask_func = partial(pw.together_mask_func, func_list=[mask_otsu_func,
                                                                      mask_features_func])
            elif mask_func_str == 'stdmean_func + features_func':
                mask_func = partial(pw.together_mask_func, func_list=[mask_stdmean_func,
                                                                      mask_features_func])
            elif mask_func_str == 'otsu_func + stdmean_func + features_func':
                mask_func = partial(pw.together_mask_func, func_list=[mask_otsu_func,
                                                                      mask_stdmean_func,
                                                                      mask_features_func])
            else:
                raise ValueError('do_prisma_img mask_func_str MATCH ERROR!!')

            if not os.path.basename(gd_file).split('.')[0].endswith('_allmask') or nbk <> 'conv2/3x3_reduce':
                """
                    只有设置了all_mask，并且可以使用allmask的引导特征图才使用(文件结尾以_allmake)
                    and nbk == 'conv2/3x3_reduce 由于下面自行循环nbk所以只来一个
                """
                all_mask = False

            if all_mask is False:
                ret_img = pw.mix_mask_with_convd(mask_func, org_file, gd_file, nbk, enhance=enhance, rb_rate=rb_rate,
                                                 cb=cb, n1=n1, n2=n2, n3=n3, convd_median_factor=convd_median_factor,
                                                 convd_big_factor=convd_big_factor, all_mask=all_mask, save=False,
                                                 show=False)
                PrismaHelper.save_array_img(ret_img, fn % mask_func_str)
            else:
                tmp_mask_rb = PrismaWorker.g_all_mask_rb_rate
                for prb in [PrismaWorker.g_all_mask_rb_rate, 1.0]:
                    PrismaWorker.g_all_mask_rb_rate = prb
                    for mask_nbk in ['conv2/3x3_reduce', 'conv2/3x3', 'conv2/norm2', 'pool2/3x3_s2',
                                     'inception_3a/1x1',
                                     'inception_3a/5x5_reduce',
                                     'inception_3a/5x5', 'inception_3b/5x5_reduce', 'inception_3b/5x5']:
                        mask_nbk_name = mask_nbk.replace('/', '_')
                        mask_fn = '{}/{}/{}/_{}_{}_{}_%s.jpg'.format(save_dir, os.path.basename(org_file).split('.')[0],
                                                                     gd_file_name,
                                                                     enhance_name,
                                                                     mask_nbk_name, prb)
                        ZCommonUtil.ensure_dir(mask_fn)

                        ret_img = pw.mix_mask_with_convd(mask_func, org_file, gd_file, mask_nbk, enhance=enhance,
                                                         rb_rate=rb_rate,
                                                         cb=cb, n1=n1, n2=n2, n3=n3,
                                                         convd_median_factor=convd_median_factor,
                                                         convd_big_factor=convd_big_factor, all_mask=all_mask, save=False,
                                                         show=False)

                        PrismaHelper.save_array_img(ret_img, mask_fn % 'allmask_func')
                PrismaWorker.g_all_mask_rb_rate = tmp_mask_rb

                """
                    如果有all_mask，下面才开始正常工作
                """
                ret_img = pw.mix_mask_with_convd(mask_func, org_file, gd_file, nbk, enhance=enhance, rb_rate=rb_rate,
                                                 cb=cb, n1=n1, n2=n2, n3=n3, convd_median_factor=convd_median_factor,
                                                 convd_big_factor=convd_big_factor, all_mask=False, save=False,
                                                 show=False)
                PrismaHelper.save_array_img(ret_img, fn % mask_func_str)

    if g_show_msg_tip:
        ShowMsg.show_msg('pid %s' % os.getpid(), os.path.basename(fn))


def product_prisma(org_file_list, gd_file_list, nbk_list, enhance_list, rb_rate_list, mask_func_str, n1, n2, n3,
                   dd_factor, std_factor, loop_factor, convd_median_factor, convd_big_factor, cb, all_mask, save_dir):
    if not isinstance(org_file_list, list) or not isinstance(gd_file_list, list):
        raise TypeError('img_path or gd_path must list for mul process handle!')

    parallel = Parallel(
        n_jobs=g_process_cnt, verbose=0, pre_dispatch='2*n_jobs')

    parallel(
        delayed(do_prisma_img)(parm_product, mask_func_str, n1, n2, n3, dd_factor, std_factor, loop_factor,
                               convd_median_factor,
                               convd_big_factor,
                               cb, all_mask, save_dir)
        for parm_product in product(org_file_list, gd_file_list, nbk_list, enhance_list, rb_rate_list))
