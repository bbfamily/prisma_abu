# -*- encoding:utf-8 -*-
"""
    deep dream封装的类，加入图像预处理
    等元素，批量处理等
"""
from __future__ import division
import os
from functools import partial

import PrismaEnv
import ZCommonUtil
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format
import caffe
import PrismaHelper
from PrismaBase import BasePrismaClass

__author__ = 'BBFamily'


class CaffePrismaClass(BasePrismaClass):
    def __init__(self, dog_mode=False):
        # net_fn = '../mode/deploy.prototxt'
        # self.net_fn = '../mode/deploy_{}.prototxt'.format(os.getpid())
        # ZCommonUtil.move_fileto(net_fn, self.net_fn)
        self.net_fn = '../mode/deploy.prototxt'
        if not dog_mode:
            # org_fn = '../mode/bvlc_googlenet.caffemodel'
            # self.param_fn = '../mode/bvlc_googlenet_{}.caffemodel'.format(os.getpid())
            # ZCommonUtil.move_fileto(org_fn, self.param_fn)
            self.param_fn = '../mode/bvlc_googlenet.caffemodel'
            mu = np.float32([104.0, 117.0, 123.0])
        else:
            self.param_fn = '../mode/dog_judge_train_iter_5000.caffemodel'
            model_mean_file = '../mode/mean.binaryproto'
            mean_blob = caffe.proto.caffe_pb2.BlobProto()
            mean_blob.ParseFromString(open(model_mean_file, 'rb').read())
            mean_npy = caffe.io.blobproto_to_array(mean_blob)
            mu = np.float32(mean_npy.mean(2).mean(2)[0])

        model = caffe.io.caffe_pb2.NetParameter()
        text_format.Merge(open(self.net_fn).read(), model)
        model.force_backward = True

        # self.t_fn ='tmp_{}.prototxt'.format(os.getpid())
        self.t_fn = '../mode/tmp.prototxt'
        open(self.t_fn, 'w').write(str(model))
        self.net = caffe.Classifier(self.t_fn, self.param_fn,
                                    mean=mu,
                                    channel_swap=(
                                        2, 1, 0))

        self.mean_pixel = self.net.transformer.mean['data']

    def __del__(self):
        os.remove(self.t_fn)

    def __str__(self):
        return 'net_fn :{0} param_fn: {1} mean_pixel: {2}'.format(self.net_fn, self.param_fn, self.mean_pixel)

    __repr__ = __str__

    def _objective_l2(self, dst):
        dst.diff[:] = dst.data

    def _objective_guide_features(self, dst, guide_features):
        x = dst.data[0].copy()
        y = guide_features
        ch = x.shape[0]
        x = x.reshape(ch, -1)
        y = y.reshape(ch, -1)
        a = x.T.dot(y)
        dst.diff[0].reshape(ch, -1)[:] = y[:, a.argmax(1)]

    def do_prisma_step(self, step_size=1.5, end='inception_4c/output',
                       jitter=32, objective=None):
        if objective is None:
            raise ValueError('make_step objective is None!!!')

        src = self.net.blobs['data']
        dst = self.net.blobs[end]
        ox, oy = np.random.randint(-jitter, jitter + 1, 2)
        src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2)

        self.net.forward(end=end)
        objective(dst)
        self.net.backward(start=end)

        g = src.diff[0]
        src.data[:] += step_size / np.abs(g).mean() * g
        src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)

    def do_prisma(self, base_img, iter_n=10, octave_n=4, octave_scale=1.4,
                  end='inception_4c/output', **step_params):
        octaves = [PrismaHelper.preprocess_with_roll(base_img, self.mean_pixel)]
        for i in xrange(octave_n - 1):
            octaves.append(nd.zoom(octaves[-1], (1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))
        src = self.net.blobs['data']
        detail = np.zeros_like(octaves[-1])
        for octave, octave_base in enumerate(octaves[::-1]):
            h, w = octave_base.shape[-2:]
            if octave > 0:
                h1, w1 = detail.shape[-2:]
                detail = nd.zoom(detail, (1, 1.0 * h / h1, 1.0 * w / w1), order=1)
            src.reshape(1, 3, h, w)
            src.data[0] = octave_base + detail
            for i in xrange(iter_n):
                self.do_prisma_step(end=end, **step_params)
            detail = src.data[0] - octave_base
        return PrismaHelper.deprocess_with_stack(src.data[0], self.mean_pixel)

    def gd_features_make(self, guide, end):
        """
        引导图风格make
        :param guide:
        :param end:
        :return:
        """
        h, w = guide.shape[:2]
        src, dst = self.net.blobs['data'], self.net.blobs[end]
        src.reshape(1, 3, h, w)
        src.data[0] = PrismaHelper.preprocess_with_roll(guide, self.mean_pixel)
        self.net.forward(end=end)
        guide_features = dst.data[0].copy()
        return guide_features

    def fit_guide_img(self, img_path, gd_path, resize=False, size=480, enhance=None, iter_n=10,
                      nbk='inception_4c/output', img_np=None):
        """
        风格引导的风格画处理
        :param img_path: 输入图像路径
        :param gd_path: 引导图像路径
        :param resize: 是否resize
        :param size: resize大小
        :param enhance: 预处理指令
        :param iter_n: iter
        :param nbk: cnn层指令
        :param img_np: 输入图像numpy对象
        :return:
        """
        guide = np.float32(PIL.Image.open(gd_path))
        guide_features = self.gd_features_make(guide, nbk)
        guide_func = partial(self._objective_guide_features, guide_features=guide_features)

        if img_np is None:
            r_img = PIL.Image.open(img_path)
            if resize:
                r_img = self.resize_img(r_img, size)
            org_img = self.handle_enhance(r_img, enhance)

            e_str = '' if enhance is None else '_' + enhance.lower()
            save_path = os.path.dirname(img_path) + '/batch_caffe/' + e_str + os.path.basename(gd_path).split('.')[0]
            ZCommonUtil.ensure_dir(save_path)
            org_img_path = save_path + 'org.jpeg'
            with open(org_img_path, 'w') as f:
                org_img.save(f, 'jpeg')

            org_img = np.float32(org_img)
        else:
            org_img = img_np
            save_path = '../sample/batch_caffe/'

        fn = save_path + nbk.replace('/', '_') + '.jpg'
        ret_img = self.do_prisma(org_img, iter_n=iter_n, end=nbk, objective=guide_func)
        PrismaHelper.save_array_img(ret_img, fn)
        return ret_img

    def fit_img(self, img_path=None, resize=False, size=480, enhance=None, iter_n=10, nbk='inception_4c/output',
                img_np=None, save=True):
        """
        风格化图像
        :param img_path: 输入图像路径
        :param resize: 是否resize
        :param size: resize大小
        :param enhance: 预处理指令
        :param iter_n: iter
        :param nbk:  cnn层指令
        :param img_np: 输入图像numpy对象
        :param save: 是否保存输出
        :return:
        """
        if img_np is None:
            r_img = PIL.Image.open(img_path)
            if resize:
                r_img = self.resize_img(r_img, size)

            org_img = self.handle_enhance(r_img, enhance)

            e_str = '' if enhance is None else '_' + enhance.lower()
            save_path = os.path.dirname(img_path) + '/batch_caffe/' + e_str
            ZCommonUtil.ensure_dir(save_path)
            org_img_path = save_path + 'org.jpeg'
            with open(org_img_path, 'w') as f:
                org_img.save(f, 'jpeg')

            org_img = np.float32(org_img)
        else:
            org_img = img_np
            if enhance is not None:
                org_img = PIL.Image.fromarray(org_img)
                org_img = self.handle_enhance(org_img, enhance)
                org_img = np.float32(org_img)
            save_path = '../sample/batch_caffe/'

        if PrismaEnv.g_force_iter_n <> -1:
            iter_n = PrismaEnv.g_force_iter_n
        ret_img = self.do_prisma(org_img, iter_n=iter_n, end=nbk, objective=self._objective_l2)
        if save:
            """
                可以选择在外部保存
            """
            fn = save_path + nbk.replace('/', '_') + '.jpg'
            PrismaHelper.save_array_img(ret_img, fn)
        return ret_img

    def fit_batch_img(self, img_path, resize=False, size=480, enhance=None):
        """
        批量处理，但不支持并行，后修改为类似tensor primsma中的并行模式
        :param img_path:
        :param resize:
        :param size:
        :param enhance:
        :return:
        """
        r_img = PIL.Image.open(img_path)
        if resize:
            r_img = self.resize_img(r_img, size)

        org_img = self.handle_enhance(r_img, enhance)

        e_str = '' if enhance is None else '_' + enhance.lower()
        save_path = os.path.dirname(img_path) + '/batch_caffe/' + e_str
        ZCommonUtil.ensure_dir(save_path)
        org_img_path = save_path + 'org.jpeg'
        with open(org_img_path, 'w') as f:
            org_img.save(f, 'jpeg')

        org_img = np.float32(org_img)
        start = 1
        end = self.net.blobs.keys().index('inception_4c/pool')
        nbks = self.net.blobs.keys()[start:end]

        """
            不能使用多进程方式在这里并行执行，因为caffe.classifier.Classifier不支持序列化
            Pickling of "caffe.classifier.Classifier" instances is not enabled
            so mul process no pass
        """
        for nbk in nbks:
            if nbk[-8:-1] == '_split_':
                continue
            fn = save_path + nbk.replace('/', '_') + '.jpg'
            ret_img = self.do_prisma(org_img, iter_n=10, end=nbk)
            PrismaHelper.save_array_img(ret_img, fn)
        return save_path
