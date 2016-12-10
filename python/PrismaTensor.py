# -*- encoding:utf-8 -*-
"""
    tensor封装的风格画类
"""
from __future__ import division
import os
import tensorflow as tf
import scipy.io
import ZCommonUtil
import numpy as np
import PIL.Image
import PrismaHelper
import ZLog
from operator import mul
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from PrismaBase import BasePrismaClass
from itertools import product

"""
from ProcessMonitor import add_process_wrapper
代码地址：https://github.com/bbfamily/monitor_parallel

如需要控制多进程可以打开, 并且把do_fit_parallel_img上的add_process_wrapper装饰器的注释删除
"""
# from ProcessMonitor import add_process_wrapper

__author__ = 'BBFamily'

K_VGG_MAT_PATH = '../mode/vgg_imagenet.mat'
K_ORG_LAYER = 'relu4_2'
K_GUIDE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
K_NET_LAYER = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)

K_LEARNING_RATE = 1e2
K_TV_WEIGHT = 1e2
K_ORG_WEIGHT = 5e0
K_GUIDE_WEIGHT = 1e2

K_PRINT_ITER = 50
K_CKP_ITER = 500
K_CKP_FN_FMT = '%s_%d.jpeg'

g_doing_parallel = False


# @add_process_wrapper
def do_fit_parallel_img(path_product, resize, size, enhance, iter_n):
    global g_doing_parallel

    """
        要在每个进程设置模块全局变量
    """
    g_doing_parallel = True

    img_path = path_product[0]
    gd_path = path_product[1]
    prisma_img = TensorPrismaClass().fit_guide_img(img_path, gd_path, resize=resize, size=size, enhance=enhance,
                                                   iter_n=iter_n)

    g_doing_parallel = False
    return path_product, prisma_img


def fit_parallel_img(img_path, gd_path, resize=False, size=480, enhance=None, iter_n=800, n_jobs=-1):
    if not isinstance(img_path, list) or not isinstance(gd_path, list):
        raise TypeError('img_path or gd_path must list for mul process handle!')

    parallel = Parallel(
        n_jobs=n_jobs, verbose=0, pre_dispatch='2*n_jobs')

    out = parallel(delayed(do_fit_parallel_img)(path_product, resize, size, enhance, iter_n) for path_product in
                   product(img_path, gd_path))

    return out


class TensorPrismaClass(BasePrismaClass):
    def _conv2d(self, img, w, b):
        """
        卷积层
        :param img:
        :param w:
        :param b:
        :return:
        """
        return tf.nn.bias_add(tf.nn.conv2d(img, tf.constant(w), strides=[1, 1, 1, 1], padding='SAME'), b)

    def _max_pool(self, img, k):
        """
        池化层
        :param img:
        :param k:
        :return:
        """
        return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def __init__(self):
        self.net_fn = K_VGG_MAT_PATH

        if not ZCommonUtil.file_exist(self.net_fn):
            raise RuntimeError('self.net_fn not exist!!!')

        self.net_layers = K_NET_LAYER
        self.net_data = scipy.io.loadmat(self.net_fn)
        self.mean = self.net_data['normalization'][0][0][0]
        self.mean_pixel = np.mean(self.mean, axis=(0, 1))
        self.weights = self.net_data['layers'][0]

    def __str__(self):
        return 'mean_pixel: {0} '.format(self.mean_pixel)

    __repr__ = __str__

    def _build_vgg_net(self, shape, image_tf=None):
        """
        通过vgg模型构造网络
        :param shape:
        :param image_tf:
        :return:
        """
        if image_tf is None:
            image_tf = tf.placeholder('float', shape=shape)
        net = dict()
        current = image_tf
        for ind, name in enumerate(self.net_layers):
            kind = name[:4]
            if kind == 'conv':
                kernels, bias = self.weights[ind][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                current = self._conv2d(current, kernels, bias)
            elif kind == 'relu':
                current = tf.nn.relu(current)
            elif kind == 'pool':
                current = self._max_pool(current, 2)
            net[name] = current
        return net, image_tf

    def _features_make(self, img, image_tf, net, features, guide):
        """
        特征矩阵组成，引导图只使用浅层特征K_GUIDE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
        原始图像需要使用深层K_ORG_LAYER = 'relu4_2'
        :param img:
        :param image_tf:
        :param net:
        :param features:
        :param guide:
        :return:
        """
        preprocess = np.array([img - self.mean_pixel])
        if guide:
            for gl in K_GUIDE_LAYERS:
                fs = net[gl].eval(feed_dict={image_tf: preprocess})
                fs = np.reshape(fs, (-1, fs.shape[3]))
                features[gl] = np.matmul(fs.T, fs) / fs.size
        else:
            features[K_ORG_LAYER] = net[K_ORG_LAYER].eval(feed_dict={image_tf: preprocess})

    def _tensor_size(self, tensor):
        return reduce(mul, (d.value for d in tensor.get_shape()), 1)

    def gd_features_make(self, org_img, guide_img):
        # noinspection PyUnusedLocal
        with tf.Graph().as_default(), tf.Session() as sess:
            org_shape = (1,) + org_img.shape
            org_net, org_img_tf = self._build_vgg_net(org_shape)
            org_features = dict()
            self._features_make(org_img, org_img_tf, org_net, org_features, False)

            guide_shapes = (1,) + guide_img.shape
            guide_net, guide_img_tf = self._build_vgg_net(guide_shapes)
            guide_features = dict()
            self._features_make(guide_img, guide_img_tf, guide_net, guide_features, True)
        return org_features, guide_features

    def do_prisma(self, org_img, guide_img, ckp_fn, iter_n):
        org_shape = (1,) + org_img.shape
        org_features, guide_features = self.gd_features_make(org_img, guide_img)

        with tf.Graph().as_default():
            # out_v = tf.zeros(org_shape, dtype=tf.float32, name=None)
            out_v = tf.random_normal(org_shape) * 0.256
            out_img = tf.Variable(out_v)
            out_net, _ = self._build_vgg_net(org_shape, out_img)

            org_loss = K_ORG_WEIGHT * (2 * tf.nn.l2_loss(
                out_net[K_ORG_LAYER] - org_features[K_ORG_LAYER]) /
                                       org_features[K_ORG_LAYER].size)

            style_loss = 0
            for guide_layer in K_GUIDE_LAYERS:
                layer = out_net[guide_layer]
                _, height, width, number = map(lambda x: x.value, layer.get_shape())
                size = height * width * number
                feats = tf.reshape(layer, (-1, number))
                gram = tf.matmul(tf.transpose(feats), feats) / size
                style_gram = guide_features[guide_layer]
                style_loss += K_GUIDE_WEIGHT * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size

            tv_y_size = self._tensor_size(out_img[:, 1:, :, :])
            tv_x_size = self._tensor_size(out_img[:, :, 1:, :])
            tv_loss = K_TV_WEIGHT * 2 * (
                (tf.nn.l2_loss(out_img[:, 1:, :, :] - out_img[:, :org_shape[1] - 1, :, :]) /
                 tv_y_size) +
                (tf.nn.l2_loss(out_img[:, :, 1:, :] - out_img[:, :, :org_shape[2] - 1, :]) /
                 tv_x_size))

            """
                 loss function的组成由org_loss + style_loss + tv_loss
             """
            loss = org_loss + style_loss + tv_loss

            train_step = tf.train.AdamOptimizer(K_LEARNING_RATE).minimize(loss)

            # noinspection PyUnresolvedReferences
            def print_progress(ind, last=False):
                if last or (ind > 0 and ind % K_PRINT_ITER == 0):
                    ZLog.info('Iteration %d/%d\n' % (ind + 1, iter_n))
                    ZLog.debug('  content loss: %g\n' % org_loss.eval())
                    ZLog.debug('    style loss: %g\n' % style_loss.eval())
                    ZLog.debug('       tv loss: %g\n' % tv_loss.eval())
                    ZLog.debug('    total loss: %g\n' % loss.eval())

            best_loss = float('inf')
            best = None
            with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())
                for i in range(iter_n):
                    last_step = (i == iter_n - 1)
                    if not g_doing_parallel:
                        print_progress(i, last=last_step)
                    train_step.run()
                    """
                        ckp或者最后一次迭代将图像保存
                    """
                    if (i > 0 and i % K_CKP_ITER == 0) or last_step:
                        # noinspection PyUnresolvedReferences
                        this_loss = loss.eval()
                        if this_loss < best_loss:
                            best_loss = this_loss
                            best = out_img.eval()
                        if not last_step:
                            ckp_fn_iter = K_CKP_FN_FMT % (ckp_fn, i)
                            PrismaHelper.save_array_img(best.reshape(org_shape[1:]) + self.mean_pixel, ckp_fn_iter)
                return best.reshape(org_shape[1:]) + self.mean_pixel

    def fit_guide_img(self, img_path, gd_path, resize=True, size=480, enhance=None, iter_n=800, **kwargs):
        """
        :param img_path: 输入图像路径
        :param gd_path: 引导图像路径
        :param resize: 是否resize
        :param size: resize大小
        :param enhance: 预处理指令
        :param iter_n: 预迭代次
        :param kwargs:
        :return:
        """
        guide = PIL.Image.open(gd_path)
        r_img = PIL.Image.open(img_path)
        if resize:
            r_img = self.resize_img(r_img, size)
            guide = self.resize_img(guide, size)
        org_img = self.handle_enhance(r_img, enhance)

        e_str = '' if enhance is None else '_' + enhance.lower()

        save_path = os.path.dirname(img_path) + '/batch_tensor/' + e_str + os.path.basename(gd_path).split('.')[0]
        ZCommonUtil.ensure_dir(save_path)

        org_img = np.float32(org_img)
        guide_img = np.float32(guide)

        ckp_fn = save_path + os.path.basename(img_path).split('.')[0]
        deep_img = self.do_prisma(org_img, guide_img, ckp_fn, iter_n=iter_n)
        last_fn = save_path + os.path.basename(img_path)
        PrismaHelper.save_array_img(deep_img, last_fn)
        return deep_img

    def fit_img(self, img_path, resize=False, size=480, enhance=None, iter_n=10, **kwargs):
        ZLog.info('TensorPrismaClass miss fit_img!!')
