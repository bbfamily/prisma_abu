# -*- encoding:utf-8 -*-
from __future__ import print_function

import functools
import time
import warnings


def warnings_filter(func):
    """
    warnings 有优化数据结构提示警告, 忽略
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.simplefilter('ignore')
        ret = func(*args, **kwargs)
        warnings.simplefilter('default')
        return ret

    return wrapper


def benchmark(func):
    """
    打印一个函数的执行时间
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t = time.clock()
        ret = func(*args, **kwargs)

        print('*' * 108 + '\n')
        print('*' * 16 + func.__name__ + ': ' + str(time.clock() - t))
        return ret

    return wrapper


def logging_ex(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print('*' * 38 + func.__name__ + " start" + '*' * 38)
        ret = func(*args, **kwargs)
        print('*' * 38 + func.__name__ + " end" + '*' * 38)
        return ret

    return wrapper


def logging(func):
    """
        记录函数日志
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        print('*' * 16 + func.__name__, args, kwargs)
        return ret

    return wrapper


def counter(func):
    """
        记录并打印一个函数的执行次数
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        ret = func(*args, **kwargs)
        print('*' * 16 + "{0} has been used: {1}x".format(func.__name__, wrapper.count))
        return ret

    wrapper.count = 0
    return wrapper
