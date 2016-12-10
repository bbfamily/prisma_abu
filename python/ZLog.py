# -*- encoding:utf-8 -*-
import logging
import warnings

g_disable = False
g_only_debug = False


def exception(e, rs=False):
    logging.exception(e)
    if rs:
        """
        继续抛出异常给上层直到err.py
        """
        raise


def warning(text):
    global g_disable, g_only_debug
    if g_disable:
        return

    if g_only_debug:
        debug(text)
        return

    warnings.warn(text)
    logging.warning(text)


def debug(text):
    global g_disable
    if g_disable:
        return
    logging.debug(text)


def newline(fill_cnt=0):
    global g_disable
    if g_disable:
        return
    logging.info('\r')
    if fill_cnt > 0:
        logging.info('*' * fill_cnt)


def info(text):
    global g_disable, g_only_debug
    if g_disable:
        return

    if g_only_debug:
        debug(text)
        return

    logging.info(text)


def init_logging():
    # reload only for ipython work on
    # python3.x reload rm
    reload(logging)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='logging.log',
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
