# -*- encoding:utf-8 -*-
from __future__ import print_function

import os


def ensure_dir(a_dir):
    # ZEnv.print_Str("ensureDir:" + aDir)
    a_dir = os.path.dirname(a_dir)
    if not os.path.exists(a_dir):
        os.makedirs(a_dir)
        print("makedirs " + a_dir)


def file_exist(a_file):
    if os.path.exists(a_file):
        return True
    return False

