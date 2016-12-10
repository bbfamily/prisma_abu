# -*- encoding:utf-8 -*-
import os


def show_msg(title, msg):
    msg_cmd = 'osascript -e \'display notification "%s" with title "%s"\'' % (msg, title)
    os.system(msg_cmd)


if __name__ == '__main__':
    show_msg("title", "popup info")
