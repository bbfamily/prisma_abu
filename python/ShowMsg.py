# -*- encoding:utf-8 -*-
import platform


def show_msg(title, msg):
    if not platform.system().lower().find("windows") >= 0:
        import ShowMsgMac
        ShowMsgMac.show_msg(title, msg)
    else:
        import ShowMsgWin
        ShowMsgWin.show_msg(title, msg)


if __name__ == '__main__':
    show_msg("title", "popup info")
