#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import argv
from os import system
import os.path


def crop(filename):
    dirname = os.path.dirname(filename) + '/'
    basename = os.path.splitext(os.path.basename(filename))[0]
    # ref: https://blog.awm.jp/2016/01/26/png/
    cmd = 'convert {}{}.png -crop 100x100+0+0 png24:{}{}_cropped.png'.format(
            dirname, basename, dirname, basename)
    print(cmd)
    system(cmd)


if __name__ == '__main__':
    for filename in argv[1:]:
        crop(filename)
