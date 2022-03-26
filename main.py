#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2022/3/26 10:51
# @Author  : xiuweihan
"""
from hxwsolar.mountain import MountainIrradiance


if __name__ == '__main__':
    mi = MountainIrradiance(dem_file='./data/N30E121.hgt')
    mi()
