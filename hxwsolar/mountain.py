#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2022/3/25 19:34
# @Author  : xiuweihan
"""
import numpy as np
from osgeo import gdal
from datetime import datetime, timedelta

from hxwsolar.irradiance import Irradiance
from hxwsolar.utils import local_datetime_array


class MountainIrradiance(object):
    """计算山地太阳能辐射"""

    def __init__(self, precision=30, dem_file='../data/N30E121.hgt'):
        super().__init__()
        self.precision = precision
        self.dem_file = dem_file

    def _calculate_xy(self, array):
        """
        计算dx和dy
        :param array:
        :return:
        """
        array = np.pad(array, 1, 'edge')  # DEM扩充, 避免边缘值损失
        sx = (array[1:-1, :-2] - array[1:-1, 2:]) / (2 * self.precision)  # WE方向
        sy = (array[2:, 1:-1] - array[:-2, 1:-1]) / (2 * self.precision)  # NS方向
        return sx, sy

    def _slope(self, array):
        """
        计算坡度--利用矩阵
        :param array:
        :return:
        """
        sx, sy = self._calculate_xy(array)
        slope = np.arctan(np.sqrt(sx ** 2 + sy ** 2)) * (180 / np.pi)
        return slope

    def _aspect(self, array):
        """
        计算坡向--利用循环
        :param array:
        :return:
        """
        _sx, _sy = self._calculate_xy(array)
        aspect = np.ones([_sx.shape[0], _sy.shape[1]]).astype(np.float32)
        for i in range(_sx.shape[0]):
            for j in range(_sy.shape[1]):
                sx = float(_sx[i, j])
                sy = float(_sy[i, j])
                if (sx == 0.0) & (sy == 0.0):
                    aspect[i, j] = -1
                elif sx == 0.0:
                    if sy > 0.0:
                        aspect[i, j] = 0.0
                    else:
                        aspect[i, j] = 180.0
                elif sy == 0.0:
                    if sx > 0.0:
                        aspect[i, j] = 90.0
                    else:
                        aspect[i, j] = 270.0
                else:
                    aspect[i, j] = float(np.arctan2(sy, sx) * (180 / np.pi))
                    if aspect[i, j] < 0.0:
                        aspect[i, j] = 90.0 - aspect[i, j]
                    elif aspect[i, j] > 90.0:
                        aspect[i, j] = 360.0 - aspect[i, j] + 90.0
                    else:
                        aspect[i, j] = 90.0 - aspect[i, j]
        return aspect

    def __call__(self, *args, **kwargs):
        # 读取
        dataset = gdal.Open(self.dem_file)
        im_width = dataset.RasterXSize
        im_height = dataset.RasterYSize
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

        # 文件说明
        im_geotrans = dataset.GetGeoTransform()
        origin_x, origin_y = im_geotrans[0], im_geotrans[3]
        pixel_width, pixel_height = im_geotrans[1], im_geotrans[5]

        # 筛选部分
        # 1.经纬度
        new_data = im_data[100:200, 100:200]
        lng_list = [origin_x+pixel_width*i for i in range(100, 200)]
        lat_list = [origin_y+pixel_height*i for i in range(100, 200)]
        lat_grid, lng_grid = np.meshgrid(lat_list, lng_list)
        # 2.坡度坡向
        slope_grid = self._slope(new_data)
        aspect_grid = self._aspect(new_data)

        # 时间
        dt = datetime(2007, 12, 23, 15, 12, 0)
        # 计算-单时间多地点
        irr = Irradiance(lat_grid, lng_grid, dt, slope_grid, aspect_grid)
        print(irr())
        print('-'*100)
        # 计算-多时间多地点
        s_dt = datetime(2007, 12, 23, 13, 0, 0)
        e_dt = datetime(2007, 12, 23, 14, 0, 0)
        time_step = timedelta(minutes=10)
        dt = local_datetime_array(s_dt, e_dt, time_step)
        irr = Irradiance(lat_grid, lng_grid, dt, slope_grid, aspect_grid)
        print(irr())


if __name__ == '__main__':
    mi = MountainIrradiance()
    mi()
