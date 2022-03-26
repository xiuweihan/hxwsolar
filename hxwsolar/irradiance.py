#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2022/3/25 16:47
# @Author  : xiuweihan
"""
import numpy as np
from datetime import datetime
from functools import cached_property
from numpy.core.umath import radians, cos, tan

from hxwsolar.solar import Solar
from hxwsolar.type_ import CONSTANTS


class Irradiance(Solar):
    """
    计算太阳能辐射
    """
    def __init__(self, lat, lng, dt,
                 slope=None, aspect=None,
                 air_mass_method='kasten-young',
                 pollution_condition='average'):
        super(Irradiance, self).__init__(lat, lng, dt)

        slope, aspect, vec_slope, slope_shape = self.__validate_slope(slope, aspect, dt)

        self.__slope = slope              # 坡度
        self.__aspect = aspect            # 坡向
        self.__vec_slope = vec_slope      # 坡度是否为向量
        self.__slope_shape = slope_shape  # 坡向是否为向量
        self.__air_mass_method = air_mass_method          # 气团计算方式
        self.__pollution_condition = pollution_condition  # 气团修正方式

    @staticmethod
    def __validate_slope(slope, aspect, dt):
        if isinstance(slope, np.ndarray) and isinstance(aspect, np.ndarray):
            vec_slope = True
            if vec_slope:
                assert len(slope.shape) == 2
            if slope.shape != aspect.shape:
                raise AttributeError("坡度坡向维度不匹配！")
            # 如果是向量，则给空间需要添加第三个维度，即时间
            if isinstance(dt, list) and vec_slope:
                slope = np.repeat(slope[:, :, np.newaxis], len(dt), axis=2)
                aspect = np.repeat(aspect[:, :, np.newaxis], len(dt), axis=2)
            slope_shape = slope.shape or aspect.shape
        else:
            vec_slope = False
            slope_shape = tuple()
        return slope, aspect, vec_slope, slope_shape

    @cached_property
    def sun_rad_vector(self):
        """到地面的辐射矢量（单位为AU）"""
        ec = self.eccentricity_earth_orbit
        ta = radians(self.sun_true_anomaly)

        rad_vector = (1.000001018 * (1 - ec ** 2)) / (1 + ec * np.cos(ta))
        return rad_vector

    @cached_property
    def atmo_refraction(self):
        """大气折射"""
        zenith = self.solar_zenith
        if isinstance(zenith, np.ndarray):
            e = 90 - zenith
            ar = np.array(e * 0)

            ar[e > 85] = 0
            ar[(e > 5) & (e <= 85)] = 58.1 / tan(radians(e[(e > 5) & (e <= 85)])) - 0.07 / tan(
                radians(e[(e > 5) & (e <= 85)])) ** 3 + 0.000086 / tan(
                radians(e[(e > 5) & (e <= 85)])) ** 5
            ar[(e > -0.575) & (e <= 5)] = 1735 + e[(e > -0.575) & (e <= 5)] * (
                    103.4 + e[(e > -0.575) & (e <= 5)] * (-12.79 + e[(e > -0.575) & (e <= 5)] * 0.711))
            ar[e <= -0.575] = -20.772 / tan(radians(e[e <= -0.575]))
            return ar / 3600
        else:
            e = 90 - zenith
            er = radians(e)

            if e > 85:
                ar = 0
            elif e > 5:
                ar = 58.1 / tan(er) - 0.07 / tan(er) ** 3 + 0.000086 / tan(er) ** 5
            elif e > -0.575:
                ar = 1735 + e * (103.4 + e * (-12.79 + e * 0.711))
            else:
                ar = -20.772 / tan(er)
            return ar / 3600

    @cached_property
    def air_mass(self):
        """
        气团计算
        https://www.osapublishing.org/ao/abstract.cfm?uri=ao-28-22-4735
        https://en.wikipedia.org/wiki/Air_mass_(solar_energy)
        :return:
        """
        zenith = self.solar_zenith
        if self.__air_mass_method is None:
            self.__air_mass_method = "kasten-young"

        if self.__air_mass_method == "spherical":
            z_r = radians(zenith)
            r = CONSTANTS.earth_radius / CONSTANTS.atm_height
            return (r * (cos(z_r) ** 2) + (2 * r) + 1) ** 0.5 - (r * cos(z_r))

        elif self.__air_mass_method == "kasten-young":
            z_r = radians(zenith)
            return 1.0 / (cos(z_r) + 0.50572 * (96.07995 - zenith) ** (-1.6364))

    @cached_property
    def earth_distance(self):
        """
        地球到太阳的距离（以米为单位）
        :return:
        """
        earth_distance = self.sun_rad_vector * 149597870700

        return earth_distance

    @cached_property
    def sun_norm_toa_irradiance(self):
        """
        入射到垂直于太阳的表面的太阳能(W/m^2)
        :return:
        """
        ed = self.earth_distance

        sun_surf_rad = CONSTANTS.sun_surf_rad
        sun_radius = CONSTANTS.sun_radius

        # 计算在地球距离法向表面的辐照度
        norm_irradiance = sun_surf_rad * (sun_radius / ed) ** 2

        return norm_irradiance

    def air_mass_correction(self, _toa_irradiance, air_mass):
        """
        气团修正太阳辐射
        :param _toa_irradiance:
        :param air_mass:
        :return:
        """
        if self.__pollution_condition is None:
            self.__pollution_condition = "average"

        if self.__pollution_condition == "clean":
            base = 0.76
            exp = 0.618
        elif self.__pollution_condition == "average":
            base = 0.7
            exp = 0.678
        elif self.__pollution_condition == "dirty":
            base = 0.56
            exp = 0.715
        else:
            raise ValueError(f"'pollution_condition参数必须为[clean, dirty, average]")

        return 1.1 * _toa_irradiance * (base ** (air_mass ** exp))

    @cached_property
    def sun_norm_boa_irradiance(self):
        """
        根据气团参数修正太阳辐射强度
        https://en.wikipedia.org/wiki/Air_mass_(solar_energy)#cite_note-interpolation-17
        :return:
        """
        _toa_irradiance = self.sun_norm_toa_irradiance
        air_mass = self.air_mass
        return self.air_mass_correction(_toa_irradiance, air_mass)

    @staticmethod
    def _polar_to_cartesian(phi, theta):
        """
        极坐标转换为笛卡尔坐标
        :param phi:
        :param theta:
        :return:
        """
        theta_r = radians(theta)
        phi_r = radians(phi)

        y = np.sin(theta_r) * np.cos(phi_r)
        x = np.sin(theta_r) * np.sin(phi_r)
        z = np.cos(theta_r)
        return np.stack([x, y, z], axis=-1)

    @cached_property
    def sun_vec_cartesian(self):
        """
        返回从地球表面直接指向太阳的向量
        :return:
        """
        return self._polar_to_cartesian(self.solar_azimuth, 90.0 - self.solar_elevation)

    @cached_property
    def surf_vec_cartesian(self):
        """
        计算表面法线向量，仅在给定坡度和坡向时使用
        :return:
        """
        if self.__slope is None and self.__aspect is None:
            return None
        else:
            return self._polar_to_cartesian(self.__aspect, self.__slope)

    @cached_property
    def sun_surf_angle(self):
        """
        太阳入射角，即阳光法向量和平面法向量夹角
        :return:
        """
        sun_vec = self.sun_vec_cartesian
        surface_vec = self.surf_vec_cartesian
        if surface_vec is None:
            return 0
        dims = sun_vec.shape
        out_dims = dims[:-1]
        flat_len = np.prod(out_dims)

        # 如果少于两个dim，则不需要einsum操作
        if len(sun_vec.shape) < 2:
            cosines = np.dot(sun_vec, surface_vec)
            angles = np.arccos(np.clip(cosines, -1.0, 1.0))
            return np.degrees(angles)

        else:
            if len(sun_vec.shape) == 2:
                v1 = sun_vec
                v2 = surface_vec
            else:
                v1 = sun_vec.reshape((flat_len, 3))
                v2 = surface_vec.reshape((flat_len, 3))

            cosines = np.einsum("ij,ij->i", v1.reshape(-1, 3), v2.reshape(-1, 3))
            angles = np.arccos(np.clip(cosines, -1.0, 1.0))
            return np.degrees(angles).reshape(out_dims)

    @cached_property
    def surf_norm_toa_irradiance(self):
        """
        根据太阳入射角矫正后的太阳辐照量
        :return:
        """
        sun_surf_angle = self.sun_surf_angle
        sun_norm_irradiance = self.sun_norm_toa_irradiance
        ssa_r = radians(sun_surf_angle)
        srf = cos(ssa_r) * sun_norm_irradiance

        # 将负值设为零。
        if isinstance(srf, np.ndarray) and srf.shape:
            srf[srf <= 0] = 0
            return srf
        else:
            return max([srf, 0.0])

    @property
    def surf_norm_boa_irradiance(self):
        """
        根据大气影响矫正的太阳辐射量
        https://en.wikipedia.org/wiki/Air_mass_(solar_energy)#cite_note-interpolation-17
        :return:
        """
        _toa_irradiance = self.surf_norm_toa_irradiance
        air_mass = self.air_mass
        return self.air_mass_correction(_toa_irradiance, air_mass)

    def surf_norm_toa_par(self):
        """
        光合成有效辐射
        :return:
        """
        return self.surf_norm_toa_irradiance * CONSTANTS.etta_par

    def __call__(self, *args, **kwargs):
        compute_list = [f for f in dir(self)
                        if not callable(getattr(self, f))
                        and not f.startswith("__")
                        and f != "summarize"]
        for f in compute_list:
            result = getattr(self, f)
            if isinstance(result, np.ndarray):
                result_mean = result.mean()
                print(f"{f:37s} {str(result.shape):20s} {result_mean}")
            else:
                dim = len(result) if isinstance(result, list) else 1
                print(f"{f:37s} {str(dim):20s} {result}")


if __name__ == '__main__':
    irradiance = Irradiance(lat=31, lng=121,
                            dt=datetime(2007, 12, 23, 15, 12, 0),
                            slope=None, aspect=None)
    print(irradiance())
