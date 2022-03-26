#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2022/3/25 14:24
# @Author  : xiuweihan
"""
import numpy as np
from functools import cached_property
from datetime import datetime, timedelta
from numpy.core.umath import radians, sin, cos, degrees, arctan2, tan, arccos

from hxwsolar.utils import _vec_timedelta


class Solar(object):
    """
    计算常见的太阳能参数
    """
    def __init__(self, lat, lng, dt):
        super(Solar, self).__init__()

        # 检查
        lat, lng, dt, vec_time, vec_space, space_shape, time_shape = self.__validate_init(lat, lng, dt)
        self.__dt = dt    # 时间
        self.__lat = lat  # 纬度，角度
        self.__lng = lng  # 经度，角度
        self.__lat_rad = np.radians(lat)  # 纬度，弧度
        self.__lng_rad = np.radians(lng)  # 经度，弧度
        self.__vec_time = vec_time        # 时间是否为向量
        self.__vec_space = vec_space      # 空间是否为向量
        self.__space_shape = space_shape  # 空间维度
        self.__time_shape = time_shape    # 时间维度

        self._jd, self._jc = self.julian_day_and_century()

    @staticmethod
    def __validate_init(lat, lng, dt):
        """
        检查数据一致性
        :param lat: 纬度
        :param lng: 经度
        :param dt: 时间
        :return:
        """
        # 判断时间类型
        if isinstance(dt, list):
            vec_time = True
            time_shape = len(dt)
        else:
            vec_time = False
            time_shape = tuple()
        # 判断空间类型
        if isinstance(lat, np.ndarray) and isinstance(lng, np.ndarray):
            vec_space = True
            if vec_space:
                assert len(lat.shape) == 2
            if lat.shape != lng.shape:
                raise AttributeError("经纬度维度不匹配！")
            # 如果时间和空间都是向量，则给空间需要添加第三个维度，即时间
            if vec_time and vec_space:
                lat = np.repeat(lat[:, :, np.newaxis], len(dt), axis=2)
                lng = np.repeat(lng[:, :, np.newaxis], len(dt), axis=2)
            space_shape = lat.shape or lng.shape
        else:
            vec_space = False
            space_shape = tuple()
        return lat, lng, dt, vec_time, vec_space, space_shape, time_shape

    @cached_property
    def _tz(self):
        """时区"""
        if not self.__space_shape:
            return round(self.__lng/15)
        else:
            return round(self.__lng.mean()/15)
            # try:
            #     return round(self.__lng[0][0]/15)
            # except (IndexError, TypeError):
            #     return round(self.__lng[0][0][0]/15)

    def julian_day_and_century(self, now_dt=None, tz=None):
        """儒略日/世纪"""
        if not now_dt:
            now_dt = self.__dt
        if not tz:
            tz = self._tz
        # 时间为list
        if isinstance(now_dt, list):
            jd_jc_tuples = [self.julian_day_and_century(ndt, tz) for ndt in now_dt]
            _jd, _jc = zip(*jd_jc_tuples)
            jd = np.array(list(_jd), dtype=float)
            jc = np.array(list(_jc), dtype=float)
            return jd, jc
        else:
            # 当前时间
            ndt = now_dt
            # 时区转换
            ndt += timedelta(hours=-tz)
            # 使用2000年1月1日的参考日
            jan_1st_2000_jd = 2451545
            jan_1st_2000 = datetime(2000, 1, 1, 12, 0, 0)
            time_del = ndt - jan_1st_2000
            jd = float(jan_1st_2000_jd) + float(time_del.total_seconds()) / 86400
            jc = (jd - 2451545) / 36525.0
        return jd, jc

    @cached_property
    def geom_mean_long_sun(self):
        """太阳的平均经度"""
        return (280.46646 + self._jc * (36000.76983 + self._jc * 0.0003032)) % 360

    @cached_property
    def geom_mean_anomaly_sun(self):
        """太阳的平均距平"""
        return 357.52911 + self._jc * (35999.05029 - 0.0001537 * self._jc)

    @cached_property
    def eccentricity_earth_orbit(self):
        """地球轨道偏心率"""
        return 0.016708634 - self._jc * (4.2037e-5 + 1.267e-7 * self._jc)

    @cached_property
    def sun_eq_of_center(self):
        """太阳中心方程"""
        gma = radians(self.geom_mean_anomaly_sun)

        sun_eq_of_center = \
            sin(gma) * (1.914602 - self._jc * (0.004817 + 0.000014 * self._jc)) + \
            sin(2 * gma) * (0.019993 - 0.000101 * self._jc) + \
            sin(3 * gma) * 0.000289
        return sun_eq_of_center

    @cached_property
    def sun_true_long(self):
        """太阳的真实经度"""
        return self.geom_mean_long_sun + self.sun_eq_of_center

    @cached_property
    def sun_true_anomaly(self):
        """太阳的真近点角"""
        return self.geom_mean_anomaly_sun + self.sun_eq_of_center

    @cached_property
    def sun_app_long(self):
        """太阳表观经度"""
        app_long = self.sun_true_long - 0.00569 - 0.00478 * sin(radians(125.04 - 1934.136 * self._jc))
        return app_long

    @cached_property
    def mean_oblique_ellipse(self):
        """地球倾斜平均椭圆"""
        oblique_mean_ellipse = 23 + (26 + (21.448 - self._jc * (
                46.815 + self._jc * (0.00059 - self._jc * 0.001813))) / 60) / 60
        return oblique_mean_ellipse

    @cached_property
    def oblique_corr(self):
        """对地球倾斜椭圆的修正"""
        moe = self.mean_oblique_ellipse
        oblique_corr = moe + 0.00256 * cos(radians(125.04 - 1934.136 * self._jc))
        return oblique_corr

    @cached_property
    def solar_right_ascension(self):
        """太阳赤经角"""
        sal = radians(self.sun_app_long)
        oc = radians(self.oblique_corr)
        right_ascension = degrees(arctan2(np.cos(oc) * sin(sal), cos(sal)))
        return right_ascension

    @cached_property
    def solar_declination(self):
        """太阳赤纬角"""
        sal = np.radians(self.sun_app_long)
        oc = np.radians(self.oblique_corr)

        declination = np.degrees(np.arcsin((np.sin(oc) * np.sin(sal))))
        return declination

    @cached_property
    def equation_of_time(self):
        """时间等式（分钟）"""
        oc = np.radians(self.oblique_corr)
        gml = radians(self.geom_mean_long_sun)
        gma = radians(self.geom_mean_anomaly_sun)
        ec = self.eccentricity_earth_orbit

        vary = tan(oc / 2) ** 2

        equation_of_time = 4 * degrees(
            vary * sin(2 * gml) - 2 * ec * sin(gma) +
            4 * ec * vary * sin(gma) * cos(2 * gml) -
            0.5 * vary * vary * sin(4 * gml) -
            1.25 * ec * ec * sin(2 * gma))

        return equation_of_time

    @cached_property
    def hour_angle_sunrise(self):
        """日出时角"""
        d = radians(self.solar_declination)
        lat = self.__lat_rad
        hour_angle_sunrise = degrees(arccos((cos(radians(90.833)) / (cos(lat) * cos(d)) - tan(lat) * tan(d))))
        return hour_angle_sunrise

    @cached_property
    def noon_time(self):
        """正午时间"""
        lng = self.__lng
        tz = self._tz
        eot = self.equation_of_time
        solar_noon = (720 - 4 * lng - eot + tz * 60) / 1440
        return solar_noon

    @cached_property
    def noon_time_(self):
        """正午时间"""
        return _vec_timedelta(days=self.noon_time)

    @cached_property
    def sunrise_time(self):
        """日出时间"""
        sn = self.noon_time
        ha = self.hour_angle_sunrise

        sunrise = (sn * 1440 - ha * 4) / 1440
        return _vec_timedelta(days=sunrise)

    @cached_property
    def sunset_time(self):
        """日落时间"""
        sn = self.noon_time
        ha = self.hour_angle_sunrise

        sunset = (sn * 1440 + ha * 4) / 1440
        return _vec_timedelta(days=sunset)

    @cached_property
    def sunlight_rate(self):
        """阳光占比"""
        sunlight = 8 * self.hour_angle_sunrise / (60 * 24)
        return sunlight

    @cached_property
    def true_solar_time(self):
        """当地真实时间"""
        lng = self.__lng
        now_datetime = self.__dt
        eot = self.equation_of_time
        rdt = now_datetime

        # 将引用日期时间转换为小数天
        if isinstance(rdt, datetime):
            frac_sec = (rdt - datetime(rdt.year, rdt.month, rdt.day)).total_seconds()
        elif isinstance(rdt, list):
            frac_sec_list = [(_rdt - datetime(_rdt.year, _rdt.month, _rdt.day)).total_seconds() for _rdt in rdt]
            frac_sec = np.array(frac_sec_list)
        else:
            raise TypeError("时间格式有误！")

        frac_hr = frac_sec / (60 * 60)
        frac_day = frac_hr / 24
        frac_day = frac_day
        true_solar = (frac_day * 1440 + eot + 4 * lng - 60 * self._tz) % 1440

        return true_solar

    @cached_property
    def hour_angle(self):
        """太阳时角"""
        ts = self.true_solar_time

        # 矩阵时角计算
        if isinstance(ts, np.ndarray):
            ha = ts
            ha[ha <= 0] = ha[ha <= 0] / 4 + 180
            ha[ha > 0] = ha[ha > 0] / 4 - 180
            hour_angle = ha

        # 标量hour_angle计算
        else:
            if ts <= 0:
                hour_angle = ts / 4 + 180
            else:
                hour_angle = ts / 4 - 180

        return hour_angle

    @cached_property
    def solar_zenith(self):
        """太阳天顶角"""
        lat_r = self.__lat_rad
        d = radians(self.solar_declination)
        ha = radians(self.hour_angle)
        lat = lat_r

        zenith = degrees(arccos(sin(lat) * sin(d) + cos(lat) * cos(d) * cos(ha)))
        return zenith

    @cached_property
    def solar_elevation(self):
        """太阳高度角"""
        zenith = self.solar_zenith
        if isinstance(zenith, np.ndarray) and zenith.shape:
            e = 90.0 - zenith
            ar = e * 0

            ar[e > 85] = 0
            ar[(e > 5) & (e <= 85)] = 58.1 / tan(
                radians(e[(e > 5) & (e <= 85)])) - 0.07 / tan(
                radians(e[(e > 5) & (e <= 85)])) ** 3 + 0.000086 / tan(
                radians(e[(e > 5) & (e <= 85)])) ** 5
            ar[(e > -0.575) & (e <= 5)] = 1735 + e[(e > -0.575) & (e <= 5)] * (
                    103.4 + e[(e > -0.575) & (e <= 5)] * (-12.79 + e[(e > -0.575) & (e <= 5)] * 0.711))
            ar[e <= -0.575] = -20.772 / tan(radians(e[e <= -0.575]))
        else:
            e = 90.0 - zenith
            er = radians(e)

            if e > 85:
                ar = 0
            elif e > 5:
                ar = 58.1 / tan(er) - 0.07 / tan(er) ** 3 + 0.000086 / tan(er) ** 5
            elif e > -0.575:
                ar = 1735 + e * (103.4 + e * (-12.79 + e * 0.711))
            else:
                ar = -20.772 / tan(er)

        elevation_noatmo = e
        atmo_refraction = ar / 3600
        elevation = elevation_noatmo + atmo_refraction

        return elevation

    @cached_property
    def solar_azimuth(self):
        """太阳方位角"""
        lat_r = self.__lat_rad
        lat = lat_r
        d = radians(self.solar_declination)  # 总是返回numpy数组，即使1x1
        ha = radians(self.hour_angle)        # 总是返回numpy数组，即使1x1
        z = radians(self.solar_zenith)       # 总是返回numpy数组，即使1x1

        # 纬度是向量
        if isinstance(lat_r, np.ndarray):
            if lat_r.shape:
                lat_vec = True
            else:
                lat_vec = False
        else:
            lat_vec = False

        if isinstance(d, np.ndarray):
            if d.shape:
                time_vec = True
            else:
                time_vec = False
        else:
            time_vec = False

        # 案例4:时间空间均为向量
        if time_vec and lat_vec:
            # 如果时间和空间都是矢量化的，我们必须将赤纬(时间)投射到第三维。通过复制所有纬度/经度对

            if len(lat_r.shape) == 3 and len(d.shape) == 1:
                if lat_r.shape[2] == d.shape[0]:
                    d = np.repeat(d[np.newaxis, :], lat_r.shape[1], axis=0)
                    d = np.repeat(d[np.newaxis, :], lat_r.shape[0], axis=0)

            az = ha * 0

            ha_p = (ha > 0)
            ha_n = (ha <= 0)

            az_ha_p = arccos(
                ((sin(lat[ha_p]) * cos(z[ha_p])) - sin(d[ha_p]))
                / (cos(lat[ha_p]) * sin(z[ha_p])))
            az[ha_p] = (degrees(az_ha_p) + 180) % 360

            az_ha_n = arccos(
                ((sin(lat[ha_n]) * cos(z[ha_n])) - sin(d[ha_n]))
                / (cos(lat[ha_n]) * sin(z[ha_n]))
            )
            az[ha_n] = (540 - degrees(az_ha_n)) % 360
            azimuth = az
        # 情形3:空间常量，时间向量(以弧度计lat_r为标量)
        elif time_vec and not lat_vec:
            az = ha * 0

            ha_p = (ha > 0)
            ha_n = (ha <= 0)

            az_ha_p = arccos(
                ((sin(lat) * cos(z[ha_p])) - sin(d[ha_p]))
                / (cos(lat) * sin(z[ha_p])))
            az[ha_p] = (degrees(az_ha_p) + 180) % 360

            az_ha_n = arccos(
                ((sin(lat) * cos(z[ha_n])) - sin(d[ha_n]))
                / (cos(lat) * sin(z[ha_n]))
            )
            az[ha_n] = (540 - degrees(az_ha_n)) % 360
            azimuth = az

        # 情形2:时间常量，空间向量(赤纬为标量)
        elif lat_vec and not time_vec:
            az = ha * 0

            ha_p = (ha > 0)
            ha_n = (ha <= 0)

            az_ha_p = arccos(
                ((sin(lat[ha_p]) * cos(z[ha_p])) - sin(d))
                / (cos(lat[ha_p]) * sin(z[ha_p])))
            az[ha_p] = (degrees(az_ha_p) + 180) % 360

            az_ha_n = arccos(
                ((sin(lat[ha_n]) * cos(z[ha_n])) - sin(d))
                / (cos(lat[ha_n]) * sin(z[ha_n]))
            )
            az[ha_n] = (540 - degrees(az_ha_n)) % 360
            azimuth = az

        # 方案1：输入否是标量
        else:

            if ha > 0:
                azimuth = (degrees(arccos(((sin(lat) * cos(z)) - sin(d)) / (cos(lat) * sin(z)))) + 180) % 360
            else:
                azimuth = (540 - degrees(arccos(((sin(lat) * cos(z)) - sin(d)) / (cos(lat) * sin(z))))) % 360

        return azimuth

    def __call__(self, *args, **kwargs):
        compute_list = [f for f in dir(self)
                        if not callable(getattr(self, f))
                        and not f.startswith("__")
                        and f != "summarize"]
        for f in compute_list:
            result = getattr(self, f)
            if isinstance(result, np.ndarray):
                result_mean = result.mean()
                print(f"{f:27s} {str(result.shape):20s} {result_mean}")
            else:
                dim = len(result) if isinstance(result, list) else 1
                print(f"{f:27s} {str(dim):20s} {result}")


if __name__ == '__main__':
    solar = Solar(lat=31, lng=121, dt=datetime(2007, 12, 23, 15, 12, 0))
    print(solar())
    print('-'*100)
    solar = Solar(lat=30, lng=120, dt=[datetime(2007, 9, 23, 13, 12, 0), datetime(2007, 9, 23, 15, 12, 0)])
    print(solar())
    _lat, _lng = [30, 31], [120, 121]
    _lat, _lng = np.meshgrid(_lat, _lng)
    solar = Solar(lat=_lat, lng=_lng,
                  dt=[datetime(2007, 9, 23, 13, 12, 0), datetime(2007, 9, 23, 15, 12, 0)])
    print(solar())
