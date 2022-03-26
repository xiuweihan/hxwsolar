#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2022/3/25 14:25
# @Author  : xiuweihan
"""
import numpy as np
from datetime import datetime, timedelta


def local_datetime_array(
        start_date: datetime,
        end_date: datetime,
        time_step: timedelta):
    """
    构建时间序列
    :param start_date:
    :param end_date:
    :param time_step:
    :return:
    """
    span = end_date - start_date
    num_steps = int(span / time_step)
    dts = [start_date + (time_step * i) for i in range(num_steps)]
    return dts


def _vec_timedelta(**kwargs):
    """
    根据时区进行时间校正
    """
    # 所有参数都可以是向量，但必须是相同的形状!
    if len(kwargs) > 1:
        if not all([v1.shape == v2.shape for v1 in kwargs.values() for v2 in kwargs.values()]):
            raise Exception(
                "If vectorized arguments are used, they must all be "
                "identical shapes, got [{}]".format(
                    ",".join(
                        ["{}=({})".format(k, len(v)) for k, v in kwargs.items()]
                    )))
    # 参数类型调节
    is_vec = False
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            is_vec = True
            kwargs[k] = kwargs[k].astype(dtype=float)
    # 生成timedelta的矢量化版本
    vf = np.vectorize(timedelta)
    # 返回合适的，如果需要时区则直接+timedelta(hours=tz)
    if is_vec:
        return vf(**kwargs)
    else:
        return timedelta(**kwargs)


if __name__ == '__main__':
    _start_date = datetime(2019, 1, 1, 0, 0)
    _end_date = datetime(2019, 1, 1, 1, 0)
    _time_step = timedelta(minutes=10)
    res = local_datetime_array(_start_date, _end_date, _time_step)
    print(res)
