from datetime import datetime
from typing import Union, List

import numpy as np


class Constants(object):
    """常数"""

    def __init__(self):
        self.sun_surf_rad = 63156942.6  # 太阳表面辐射(W/m^2)
        self.sun_radius = 695800000.  # 太阳的半径，单位是米
        self.orbital_period = 365.2563630  # 地球自转一周需要多少天
        self.altitude = -0.01448623  # 太阳盘中心高度

        # 太阳能常量
        self.earth_radius = 6371000  # 单球地球半径(m)
        self.atm_height = 9000       # 有效大气高度

        # 光合作用常数
        self.etta_photon = 4.56  # micro mol / J
        self.etta_par = 0.368  # 5800K黑体太阳辐射中具有光合活性的部分(W/W)


CONSTANTS = Constants()
FlexNum = Union[float, int, np.ndarray]
FlexDate = Union[datetime, List[datetime]]
