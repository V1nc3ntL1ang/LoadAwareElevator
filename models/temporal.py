"""
Temporal model / 停站时间模型
-----------------------------

EN: Door dwell time as a function of boarding and alighting mass. Captures
normal vs. congested regimes via a two-slope model.

ZH: 基于上下客重量估计开门停站时间，通过两段斜率刻画正常/拥挤两种工况。
"""

from models import config as cfg


def hold_time(boarding_weight, alighting_weight):
    """Door dwell vs passenger mass / 停站时间与客流重量。

    EN: Below the congestion threshold, time scales with total mass using the
    normal slope; the excess beyond threshold uses the congested slope.

    ZH: 当总重量不超过拥挤阈值时按“正常斜率”增长；超过部分按“拥挤斜率”增长。
    """
    total_weight = boarding_weight + alighting_weight
    if total_weight <= cfg.HOLD_CONGESTION_THRESHOLD:
        return cfg.HOLD_BASE_TIME + cfg.HOLD_EFF_NORMAL * total_weight
    else:
        normal_part = cfg.HOLD_EFF_NORMAL * cfg.HOLD_CONGESTION_THRESHOLD
        # 超出阈值部分按拥挤系数计算 / congested segment beyond threshold
        congested_part = cfg.HOLD_EFF_CONGESTED * (
            total_weight - cfg.HOLD_CONGESTION_THRESHOLD
        )
        return cfg.HOLD_BASE_TIME + normal_part + congested_part
