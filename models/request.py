import random
import config as cfg
from models.variables import Request


# ============================================================
# 基础生成函数
# ============================================================


def generate_offpeak_uniform(
    num_requests: int,
    start_time: float,
    end_time: float,
    intensity: float = cfg.OFFPEAK_INTENSITY,
    mainflow_ratio: float = cfg.OFFPEAK_MAINFLOW_RATIO,
    load_min: float = cfg.OFFPEAK_LOAD_MIN,
    load_max: float = cfg.OFFPEAK_LOAD_MAX,
):
    """
    Generate requests during off-peak hours with uniform time distribution.
    - 时间分布均匀
    - 控制一楼相关比例(mainflow_ratio)
    - 强度intensity可缩放请求数量
    """
    random.seed(cfg.SIM_RANDOM_SEED)
    n_requests = int(num_requests * intensity)
    requests = []

    for i in range(n_requests):
        # 一楼相关流向
        if random.random() < mainflow_ratio:
            if random.random() < 0.5:
                origin, destination = 1, random.randint(2, cfg.BUILDING_FLOORS)
            else:
                origin, destination = random.randint(2, cfg.BUILDING_FLOORS), 1
        else:
            # 楼层间移动
            origin = random.randint(2, cfg.BUILDING_FLOORS)
            destination = random.randint(2, cfg.BUILDING_FLOORS)
            while destination == origin:
                destination = random.randint(2, cfg.BUILDING_FLOORS)

        load = random.uniform(load_min, load_max)
        arrival_time = random.uniform(start_time, end_time)

        requests.append(Request(i + 1, origin, destination, load, arrival_time))

    return requests


def generate_peak_gaussian(
    num_requests: int,
    start_time: float,
    end_time: float,
    peak_type: str = "morning",
    intensity: float = cfg.PEAK_INTENSITY,
    mainflow_ratio: float = cfg.PEAK_MAINFLOW_RATIO,
    load_min: float = cfg.PEAK_LOAD_MIN,
    load_max: float = cfg.PEAK_LOAD_MAX,
    sigma_ratio: float = cfg.PEAK_SIGMA_RATIO,
):
    """
    Generate requests during peak hours with Gaussian time distribution.
    - 时间呈正态分布
    - 支持方向比例(mainflow_ratio)
    """
    random.seed(cfg.SIM_RANDOM_SEED + (1 if peak_type == "evening" else 0))
    n_requests = int(num_requests * intensity)
    requests = []

    mu_ratio = (
        cfg.PEAK_MORNING_MU_RATIO
        if peak_type == "morning"
        else cfg.PEAK_EVENING_MU_RATIO
    )
    mu_time = mu_ratio * cfg.DAY_DURATION

    for i in range(n_requests):
        # 生成方向
        if peak_type == "morning":
            # 主流下行
            if random.random() < mainflow_ratio:
                origin = random.randint(2, cfg.BUILDING_FLOORS)
                destination = 1
            else:
                origin = 1
                destination = random.randint(2, cfg.BUILDING_FLOORS)
        else:
            # 主流上行
            if random.random() < mainflow_ratio:
                origin = 1
                destination = random.randint(2, cfg.BUILDING_FLOORS)
            else:
                origin = random.randint(2, cfg.BUILDING_FLOORS)
                destination = 1

        load = random.uniform(load_min, load_max)

        arrival_time = random.gauss(mu=mu_time, sigma=cfg.DAY_DURATION * sigma_ratio)
        arrival_time = max(start_time, min(arrival_time, end_time))

        requests.append(Request(i + 1, origin, destination, load, arrival_time))

    return requests


# ============================================================
# 一天模拟函数
# ============================================================


def generate_requests_aday(total_requests: int):
    """
    Generate all requests for one day by combining peak and off-peak periods.
    """
    requests = []

    # ---- 1. 各时段请求数 ----
    n_morning = int(total_requests * cfg.PEAK_MORNING_RATIO)
    n_day = int(total_requests * cfg.OFFPEAK_DAY_RATIO)
    n_evening = int(total_requests * cfg.PEAK_EVENING_RATIO)
    n_night = int(total_requests * cfg.OFFPEAK_NIGHT_RATIO)

    # ---- 2. 时间范围 (秒) ----
    def to_sec(h):
        return h * 3600

    # 早高峰 7:00–9:00
    requests += generate_peak_gaussian(
        n_morning, to_sec(7), to_sec(9), peak_type="morning"
    )

    # 白天平峰 9:00–17:00
    requests += generate_offpeak_uniform(
        n_day, to_sec(9), to_sec(17), intensity=cfg.OFFPEAK_INTENSITY
    )

    # 晚高峰 17:00–21:00
    requests += generate_peak_gaussian(
        n_evening, to_sec(17), to_sec(21), peak_type="evening"
    )

    # 夜间平峰 21:00–次日7:00
    requests += generate_offpeak_uniform(
        n_night, to_sec(21), to_sec(31), intensity=cfg.OFFPEAK_INTENSITY
    )

    requests.sort(key=lambda r: r.arrival_time)
    return requests
