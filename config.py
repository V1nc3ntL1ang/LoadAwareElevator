# ============================================================
# Load-Aware Elevator Scheduling — Configuration File
# ============================================================

from models.utils import duration_seconds, h2s

# ------------------------
# Building Parameters
# ------------------------
BUILDING_FLOORS = 15
BUILDING_FLOOR_HEIGHT = 3.5
ELEVATOR_COUNT = 4
ELEVATOR_CAPACITY = 1200.0

# ------------------------
# Kinematic Parameters
# ------------------------
KIN_MAX_SPEED_UP_EMPTY = 3.0
KIN_MAX_SPEED_UP_FULL = 2.5
KIN_MAX_SPEED_DOWN_EMPTY = 3.0
KIN_MAX_SPEED_DOWN_FULL = 2.6
KIN_SPEED_DECAY_RATE = 1.2

KIN_ACC_UP_EMPTY = 1.2
KIN_ACC_UP_FULL = 0.9
KIN_DEC_DOWN_EMPTY = 1.2
KIN_DEC_DOWN_FULL = 1.0
KIN_ACC_DECAY_RATE = 1.3

# ------------------------
# Temporal Parameters
# ------------------------
HOLD_BASE_TIME = 1.5
HOLD_EFF_NORMAL = 0.002
HOLD_EFF_CONGESTED = 0.005
HOLD_CONGESTION_THRESHOLD = 400

# ------------------------
# Energy Parameters
# ------------------------
ENERGY_CAR_MASS = 600.0
ENERGY_COUNTERWEIGHT_MASS = 500.0
ENERGY_FRICTION_PER_METER = 50.0
ENERGY_MOTOR_EFFICIENCY = 0.85
ENERGY_STANDBY_POWER = 500.0

# ------------------------
# Simulation Parameters
# ------------------------
SIM_TIME_HORIZON = 300
SIM_TIME_STEP = 1.0
SIM_RANDOM_SEED = 42
SIM_TOTAL_REQUESTS = 1000
SIM_ENABLE_PLOTS = False
SIM_ENABLE_LOG = True

# ------------------------
# Request Generation Parameters
# ------------------------

# Off-peak request distribution
OFFPEAK_MAIN_FLOW_RATIO = 0.8  # fraction of requests involving 1st floor
OFFPEAK_LOAD_MIN = 50  # kg
OFFPEAK_LOAD_MAX = 110  # kg

# Peak request distribution
PEAK_LOAD_MIN = 60  # kg
PEAK_LOAD_MAX = 150  # kg
PEAK_SIGMA_RATIO = 0.1  # std = ratio * SIM_TIME_HORIZON
PEAK_MORNING_MU_RATIO = 0.2  # early peak
PEAK_EVENING_MU_RATIO = 0.7  # late peak

# ------------------------
# Objective Weights
# ------------------------
WEIGHT_TIME = 1.0
WEIGHT_ENERGY = 0.001


# ============================================================
# Request Generation Parameters
# ============================================================


# ------------------------
# 通用控制项
# ------------------------
DEFAULT_LOAD_MIN = 50
DEFAULT_LOAD_MAX = 110
DEFAULT_SIGMA_RATIO = 0.05
DEFAULT_INTENSITY = 1.0

# ============================================================
# 1. 平峰参数 (Uniform Distribution)
# ============================================================

OFFPEAK_INTENSITY = 0.4  # 平峰人流强度（相对高峰）
OFFPEAK_LOAD_MIN = 50
OFFPEAK_LOAD_MAX = 110

# 三个比例之和应为 1：
OFFPEAK_RATIO_ORIGIN1 = 0.45  # 从一楼出发（上行）
OFFPEAK_RATIO_DEST1 = 0.45  # 到一楼（下行）
OFFPEAK_RATIO_OTHER = 0.10  # 楼层间流动（上下均可）

# ============================================================
# 2. 高峰参数 (Gaussian Distribution)
# ============================================================

PEAK_INTENSITY = 1.0
PEAK_LOAD_MIN = 60
PEAK_LOAD_MAX = 150
PEAK_SIGMA_RATIO = 0.05  # 时间分布尖锐程度（越小越集中）

# ---- 早高峰 (morning peak)
PEAK_MORNING_RATIO_ORIGIN1 = 0.05  # 少数从一楼出发（上行）
PEAK_MORNING_RATIO_DEST1 = 0.90  # 多数到一楼（下行）
PEAK_MORNING_RATIO_OTHER = 0.05  # 少量楼层间流动

# ---- 晚高峰 (evening peak)
PEAK_EVENING_RATIO_ORIGIN1 = 0.90  # 多数从一楼出发（上行）
PEAK_EVENING_RATIO_DEST1 = 0.05  # 少数到一楼（下行）
PEAK_EVENING_RATIO_OTHER = 0.05  # 少量楼层间流动

# ============================================================
# 3. 时段时间定义（支持小时:分钟）
# ============================================================

# 早高峰 7:00–9:00
PEAK_MORNING_START = (8, 0)
PEAK_MORNING_END = (9, 0)

# 白天平峰 9:00–17:00
OFFPEAK_DAY_START = (9, 0)
OFFPEAK_DAY_END = (17, 0)

# 晚高峰 17:00–21:00
PEAK_EVENING_START = (17, 0)
PEAK_EVENING_END = (21, 0)

# 夜间平峰 21:00–次日7:00（跨日）
OFFPEAK_NIGHT_START = (21, 0)
OFFPEAK_NIGHT_END = (7 + 24, 0)  # 次日 7:00 = 31 小时

# 高峰中心
PEAK_MORNING_MU = "8:30"
PEAK_EVENING_MU = "18:00"

# ============================================================
# 自动计算各时段比例（无需手动修改）
# ============================================================

DAY_DURATION = 24 * 3600  # 一天总秒数


# 各时段长度
_duration_morning = duration_seconds(PEAK_MORNING_START, PEAK_MORNING_END)
_duration_day = duration_seconds(OFFPEAK_DAY_START, OFFPEAK_DAY_END)
_duration_evening = duration_seconds(PEAK_EVENING_START, PEAK_EVENING_END)
_duration_night = duration_seconds(OFFPEAK_NIGHT_START, OFFPEAK_NIGHT_END)

_total = _duration_morning + _duration_day + _duration_evening + _duration_night

# 自动计算比例
PEAK_MORNING_RATIO = _duration_morning / _total
OFFPEAK_DAY_RATIO = _duration_day / _total
PEAK_EVENING_RATIO = _duration_evening / _total
OFFPEAK_NIGHT_RATIO = _duration_night / _total
