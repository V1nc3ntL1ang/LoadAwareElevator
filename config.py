# ============================================================
# Load-Aware Elevator Scheduling — Configuration File
# ============================================================

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
# Load-Aware Elevator Scheduling — Configuration File
# ============================================================

# ------------------------
# Building Parameters
# ------------------------
BUILDING_FLOORS = 15
BUILDING_ELEVATORS = 4
BUILDING_FLOOR_HEIGHT = 3.5  # meters

# ------------------------
# Simulation Parameters
# ------------------------
SIM_RANDOM_SEED = 42
SIM_TIME_HORIZON = 300  # 单段仿真时间（秒）
DAY_DURATION = 24 * 3600  # 一天（秒）

# ------------------------
# Request Generation — Common
# ------------------------
DEFAULT_LOAD_MIN = 50
DEFAULT_LOAD_MAX = 110
DEFAULT_SIGMA_RATIO = 0.05
DEFAULT_MAINFLOW_RATIO = 0.8
DEFAULT_INTENSITY = 1.0

# ------------------------
# Off-Peak Parameters
# ------------------------
OFFPEAK_INTENSITY = 0.4  # 人流强度相对高峰
OFFPEAK_MAINFLOW_RATIO = 0.8  # 一楼相关请求比例
OFFPEAK_LOAD_MIN = 50
OFFPEAK_LOAD_MAX = 110

# ------------------------
# Peak Parameters
# ------------------------
PEAK_INTENSITY = 1.0  # 相对人流强度（标准化为1）
PEAK_MAINFLOW_RATIO = 0.95  # 主流方向比例（早高峰下行，晚高峰上行）
PEAK_LOAD_MIN = 60
PEAK_LOAD_MAX = 150
PEAK_SIGMA_RATIO = 0.05  # 时间集中程度 (σ)

# ------------------------
# Daytime Ratio (sum ≈ 1.0)
# ------------------------
PEAK_MORNING_RATIO = 0.25
OFFPEAK_DAY_RATIO = 0.35
PEAK_EVENING_RATIO = 0.25
OFFPEAK_NIGHT_RATIO = 0.15

# ------------------------
# Peak time centers (as ratio of 24h)
# ------------------------
PEAK_MORNING_MU_RATIO = 0.08  # ≈ 07:40
PEAK_EVENING_MU_RATIO = 0.75  # ≈ 18:00
