from dataclasses import dataclass


@dataclass
class Request:
    id: int
    origin: int
    destination: int
    load: float
    arrival_time: float
    """Passenger request / 乘客请求。

    EN: A single hall+car call represented as origin, destination, load (kg),
    and arrival time (s). Times are in seconds-of-day.

    ZH: 一个乘客请求，包含起点、终点、载荷（kg）与到达时刻（秒）。
    """


@dataclass
class ElevatorState:
    id: int
    floor: int
    load: float = 0.0
    direction: str = "idle"  # "up", "down", or "idle"
    """Elevator state / 电梯状态。

    EN: Minimal state for scheduling decisions: current floor, load and
    coarse direction flag.

    ZH: 用于调度决策的最小状态：当前楼层、载荷与方向标记。
    """
"""
Core data structures / 核心数据结构
-----------------------------------

EN: Lightweight dataclasses for requests and elevator state. These are the
only shared structs passed between schedulers and the simulator core.

ZH: 定义请求与电梯状态的轻量数据类，是调度器与仿真核心之间共享的数据结构。
"""
