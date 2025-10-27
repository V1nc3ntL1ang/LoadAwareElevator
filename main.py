from models.request import generate_requests_aday
from models.variables import ElevatorState
from models.baseline import assign_requests_baseline, simulate_baseline
from models.objective import compute_objective
from models.utils import (
    plot_elevator_movements,
    print_elevator_queues,
    log_results_to_file,
)
import config as cfg

SIM_NUM_REQUESTS = 1000


def main():
    # 1. 生成请求
    requests = generate_requests_aday(SIM_NUM_REQUESTS)

    # 2. 初始化电梯
    elevators = [ElevatorState(id=k + 1, floor=1) for k in range(cfg.ELEVATOR_COUNT)]

    # 3. 基线调度与模拟
    assign_requests_baseline(requests, elevators)
    total_time, total_energy = simulate_baseline(elevators)
    total_cost = compute_objective(total_time, total_energy)

    # 4. 输出与可视化
    print_elevator_queues(elevators)
    log_results_to_file(elevators, total_time, total_energy, total_cost)
    plot_elevator_movements(elevators)

    print(
        f"Total Time: {total_time:.2f}s | "
        f"Total Energy: {total_energy:.2f}J | "
        f"Cost: {total_cost:.2f}"
    )


if __name__ == "__main__":
    main()
