import os
import random
from typing import Tuple


def h2s(hour, minute: int = 0) -> float:
    """
    Convert time-of-day to seconds.
    支持以下三种输入：
      - 两个整数：h2s(7, 30)
      - 单个整数：h2s(7)
      - 字符串：h2s("7:30") 或 h2s("18:45")
    """
    if isinstance(hour, str):
        parts = hour.strip().split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
    return hour * 3600.0 + minute * 60.0


def duration_seconds(start, end) -> float:
    """计算时间段长度（秒），支持跨日。"""
    if isinstance(start, tuple):
        start_s = h2s(*start)
    else:
        start_s = h2s(start)
    if isinstance(end, tuple):
        end_s = h2s(*end)
    else:
        end_s = h2s(end)
    if end_s < start_s:
        end_s += 24 * 3600  # 跨日修正
    return end_s - start_s


def validate_ratios(r_origin1: float, r_dest1: float, r_other: float) -> Tuple[float, float]:
    """Normalize direction ratios and return cumulative cutoffs."""
    total = r_origin1 + r_dest1 + r_other
    if total <= 0:
        raise ValueError("All ratios are zero or negative.")

    r_origin1 /= total
    r_dest1 /= total
    r_other /= total

    c1 = r_origin1
    c2 = r_origin1 + r_dest1
    return c1, c2


def rand_upper_floor(max_floor: int) -> int:
    """Randomly select a floor in [2, max_floor]."""
    if max_floor < 2:
        raise ValueError("max_floor must be at least 2.")
    return random.randint(2, max_floor)


def rand_other_pair(max_floor: int) -> Tuple[int, int]:
    """返回两个不相等且都不为 1 的楼层（2..max_floor）。"""
    a = rand_upper_floor(max_floor)
    b = rand_upper_floor(max_floor)
    while b == a:
        b = rand_upper_floor(max_floor)
    return a, b


def ensure_directory(path: str):
    """Create directory if it does not exist."""
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def plot_elevator_movements(elevators, filename="results/plots/elevator_schedule.png"):
    """
    Plot elevator service schedule (floor vs. task index).
    Each elevator's served requests are visualized sequentially.
    """
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        os.environ.setdefault("KMP_AFFINITY", "disabled")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Plot Skipped] {exc}")
        return

    ensure_directory(os.path.dirname(filename))
    plt.figure(figsize=(8, 5))
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    for i, elev in enumerate(elevators):
        floors = []
        tasks = []
        current_task = 0

        for req in elev.served_requests:
            floors += [req.origin, req.destination]
            tasks += [current_task, current_task + 1]
            current_task += 1

        if floors:
            plt.plot(
                tasks,
                floors,
                marker="o",
                color=colors[i % len(colors)],
                label=f"Elevator {elev.id}",
            )

    plt.xlabel("Task Index")
    plt.ylabel("Floor Level")
    plt.title("Elevator Service Schedule (Baseline Strategy)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[Plot Saved] Elevator movement plot saved to: {filename}")


def print_elevator_queues(elevators):
    """
    Print the list of requests served by each elevator.
    """
    print("\n=== Elevator Queues ===")
    for elev in elevators:
        print(f"Elevator {elev.id}:")
        if not elev.served_requests:
            print("  (No requests assigned)")
            continue
        for req in elev.served_requests:
            print(
                f"  Req#{req.id}: Floor {req.origin} → {req.destination} | Load={req.load:.1f}kg"
            )
    print("========================\n")


def log_results_to_file(
    elevators,
    total_time,
    total_energy,
    total_cost,
    filename="results/plots/summary.txt",
):
    """
    Save textual results and queue info to a text file.
    """
    ensure_directory(os.path.dirname(filename))
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== Elevator Baseline Simulation Summary ===\n")
        f.write(f"Total Time: {total_time:.2f} s\n")
        f.write(f"Total Energy: {total_energy:.2f} J\n")
        f.write(f"Total Objective Cost: {total_cost:.2f}\n\n")

        for elev in elevators:
            f.write(f"Elevator {elev.id} Queue:\n")
            if not elev.served_requests:
                f.write("  (No requests)\n")
            else:
                for req in elev.served_requests:
                    f.write(
                        f"  Req#{req.id}: {req.origin} → {req.destination}, load={req.load:.1f}kg\n"
                    )
            f.write("\n")

    print(f"[Log Saved] Summary written to {filename}")
