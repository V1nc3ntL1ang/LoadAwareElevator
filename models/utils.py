import matplotlib.pyplot as plt
import config as cfg


def plot_elevator_movements(elevators, filename="results/plots/elevator_schedule.png"):
    """
    Plot elevator service schedule (floor vs. task index).
    Each elevator's served requests are visualized sequentially.
    """
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
