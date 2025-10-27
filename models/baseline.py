import config as cfg
from models.kinematics import travel_time
from models.temporal import hold_time
from models.energy import segment_energy


def assign_requests_baseline(requests, elevators):
    for e in elevators:
        e.queue = []
        e.served_requests = []

    for req in requests:
        best_elev = min(elevators, key=lambda e: abs(e.floor - req.origin))
        best_elev.queue.append(req)
        best_elev.served_requests.append(req)


def simulate_baseline(elevators):
    """
    Execute baseline simulation: process all elevator queues sequentially.
    Returns total_time, total_energy
    """
    total_time = 0
    total_energy = 0

    for elev in elevators:
        current_floor = elev.floor
        current_load = 0

        while elev.queue:
            req = elev.queue.pop(0)
            direction = "up" if req.destination > current_floor else "down"

            # move to origin
            t_to_origin = travel_time(current_load, current_floor, req.origin)
            e_to_origin = segment_energy(
                current_load,
                abs(req.origin - current_floor) * cfg.BUILDING_FLOOR_HEIGHT,
                direction,
            )
            total_time += t_to_origin
            total_energy += e_to_origin

            # boarding hold
            t_hold = hold_time(req.load, 0)
            total_time += t_hold

            # move to destination
            t_to_dest = travel_time(req.load, req.origin, req.destination)
            e_to_dest = segment_energy(
                req.load,
                abs(req.destination - req.origin) * cfg.BUILDING_FLOOR_HEIGHT,
                direction,
            )
            total_time += t_to_dest
            total_energy += e_to_dest

            current_floor = req.destination
            current_load = 0

    return total_time, total_energy
