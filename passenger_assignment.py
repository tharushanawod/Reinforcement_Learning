"""
passenger_assignment.py
=======================
Passenger Assignment & Frequency Calculation module for BNDFS.

Implements:
  - Frequency setting: f_k = Q_{k,max} / (delta_max * C_k)
  - Fleet rescaling when total fleet exceeds 99 buses
  - 0-transfer (direct) passenger routing
  - 1-transfer passenger routing with 10% travel-time tolerance
  - Performance metrics computation

Reference: Yoo, Lee, & Han (2023)
"""

import numpy as np
from collections import defaultdict
from mandl_network import (
    NUM_NODES, LINK_TRAVEL_TIME, OD_DEMAND, TOTAL_DEMAND,
    SHORTEST_PATH_DIST, ADJACENCY,
)

# ---------------------------------------------------------------------------
# Operational constants
# ---------------------------------------------------------------------------
MAX_FLEET = 99
BUS_CAPACITY = 50          # 40 seats * 1.25 load factor
SEAT_CAPACITY = 40         # C_k in the paper
LOAD_FACTOR = 1.25         # delta_max
TRANSFER_PENALTY = 5.0     # minutes
MIN_FREQUENCY = 1.0        # minimum 1 bus / hour on any active route


def route_travel_time(route):
    """Compute one-way travel time for a route (sequence of 0-indexed nodes)."""
    tt = 0.0
    for i in range(len(route) - 1):
        key = (route[i], route[i + 1])
        tt += LINK_TRAVEL_TIME[key]
    return tt


def route_round_trip_time(route):
    """Round-trip travel time = 2 * one-way travel time."""
    return 2.0 * route_travel_time(route)


def build_route_stop_sets(routes):
    """For each route, return the set of stops and a mapping stop -> index."""
    stop_sets = []
    stop_indices = []
    for route in routes:
        s = set(route)
        idx = {node: i for i, node in enumerate(route)}
        stop_sets.append(s)
        stop_indices.append(idx)
    return stop_sets, stop_indices


def compute_in_vehicle_time(route, i_idx, j_idx):
    """In-vehicle time from stop at index i_idx to stop at index j_idx in route."""
    if i_idx > j_idx:
        i_idx, j_idx = j_idx, i_idx
    tt = 0.0
    for k in range(i_idx, j_idx):
        tt += LINK_TRAVEL_TIME[(route[k], route[k + 1])]
    return tt


def evaluate_route_set(routes):
    """
    Full evaluation of a route set:
      1. Find direct (0-transfer) paths for each OD pair.
      2. Find 1-transfer paths for remaining demand.
      3. Compute initial frequencies from peak link flows.
      4. Rescale if fleet exceeds MAX_FLEET.
      5. Return performance metrics.

    Parameters
    ----------
    routes : list of list[int]
        Each inner list is a sequence of 0-indexed node IDs.

    Returns
    -------
    metrics : dict with keys:
        'd_sat'              - fraction of demand satisfied
        'total_travel_time'  - total travel time across all passengers (min)
        'in_vehicle_time'    - total in-vehicle time (min)
        'waiting_time'       - total waiting time (min)
        'transfer_time'      - total transfer penalty time (min)
        'direct_pct'         - % of demand served directly (0-transfer)
        'transfer_pct'       - % of demand served with 1 transfer
        'unmet_pct'          - % of unmet demand
        'fleet_size'         - total buses required
        'frequencies'        - list of frequencies per route
        'reward'             - d_sat / total_travel_time  (the RL reward)
    """
    if not routes:
        return _empty_metrics()

    n_routes = len(routes)
    stop_sets, stop_indices = build_route_stop_sets(routes)

    # ------------------------------------------------------------------
    # Phase 1: Identify candidate paths for each OD pair
    # ------------------------------------------------------------------
    # direct_paths[o][d] = list of (route_idx, ivt)
    # transfer_paths[o][d] = list of (route_idx1, route_idx2, transfer_node, ivt_total)
    direct_paths = defaultdict(lambda: defaultdict(list))
    transfer_paths = defaultdict(lambda: defaultdict(list))

    for r_idx, route in enumerate(routes):
        stops = stop_sets[r_idx]
        idx_map = stop_indices[r_idx]
        route_nodes = list(route)
        n_stops = len(route_nodes)
        # Direct paths
        for i in range(n_stops):
            for j in range(n_stops):
                if i == j:
                    continue
                o, d = route_nodes[i], route_nodes[j]
                ivt = compute_in_vehicle_time(route, idx_map[o], idx_map[d])
                direct_paths[o][d].append((r_idx, ivt))

    # 1-transfer paths: enumerate pairs of routes sharing at least one stop
    for r1 in range(n_routes):
        for r2 in range(n_routes):
            if r1 == r2:
                continue
            common = stop_sets[r1] & stop_sets[r2]
            if not common:
                continue
            for transfer_node in common:
                for o in stop_sets[r1]:
                    if o == transfer_node:
                        continue
                    for d in stop_sets[r2]:
                        if d == transfer_node or d == o:
                            continue
                        ivt1 = compute_in_vehicle_time(
                            routes[r1],
                            stop_indices[r1][o],
                            stop_indices[r1][transfer_node],
                        )
                        ivt2 = compute_in_vehicle_time(
                            routes[r2],
                            stop_indices[r2][transfer_node],
                            stop_indices[r2][d],
                        )
                        ivt_total = ivt1 + ivt2
                        # 10% tolerance check against shortest path
                        sp = SHORTEST_PATH_DIST[o, d]
                        if sp < 1e8 and ivt_total <= sp * 1.10 + 1e-9:
                            transfer_paths[o][d].append(
                                (r1, r2, transfer_node, ivt_total)
                            )

    # ------------------------------------------------------------------
    # Phase 2: Assign demand & compute link flows for frequency setting
    # ------------------------------------------------------------------
    # We need an iterative approach: frequencies depend on flows, flows
    # depend on frequency-proportional allocation.  We bootstrap with
    # uniform frequencies, then iterate a few times.

    # Initial uniform frequency = 1.0 for all routes
    freqs = np.ones(n_routes, dtype=np.float64)

    for _iteration in range(3):  # 3 passes suffice for convergence
        # Reset accumulators
        demand_direct = 0.0
        demand_transfer = 0.0
        demand_unmet = 0.0

        total_ivt = 0.0
        total_wait = 0.0
        total_xfer_penalty = 0.0

        # link_flow[route_idx][(i, j)] = total passengers on that directed link
        link_flows = [defaultdict(float) for _ in range(n_routes)]

        for o in range(NUM_NODES):
            for d in range(NUM_NODES):
                dem = OD_DEMAND[o, d]
                if dem <= 0:
                    continue

                # --- Try direct routes first ---
                dp = direct_paths[o][d]
                if dp:
                    # Allocate proportionally to frequency
                    weights = np.array([freqs[r] for r, _ in dp], dtype=np.float64)
                    total_w = weights.sum()
                    if total_w < 1e-12:
                        total_w = 1.0
                    probs = weights / total_w

                    for idx, (r_idx, ivt) in enumerate(dp):
                        frac = probs[idx]
                        pax = dem * frac
                        demand_direct += pax

                        # In-vehicle time
                        total_ivt += pax * ivt

                        # Waiting time = 60 / (2 * f_k) = 30 / f_k  (avg wait)
                        f_k = max(freqs[r_idx], MIN_FREQUENCY)
                        wait = 30.0 / f_k
                        total_wait += pax * wait

                        # Accumulate link flows along route from o to d
                        _accumulate_link_flow(
                            routes[r_idx], stop_indices[r_idx],
                            o, d, pax, link_flows[r_idx]
                        )
                    continue

                # --- Try 1-transfer routes ---
                tp = transfer_paths[o][d]
                if tp:
                    weights = np.array(
                        [freqs[r1] * freqs[r2] for r1, r2, _, _ in tp],
                        dtype=np.float64,
                    )
                    total_w = weights.sum()
                    if total_w < 1e-12:
                        total_w = 1.0
                    probs = weights / total_w

                    for idx, (r1, r2, tn, ivt_total) in enumerate(tp):
                        frac = probs[idx]
                        pax = dem * frac
                        demand_transfer += pax

                        total_ivt += pax * ivt_total

                        # Waiting on both legs
                        f1 = max(freqs[r1], MIN_FREQUENCY)
                        f2 = max(freqs[r2], MIN_FREQUENCY)
                        wait = 30.0 / f1 + 30.0 / f2
                        total_wait += pax * wait

                        # Transfer penalty
                        total_xfer_penalty += pax * TRANSFER_PENALTY

                        # Link flows on both legs
                        _accumulate_link_flow(
                            routes[r1], stop_indices[r1],
                            o, tn, pax, link_flows[r1]
                        )
                        _accumulate_link_flow(
                            routes[r2], stop_indices[r2],
                            tn, d, pax, link_flows[r2]
                        )
                    continue

                # --- Unmet demand ---
                demand_unmet += dem

        # --- Recompute frequencies from peak link flows ---
        for r_idx in range(n_routes):
            if not link_flows[r_idx]:
                freqs[r_idx] = MIN_FREQUENCY
                continue
            q_max = max(link_flows[r_idx].values())
            f_k = q_max / (LOAD_FACTOR * SEAT_CAPACITY)
            freqs[r_idx] = max(f_k, MIN_FREQUENCY)

        # Fleet constraint: rescale if needed
        fleet_per_route = np.zeros(n_routes)
        for r_idx in range(n_routes):
            T_k = route_round_trip_time(routes[r_idx])
            fleet_per_route[r_idx] = freqs[r_idx] * T_k / 60.0

        total_fleet = fleet_per_route.sum()
        if total_fleet > MAX_FLEET:
            scale = MAX_FLEET / total_fleet
            freqs *= scale
            # Ensure minimum frequency
            freqs = np.maximum(freqs, MIN_FREQUENCY)
            fleet_per_route = np.zeros(n_routes)
            for r_idx in range(n_routes):
                T_k = route_round_trip_time(routes[r_idx])
                fleet_per_route[r_idx] = freqs[r_idx] * T_k / 60.0

    # ------------------------------------------------------------------
    # Phase 3: Compute final metrics
    # ------------------------------------------------------------------
    total_fleet_final = int(np.ceil(fleet_per_route.sum()))
    total_fleet_final = min(total_fleet_final, MAX_FLEET)

    demand_served = demand_direct + demand_transfer
    d_sat = demand_served / TOTAL_DEMAND if TOTAL_DEMAND > 0 else 0.0

    total_travel_time = total_ivt + total_wait + total_xfer_penalty

    # Avoid division by zero in reward
    if total_travel_time < 1e-9:
        reward = 0.0
    else:
        reward = d_sat / total_travel_time

    metrics = {
        'd_sat': d_sat,
        'total_travel_time': total_travel_time,
        'in_vehicle_time': total_ivt,
        'waiting_time': total_wait,
        'transfer_time': total_xfer_penalty,
        'direct_pct': (demand_direct / TOTAL_DEMAND * 100) if TOTAL_DEMAND > 0 else 0.0,
        'transfer_pct': (demand_transfer / TOTAL_DEMAND * 100) if TOTAL_DEMAND > 0 else 0.0,
        'unmet_pct': (demand_unmet / TOTAL_DEMAND * 100) if TOTAL_DEMAND > 0 else 0.0,
        'fleet_size': total_fleet_final,
        'frequencies': freqs.tolist(),
        'reward': reward,
    }
    return metrics


def _accumulate_link_flow(route, idx_map, from_node, to_node, pax, flow_dict):
    """
    Add `pax` passengers flowing from `from_node` to `to_node` along `route`.
    Accumulates into `flow_dict` on each directed link segment.
    """
    i_idx = idx_map[from_node]
    j_idx = idx_map[to_node]
    if i_idx < j_idx:
        for k in range(i_idx, j_idx):
            flow_dict[(route[k], route[k + 1])] += pax
    else:
        for k in range(i_idx, j_idx, -1):
            flow_dict[(route[k], route[k - 1])] += pax


def _empty_metrics():
    """Return metrics for an empty route set."""
    return {
        'd_sat': 0.0,
        'total_travel_time': 0.0,
        'in_vehicle_time': 0.0,
        'waiting_time': 0.0,
        'transfer_time': 0.0,
        'direct_pct': 0.0,
        'transfer_pct': 0.0,
        'unmet_pct': 100.0,
        'fleet_size': 0,
        'frequencies': [],
        'reward': 0.0,
    }
