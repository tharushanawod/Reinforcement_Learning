"""
main.py
=======
Main execution script for BNDFS Q-Learning.

Trains the Tabular Q-Learning agent on the Mandl Swiss Network benchmark
and prints final performance metrics.

Reference: Yoo, Lee, & Han (2023) - Bus Network Design and Frequency Setting
           using Reinforcement Learning.
"""

import sys
import time
import numpy as np

from mandl_network import NUM_NODES, TOTAL_DEMAND, OD_DEMAND
from passenger_assignment import evaluate_route_set
from q_learning_agent import QLearningAgent


def print_route_set(routes):
    """Pretty-print a route set using 1-indexed node labels."""
    print("\n  Route Set:")
    for i, route in enumerate(routes):
        route_1idx = [n + 1 for n in route]
        print(f"    Route {i+1}: {' -> '.join(map(str, route_1idx))}  "
              f"({len(route)} stops)")


def print_metrics(metrics, routes):
    """Print comprehensive performance metrics."""
    print("\n" + "=" * 70)
    print("FINAL PERFORMANCE METRICS")
    print("=" * 70)

    print_route_set(routes)

    print(f"\n  Number of Routes:        {len(routes)}")
    print(f"  Total OD Demand:         {TOTAL_DEMAND:.0f} passengers")

    print(f"\n  --- Travel Time Breakdown ---")
    print(f"  Total Travel Time:       {metrics['total_travel_time']:>12.2f} min")
    print(f"  In-Vehicle Time:         {metrics['in_vehicle_time']:>12.2f} min")
    print(f"  Waiting Time:            {metrics['waiting_time']:>12.2f} min")
    print(f"  Transfer Penalty Time:   {metrics['transfer_time']:>12.2f} min")

    print(f"\n  --- Demand Coverage ---")
    print(f"  Direct Trip (0-transfer): {metrics['direct_pct']:>8.2f}%")
    print(f"  1-Transfer Trip:          {metrics['transfer_pct']:>8.2f}%")
    print(f"  Unmet Demand:             {metrics['unmet_pct']:>8.2f}%")
    print(f"  Demand Satisfied (d_sat): {metrics['d_sat']*100:>8.2f}%")

    print(f"\n  --- Fleet ---")
    print(f"  Required Fleet Size:     {metrics['fleet_size']} buses")

    if metrics['frequencies']:
        print(f"\n  --- Route Frequencies (buses/hour) ---")
        for i, f in enumerate(metrics['frequencies']):
            print(f"    Route {i+1}: {f:.2f} buses/hour")

    print(f"\n  --- Reward ---")
    print(f"  Reward (d_sat / TT):     {metrics['reward']:.6e}")
    print("=" * 70)


def main():
    """
    Main entry point: Train the Q-learning agent and report results.

    Configuration:
      - 5 replications × 10,000 iterations each
      - Max 8 stops per route (standard Mandl benchmark)
      - Random seed = 42 for reproducibility
    """
    print("=" * 70)
    print("  BNDFS: Bus Network Design & Frequency Setting")
    print("  Method: Tabular Q-Learning (Yoo, Lee, & Han 2023)")
    print("  Benchmark: Mandl Swiss Network (15 nodes, 21 links)")
    print(f"  Total OD Demand: {TOTAL_DEMAND:.0f} passengers")
    print("=" * 70)

    # --- Validate network data ---
    print(f"\n  OD Matrix sum verification: {OD_DEMAND.sum():.0f}")
    assert abs(OD_DEMAND.sum() - TOTAL_DEMAND) < 1, "OD demand mismatch!"

    # --- Training configuration ---
    MAX_STOPS = 8
    MAX_ITERATIONS = 10000
    NUM_REPLICATIONS = 5
    SEED = 42

    # --- Initialize and train agent ---
    agent = QLearningAgent(
        max_stops_per_route=MAX_STOPS,
        max_iterations=MAX_ITERATIONS,
        num_replications=NUM_REPLICATIONS,
        seed=SEED,
    )

    print("\nStarting training...\n")
    best_routes, best_metrics = agent.train()

    # --- Re-evaluate the best design for clean metrics ---
    if best_routes:
        final_metrics = evaluate_route_set(best_routes)
        print_metrics(final_metrics, best_routes)
    else:
        print("\nWARNING: No valid route design was found during training.")
        print("Try increasing the number of iterations or adjusting parameters.")

    return best_routes, best_metrics


if __name__ == "__main__":
    main()
