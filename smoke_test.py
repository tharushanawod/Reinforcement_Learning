"""
Quick smoke test: verify the environment, passenger assignment,
and Q-learning agent work correctly with minimal iterations.
"""
import sys
sys.path.insert(0, r"e:\Research Experiments\Reinforcement_Learning")

from mandl_network import (
    NUM_NODES, TOTAL_DEMAND, OD_DEMAND, ADJACENCY,
    LINK_TRAVEL_TIME, SHORTEST_PATH_DIST,
)
from passenger_assignment import evaluate_route_set, route_travel_time
from bndfs_environment import BNDFSEnvironment

print("=" * 60)
print("SMOKE TEST: Mandl Network Data")
print("=" * 60)
print(f"  Nodes: {NUM_NODES}")
print(f"  Links: {len(LINK_TRAVEL_TIME) // 2}")
print(f"  Total OD demand: {TOTAL_DEMAND:.0f}")
print(f"  OD matrix shape: {OD_DEMAND.shape}")

# Verify adjacency
print(f"\n  Adjacency check:")
for n in range(NUM_NODES):
    neighbors = sorted([x+1 for x in ADJACENCY.get(n, [])])
    print(f"    Node {n+1}: neighbors = {neighbors}")

# Test a known route
print("\n" + "=" * 60)
print("SMOKE TEST: Route Evaluation")
print("=" * 60)
# A simple 2-route design
route1 = [0, 1, 2, 5, 14, 6, 9, 10]   # 1-2-3-6-15-7-10-11
route2 = [0, 1, 3, 4, 1, 2]            # invalid: has repeated node
# Use a valid route instead
route2_valid = [0, 1, 3, 5, 7]         # 1-2-4-6-8

# Check travel times
tt1 = route_travel_time(route1)
print(f"  Route 1 one-way travel time: {tt1} min")
print(f"  Route 1: {[n+1 for n in route1]}")

tt2 = route_travel_time(route2_valid)
print(f"  Route 2 one-way travel time: {tt2} min")
print(f"  Route 2: {[n+1 for n in route2_valid]}")

# Evaluate
metrics = evaluate_route_set([route1, route2_valid])
print(f"\n  Evaluation results:")
print(f"    d_sat:         {metrics['d_sat']*100:.2f}%")
print(f"    Direct:        {metrics['direct_pct']:.2f}%")
print(f"    Transfer:      {metrics['transfer_pct']:.2f}%")
print(f"    Unmet:         {metrics['unmet_pct']:.2f}%")
print(f"    Total TT:      {metrics['total_travel_time']:.2f} min")
print(f"    Fleet:         {metrics['fleet_size']} buses")
print(f"    Reward:        {metrics['reward']:.6e}")

# Test environment
print("\n" + "=" * 60)
print("SMOKE TEST: Environment Step")
print("=" * 60)
env = BNDFSEnvironment(max_stops_per_route=8)
state = env.reset()
print(f"  Initial state: {state}")
actions = env.get_available_actions()
print(f"  Available actions ({len(actions)}): first 5 = {actions[:5]}")

# Take a few steps
state, r, done, info = env.step(('NEW', 0))  # Start route at node 0
print(f"  After NEW(0): state={state}, actions={len(env.get_available_actions())}")

state, r, done, info = env.step(('ADD', 1))
print(f"  After ADD(1): reward={r:.6e}")

state, r, done, info = env.step(('ADD', 2))
print(f"  After ADD(2): reward={r:.6e}, d_sat={info['d_sat']*100:.1f}%")

state, r, done, info = env.step(('ADD', 5))
print(f"  After ADD(5): reward={r:.6e}, d_sat={info['d_sat']*100:.1f}%")

state, r, done, info = env.step(('END',))
print(f"  After END: reward={r:.6e}, done={done}, d_sat={info['d_sat']*100:.1f}%")
print(f"  Final routes: {[[n+1 for n in r] for r in env.get_final_routes()]}")

print("\n  Shortest path from node 1 to node 14:")
print(f"    Distance = {SHORTEST_PATH_DIST[0, 13]:.1f} min")

print("\n  ALL SMOKE TESTS PASSED!")
