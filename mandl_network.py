"""
mandl_network.py
================
Mandl Swiss Network Benchmark Data (1980).

Contains the 15-node graph topology with 21 bidirectional road links,
direct link travel times, and the 15×15 OD demand matrix used in
Yoo, Lee, & Han (2023) for Bus Network Design and Frequency Setting.
"""

import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------------
# Number of stops / nodes
# ---------------------------------------------------------------------------
NUM_NODES = 15  # Nodes labelled 1..15 (we use 0..14 internally)

# ---------------------------------------------------------------------------
# Link travel times  (node_i, node_j, travel_time_in_minutes)
# Nodes are stored 0-indexed internally (subtract 1 from paper labels).
# ---------------------------------------------------------------------------
_LINKS_1INDEXED = [
    (1,  2,   8),
    (2,  3,   2),
    (2,  4,   3),
    (2,  5,   6),
    (3,  6,   3),
    (4,  5,   4),
    (4,  6,   4),
    (4, 12,  10),
    (6,  8,   2),
    (6, 15,   3),
    (7, 10,   7),
    (7, 15,   2),
    (8, 10,   8),
    (8, 15,   2),
    (9, 15,   8),
    (10, 11,  5),
    (10, 13, 10),
    (10, 14,  8),
    (11, 12, 10),
    (11, 13,  5),
    (13, 14,  2),
]

# Build adjacency structures (0-indexed)
LINK_TRAVEL_TIME = {}       # (i, j) -> travel time  (both directions)
ADJACENCY = defaultdict(set)  # node -> set of neighbours

for u, v, t in _LINKS_1INDEXED:
    u0, v0 = u - 1, v - 1
    LINK_TRAVEL_TIME[(u0, v0)] = t
    LINK_TRAVEL_TIME[(v0, u0)] = t
    ADJACENCY[u0].add(v0)
    ADJACENCY[v0].add(u0)

# Convert to regular dict for safety
ADJACENCY = dict(ADJACENCY)

# ---------------------------------------------------------------------------
# OD Demand Matrix (15×15, 0-indexed)
# From the standard Mandl benchmark.  Total demand = 15,570 passengers.
# Row = origin, Column = destination.  Diagonal = 0.
# ---------------------------------------------------------------------------
OD_DEMAND = np.array([
    [0,   400, 200, 60,  80, 150, 75, 75, 30, 160, 30, 25, 35,  0,  0],
    [400, 0,   50,  120, 20, 180, 90, 90, 15, 130, 20, 10, 10,  5,  0],
    [200, 50,  0,  40,  60, 180, 90, 90, 15, 45,  20, 10, 10,  5,  0],
    [60,  120, 40,  0,  50, 100, 50, 50, 15, 240, 40, 25, 10,  5,  0],
    [80,  20,  60,  50,  0,  50, 25, 25, 10, 120, 20, 15,  5,  0,  0],
    [150, 180, 180, 100, 50,  0, 100,100,30, 880, 60, 15, 15, 10,  0],
    [75,  90,  90,  50,  25, 100, 0,  50, 15, 440, 35, 10, 10,  5,  0],
    [75,  90,  90,  50,  25, 100, 50, 0,  15, 440, 35, 10, 10,  5,  0],
    [30,  15,  15,  15,  10, 30,  15, 15, 0,  140, 20,  5,  0,  0,  0],
    [160, 130, 45,  240, 120, 880, 440,440,140, 0, 600,250,500,200,  0],
    [30,  20,  20,  40,  20, 60,  35, 35, 20, 600, 0,  75, 95, 15,  0],
    [25,  10,  10,  25,  15, 15, 10, 10, 5,  250, 75, 0,  70,  0,  0],
    [35,  10,  10,  10,  5,  15, 10, 10, 0,  500, 95, 70, 0,  45,  0],
    [0,   5,   5,   5,   0,  10, 5,  5,  0,  200, 15,  0,  45, 0,  0],
    [0,   0,   0,   0,   0,   0,  0,  0,  0,  0,   0,  0,  0,  0,  0]
], dtype=np.float64)

TOTAL_DEMAND = OD_DEMAND.sum()  # Should be 15,570

# ---------------------------------------------------------------------------
# Shortest-path travel times (Floyd-Warshall) between all node pairs.
# Used for transfer-path feasibility checks (10% tolerance).
# ---------------------------------------------------------------------------
INF = 1e9

def compute_shortest_paths():
    """Return (dist, next_hop) matrices via Floyd-Warshall."""
    n = NUM_NODES
    dist = np.full((n, n), INF)
    np.fill_diagonal(dist, 0.0)
    for (u, v), t in LINK_TRAVEL_TIME.items():
        dist[u, v] = t
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    return dist

SHORTEST_PATH_DIST = compute_shortest_paths()
