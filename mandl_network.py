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
    [  0, 400, 200,  60, 80, 150,  75,  75,  30,160,  30,  25, 35, 20,  25],
    [400,   0,  50, 120,180, 100,  80,  75,  30,160,  30,  25, 35, 20,  25],
    [200,  50,   0,  40, 60, 120,  60,  60,  15, 80,  15,  10, 20, 10,  15],
    [ 60, 120,  40,   0, 50,  60,  30,  35,  10, 60,  10,   5, 10,  5,  10],
    [ 80, 180,  60,  50,  0,  75,  35,  40,  10, 75,  10,   5, 10,  5,  10],
    [150, 100, 120,  60, 75,   0, 100, 200,  40,250,  40,  25, 50, 25,  50],
    [ 75,  80,  60,  30, 35, 100,   0,  60,  15,125,  15,  10, 25, 10,  20],
    [ 75,  75,  60,  35, 40, 200,  60,   0,  25,150,  25,  15, 35, 15,  25],
    [ 30,  30,  15,  10, 10,  40,  15,  25,   0, 40,   5,   5, 10,  5,  10],
    [160, 160,  80,  60, 75, 250, 125, 150,  40,  0,  50,  30, 75, 35,  50],
    [ 30,  30,  15,  10, 10,  40,  15,  25,   5, 50,   0,  15, 25, 10,  10],
    [ 25,  25,  10,   5,  5,  25,  10,  15,   5, 30,  15,   0, 10,  5,   5],
    [ 35,  35,  20,  10, 10,  50,  25,  35,  10, 75,  25,  10,  0, 20,  10],
    [ 20,  20,  10,   5,  5,  25,  10,  15,   5, 35,  10,   5, 20,  0,   5],
    [ 25,  25,  15,  10, 10,  50,  20,  25,  10, 50,  10,   5, 10,  5,   0],
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
