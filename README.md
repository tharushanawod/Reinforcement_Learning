# Bus Network Design and Frequency Setting (BNDFS) via Tabular Q-Learning

A Reinforcement Learning (RL) framework for solving the joint **Bus Network Design and Frequency Setting (BNDFS)** problem on the benchmark **Mandl Swiss Transit Network**, based on the methodology established by **Yoo, Lee, & Han (2023)**.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Description & Objectives](#problem-description--objectives)
- [How It Solves the Problem](#how-it-solves-the-problem)
  - [1. Markov Decision Process (MDP) Formulation](#1-markov-decision-process-mdp-formulation)
  - [2. Passenger Assignment & Frequency Setting](#2-passenger-assignment--frequency-setting)
  - [3. Tabular Q-Learning Agent](#3-tabular-q-learning-agent)
- [Project Structure](#project-structure)
- [Getting Started & How to Run](#getting-started--how-to-run)
  - [Prerequisites](#prerequisites)
  - [Running the Smoke Test](#running-the-smoke-test)
  - [Running Full Training](#running-full-training)
- [Evaluation Metrics & Benchmark Parameters](#evaluation-metrics--benchmark-parameters)
- [References](#references)

---

## Project Overview

The **Bus Network Design and Frequency Setting (BNDFS)** problem is an NP-hard combinatorial optimization challenge in public transit planning. It involves simultaneously determining:
1. **Network Topology / Routing:** Selecting transit paths (ordered sequences of adjacent physical road links) that connect origins and destinations across a city.
2. **Operational Frequencies:** Assigning departure frequencies (buses per hour) to each route to accommodate passenger demand while satisfying bus fleet constraints.

This repository implements a tabular Reinforcement Learning agent (Q-Learning) designed to iteratively build transit routes stop-by-stop and optimize the network for passenger satisfaction and travel time efficiency.

---

## Problem Description & Objectives

Transit network planning inherently balances conflicting operator and passenger interests:
- **Passenger Objective:** Maximize accessibility (served demand percentage $d_{\text{sat}}$) while minimizing total travel time (in-vehicle travel time, average waiting time at stops, and transfer delays).
- **Operator Constraints:** Operate within a finite fleet budget (maximum 99 buses), maintain vehicle seat capacity guidelines (40 seats with a 1.25 maximum load factor), and respect route length boundaries (2 to 8 stops per route).

The optimization objective is formalized through the reward function:
$$\text{Reward} = \frac{d_{\text{sat}}}{\text{Total Travel Time}}$$

Where:
- $d_{\text{sat}} = \frac{\text{Direct Demand} + \text{1-Transfer Demand}}{\text{Total Demand}}$
- $\text{Total Travel Time} = \text{In-Vehicle Time} + \text{Waiting Time} + \text{Transfer Penalty Time}$

---

## How It Solves the Problem

The framework decomposes the combinatorial search into a sequential decision-making process coupled with an iterative passenger assignment engine.

```
       +-------------------------------------------------------------+
       |                     QLearningAgent                          |
       |  - Epsilon-greedy exploration with linear decay             |
       |  - Priority tie-breaking: ADD > NEW > END                   |
       |  - Retroactive trajectory Q-updates                         |
       +-------------------------------------------------------------+
                          | Action                  ^ State,
                          v (ADD / NEW / END)       | Reward
       +-------------------------------------------------------------+
       |                    BNDFSEnvironment                         |
       |  - State: (completed_routes, current_route)                 |
       |  - Legal actions from network adjacency (no self-loops)     |
       +-------------------------------------------------------------+
                          | Evaluates routes
                          v
       +-------------------------------------------------------------+
       |                  Passenger Assignment                       |
       |  1. Find 0-transfer paths                                   |
       |  2. Find 1-transfer paths (10% travel-time tolerance)       |
       |  3. Assign demand proportionally to route frequencies       |
       |  4. Calculate peak link flows (Q_k,max)                     |
       |  5. Set frequencies & rescale to MAX_FLEET (99 buses)       |
       |  6. Return metrics & Reward = d_sat / Total_Travel_Time     |
       +-------------------------------------------------------------+
                          |
                          v
       +-------------------------------------------------------------+
       |                    Mandl Swiss Network                      |
       |  15 Nodes | 21 Road Links | 15,570 Total Passenger Demand   |
       +-------------------------------------------------------------+
```

### 1. Markov Decision Process (MDP) Formulation
Managed in `bndfs_environment.py`:
- **State Space:** Represented as `(completed_routes, current_route)` where `completed_routes` is an immutable tuple of finished routes, and `current_route` is the route currently under construction.
- **Action Space:**
  - `('ADD', node)`: Extends the active route to an unvisited adjacent road node (enforcing no repeated nodes).
  - `('NEW', start_node)`: Commits the active route (if it has $\ge 2$ stops) to `completed_routes` and initializes a new route starting at `start_node`.
  - `('END',)`: Finalizes the transit network design and triggers final evaluation.

### 2. Passenger Assignment & Frequency Setting
Implemented in `passenger_assignment.py`:
- **0-Transfer (Direct) Routing:** Passengers traveling between origin $o$ and destination $d$ are allocated to direct lines serving both stops in proportion to route service frequencies.
- **1-Transfer Routing:** Demand not served directly is routed via a single transfer stop $tn$ shared between two intersecting lines, provided the transfer in-vehicle time does not exceed $110\%$ of the shortest path distance ($\text{IVT} \le 1.10 \times \text{Shortest Path}$).
- **Frequency Setting:** Frequencies $f_k$ are determined by the peak directional link flow $Q_{k,\max}$:
  $$f_k = \frac{Q_{k,\max}}{\delta_{\max} \cdot C_k} = \frac{Q_{k,\max}}{1.25 \times 40} = \frac{Q_{k,\max}}{50}$$
- **Fleet Rescaling:** The total required fleet $N = \sum \frac{f_k \cdot T_k}{60}$ (where $T_k$ is route round-trip time in minutes) is constrained to $N \le 99$ buses. If exceeded, frequencies are rescaled proportionally while enforcing a floor of $1.0\text{ bus/hour}$.
- **Waiting & Transfer Penalties:** Average waiting time is modeled as $\frac{30}{f_k}$ minutes per leg. Each transfer adds a 5-minute penalty.

### 3. Tabular Q-Learning Agent
Implemented in `q_learning_agent.py`:
- **Linear $\epsilon$-Decay:**
  $$\epsilon_k = \frac{I_{\max} - I_k}{I_{\max}}$$
- **Priority Tie-Breaking:** When multiple actions share the optimal Q-value, actions are prioritized:
  $$\text{ADD (3)} > \text{NEW (2)} > \text{END (1)}$$
  This encourages building coherent, continuous routes before starting new lines or terminating.
- **Custom Non-Discounted Q-Update ($\gamma = 0$):**
  $$Q(s, a) \leftarrow \max(Q(s, a), r)$$
  Preserves the best historical reward found from state-action pair $(s, a)$.
- **Retroactive Trajectory Updates:** When an episode finishes with a positive final design reward, this reward is retroactively propagated backwards through all state-action pairs in the episode trajectory.
- **Multi-Replication Training:** Executes 5 independent replications of 10,000 episodes each to avoid local extrema and discover robust route topologies.

---

## Project Structure

```
Reinforcement_Learning/
│
├── mandl_network.py          # Mandl benchmark graph: 15 nodes, 21 bidirectional links,
│                             # 15x15 OD demand matrix (15,570 passengers), Floyd-Warshall paths.
│
├── passenger_assignment.py   # Passenger routing (0-transfer and 1-transfer), link flow
│                             # accumulation, frequency setting, fleet constraint, and metrics.
│
├── bndfs_environment.py      # RL Environment implementing state transitions, action masks,
│                             # route length constraints (min 2, max 8 stops), and reward calculation.
│
├── q_learning_agent.py       # Tabular Q-Learning agent with retroactive updates, tie-breaking,
│                             # epsilon-decay exploration, and replication tracking.
│
├── main.py                   # Main runner script: trains the agent across 5 replications x 10,000
│                             # iterations and prints detailed performance reports.
│
├── smoke_test.py             # Validation script testing network integrity, passenger routing,
│                             # route travel times, and basic environment transitions.
│
└── README.md                 # Project documentation and user guide.
```

---

## Getting Started & How to Run

### Prerequisites
- Python 3.8 or higher
- `numpy`

Install dependencies:
```bash
pip install numpy
```

### Running the Smoke Test
Verify environment mechanics, routing calculations, and network adjacency:
```bash
python smoke_test.py
```

Expected output:
- Topology verification (15 nodes, 21 links, 15,570 demand sum)
- Sample route evaluation with travel times and modal split
- Step-by-step environment transition test passing

### Running Full Training
To execute the complete Q-learning optimization process (5 replications $\times$ 10,000 iterations):
```bash
python main.py
```

During training, progress logs display the current replication, iteration, exploration rate ($\epsilon$), episode reward, satisfied demand percentage ($d_{\text{sat}}$), and elapsed time. At completion, a summary table reports the best network configuration found:
- Route topologies (1-indexed stops)
- Total travel time, in-vehicle time, waiting time, and transfer penalty time
- Direct (0-transfer), 1-transfer, and unmet demand percentages
- Required fleet size and per-route hourly frequencies

---

## Evaluation Metrics & Benchmark Parameters

### Mandl Swiss Network Characteristics
| Parameter | Value |
|---|---|
| Nodes (Stops) | 15 |
| Bidirectional Road Links | 21 |
| Total OD Demand | 15,570 passengers |
| Seating Capacity ($C_k$) | 40 seats |
| Max Load Factor ($\delta_{\max}$) | 1.25 (50 passengers/bus) |
| Max Fleet Size | 99 buses |
| Transfer Penalty | 5.0 minutes |
| Transfer Feasibility Tolerance | $\le 110\%$ of shortest path |
| Route Length Limit | 2 to 8 stops (expandable to 15) |

---

## References
- **Yoo, S., Lee, Y., & Han, K. (2023).** *Bus Network Design and Frequency Setting using Reinforcement Learning.*
- **Mandl, C. E. (1980).** *Evaluation and optimization of urban public transportation networks.* European Journal of Operational Research, 5(6), 396-404.
