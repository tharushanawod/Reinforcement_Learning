"""
q_learning_agent.py
====================
Tabular Q-Learning Agent for Bus Network Design and Frequency Setting.

Implements the exact Q-learning formulation from Yoo, Lee, & Han (2023):
  - Epsilon-greedy with linear decay
  - Custom Q-update rule (alpha_t = 1 if r > Q_old, else 0)
  - Retroactive trajectory update (replace lower Q-values with final reward)
  - Gamma = 0.0 (no discount)
  - Priority rule for tie-breaking among equal Q-values

Training: 5 independent replications × 10,000 iterations each.
"""

import random
import time
from collections import defaultdict
from bndfs_environment import BNDFSEnvironment
from passenger_assignment import evaluate_route_set
from mandl_network import NUM_NODES


class QLearningAgent:
    """
    Tabular Q-Learning agent for BNDFS.

    Parameters
    ----------
    max_stops_per_route : int
        Maximum stops per route (8 or 15).
    max_iterations : int
        Number of training iterations per replication.
    num_replications : int
        Number of independent replications.
    seed : int or None
        Base random seed.
    """

    # Action-type priority for tie-breaking (higher = preferred)
    ACTION_PRIORITY = {'ADD': 3, 'NEW': 2, 'END': 1}

    def __init__(
        self,
        max_stops_per_route=8,
        max_iterations=10000,
        num_replications=5,
        seed=None,
    ):
        self.max_stops = max_stops_per_route
        self.max_iterations = max_iterations
        self.num_replications = num_replications
        self.base_seed = seed

        # Q-table: dict of {state: {action: Q-value}}
        self.q_table = defaultdict(lambda: defaultdict(float))

        # Best design tracking
        self.best_reward = -1e9
        self.best_routes = None
        self.best_metrics = None
        self.best_replication = -1
        self.best_iteration = -1

    def _get_epsilon(self, iteration):
        """Linear epsilon decay: eps_k = (I_max - I_k) / I_max."""
        return max(0.0, (self.max_iterations - iteration) / self.max_iterations)

    def _select_action(self, state, available_actions, epsilon):
        """
        Epsilon-greedy action selection with priority tie-breaking.

        Parameters
        ----------
        state : hashable
            Current state.
        available_actions : list of tuples
            Legal actions.
        epsilon : float
            Exploration probability.

        Returns
        -------
        action : tuple
            Selected action.
        """
        if not available_actions:
            return None

        if random.random() < epsilon:
            # Exploration: random action
            return random.choice(available_actions)
        else:
            # Exploitation: pick action with highest Q-value, with tie-breaking
            q_vals = self.q_table[state]
            best_q = max(q_vals.get(a, 0.0) for a in available_actions)

            # Collect all actions with the best Q-value
            best_actions = [
                a for a in available_actions
                if abs(q_vals.get(a, 0.0) - best_q) < 1e-15
            ]

            # Priority tie-breaking: ADD > NEW > END
            best_actions.sort(
                key=lambda a: self.ACTION_PRIORITY.get(a[0], 0),
                reverse=True,
            )
            return best_actions[0]

    def _update_q(self, state, action, reward):
        """
        Q-value update rule (gamma = 0.0):
          Q_new = Q_old + alpha * [r - Q_old]
          where alpha = 1.0 if r > Q_old, else alpha = 0.0

        This preserves the maximum historical reward for each (s, a).
        """
        q_old = self.q_table[state][action]
        if reward > q_old:
            self.q_table[state][action] = reward

    def _update_end_q(self, state, reward):
        """
        Exception for END action: set Q(s, END) = reward.
        """
        self.q_table[state][('END',)] = reward

    def _retroactive_update(self, trajectory, final_reward):
        """
        Replace all lower Q-values in the episode trajectory with
        the final (higher) reward.
        """
        for state, action in trajectory:
            self._update_q(state, action, final_reward)

    def train(self):
        """
        Run the full training procedure:
          5 replications × 10,000 iterations each.

        Returns
        -------
        best_routes : list of list[int]
        best_metrics : dict
        """
        print("=" * 70)
        print("BNDFS Q-Learning Training")
        print(f"  Replications: {self.num_replications}")
        print(f"  Iterations per replication: {self.max_iterations}")
        print(f"  Max stops per route: {self.max_stops}")
        print("=" * 70)

        overall_start = time.time()

        for rep in range(self.num_replications):
            rep_start = time.time()

            # Reset Q-table for each independent replication
            self.q_table = defaultdict(lambda: defaultdict(float))

            if self.base_seed is not None:
                random.seed(self.base_seed + rep)

            rep_best_reward = -1e9
            rep_best_routes = None
            rep_best_metrics = None

            env = BNDFSEnvironment(
                max_stops_per_route=self.max_stops,
                min_stops_per_route=2,
                max_routes=20,
            )

            for iteration in range(self.max_iterations):
                epsilon = self._get_epsilon(iteration)
                state = env.reset()
                trajectory = []  # list of (state, action) pairs
                done = False
                episode_reward = 0.0

                # Safety limit to prevent infinite loops
                max_steps = NUM_NODES * 30
                step_count = 0

                while not done and step_count < max_steps:
                    step_count += 1
                    available_actions = env.get_available_actions()

                    if not available_actions:
                        break

                    action = self._select_action(state, available_actions, epsilon)
                    if action is None:
                        break

                    next_state, reward, done, info = env.step(action)

                    # Record trajectory
                    trajectory.append((state, action))

                    # Standard Q-update for non-END actions
                    if action[0] != 'END':
                        self._update_q(state, action, reward)
                    else:
                        # END action: Q(s, END) = reward
                        self._update_end_q(state, reward)

                    episode_reward = reward
                    state = next_state

                # Retroactive update: propagate final reward backward
                if episode_reward > 0:
                    self._retroactive_update(trajectory, episode_reward)

                # Track best design in this replication
                final_routes = env.get_final_routes()
                if episode_reward > rep_best_reward and final_routes:
                    rep_best_reward = episode_reward
                    rep_best_routes = [list(r) for r in final_routes]
                    rep_best_metrics = env.last_metrics

                # Progress logging
                if (iteration + 1) % 1000 == 0 or iteration == 0:
                    elapsed = time.time() - rep_start
                    d_sat = env.last_metrics['d_sat'] * 100 if env.last_metrics else 0
                    print(
                        f"  Rep {rep+1}/{self.num_replications} | "
                        f"Iter {iteration+1:>5}/{self.max_iterations} | "
                        f"eps={epsilon:.3f} | "
                        f"reward={episode_reward:.6e} | "
                        f"d_sat={d_sat:.1f}% | "
                        f"best_reward={rep_best_reward:.6e} | "
                        f"time={elapsed:.1f}s"
                    )

            # Update global best
            if rep_best_reward > self.best_reward and rep_best_routes:
                self.best_reward = rep_best_reward
                self.best_routes = rep_best_routes
                self.best_metrics = rep_best_metrics
                self.best_replication = rep + 1
                self.best_iteration = -1  # tracked during run

            rep_elapsed = time.time() - rep_start
            print(
                f"  Replication {rep+1} complete | "
                f"Best reward: {rep_best_reward:.6e} | "
                f"Time: {rep_elapsed:.1f}s"
            )
            print("-" * 70)

        total_elapsed = time.time() - overall_start
        print(f"\nTotal training time: {total_elapsed:.1f}s")
        print(f"Best design found in replication {self.best_replication}")

        return self.best_routes, self.best_metrics

    def get_best_design(self):
        """Return the best routes and metrics found during training."""
        return self.best_routes, self.best_metrics
