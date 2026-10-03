"""
bndfs_environment.py
====================
Reinforcement Learning Environment for Bus Network Design
and Frequency Setting (BNDFS).

Implements the state space, action space, transitions, and reward
as specified in Yoo, Lee, & Han (2023).

State:  (routes_so_far, current_route_being_built)
Action: add_stop | new_route(start_node) | END
"""

from mandl_network import NUM_NODES, ADJACENCY, LINK_TRAVEL_TIME
from passenger_assignment import evaluate_route_set


class BNDFSEnvironment:
    """
    BNDFS Environment for tabular Q-learning.

    Parameters
    ----------
    max_stops_per_route : int
        Maximum number of stops allowed per single route (8 or 15).
    min_stops_per_route : int
        Minimum number of stops per route (default 2).
    max_routes : int
        Maximum number of routes in a design (soft limit for safety).
    """

    def __init__(self, max_stops_per_route=8, min_stops_per_route=2, max_routes=20):
        self.max_stops = max_stops_per_route
        self.min_stops = min_stops_per_route
        self.max_routes = max_routes

    def reset(self):
        """
        Reset environment to initial state.

        Returns
        -------
        state : tuple
            Hashable state representation.
        """
        self.completed_routes = []      # list of completed route tuples
        self.current_route = []         # route currently under construction
        self.done = False
        self.last_metrics = None
        return self._get_state()

    def _get_state(self):
        """
        Return a hashable state representation.

        State = (tuple of completed route tuples, tuple of current route)
        """
        completed = tuple(tuple(r) for r in self.completed_routes)
        current = tuple(self.current_route)
        return (completed, current)

    def get_available_actions(self):
        """
        Enumerate all legal actions from the current state.

        Returns
        -------
        actions : list of tuples
            Each action is one of:
              ('ADD', node)         - add adjacent node to current route
              ('NEW', start_node)   - finish current route, start new one at start_node
              ('END',)              - terminate design iteration
        """
        actions = []

        if self.done:
            return actions

        # --- Actions to ADD a stop to the current route ---
        if self.current_route:
            if len(self.current_route) < self.max_stops:
                last_node = self.current_route[-1]
                for neighbor in sorted(ADJACENCY.get(last_node, [])):
                    # Avoid revisiting a stop already in the current route
                    if neighbor not in self.current_route:
                        actions.append(('ADD', neighbor))

        # --- Actions to start a NEW route ---
        # Current route must meet minimum length to be "finished",
        # OR current route is empty (we're choosing the very first start node).
        can_finish_current = (
            len(self.current_route) == 0 or
            len(self.current_route) >= self.min_stops
        )

        if can_finish_current:
            if len(self.current_route) == 0 and len(self.completed_routes) == 0:
                # Very start: pick any node to begin the first route
                for node in range(NUM_NODES):
                    actions.append(('NEW', node))
            elif len(self.current_route) >= self.min_stops:
                # Finish current route and start a new one
                total_routes = len(self.completed_routes) + 1  # +1 for the one we'd finish
                if total_routes < self.max_routes:
                    for node in range(NUM_NODES):
                        actions.append(('NEW', node))

        # --- END action ---
        # Can only END if we have at least 1 completed route or current route is valid
        has_valid_design = (
            len(self.completed_routes) > 0 or
            len(self.current_route) >= self.min_stops
        )
        if has_valid_design:
            actions.append(('END',))

        return actions

    def step(self, action):
        """
        Take an action, transition to next state.

        Parameters
        ----------
        action : tuple
            One of ('ADD', node), ('NEW', start_node), ('END',).

        Returns
        -------
        next_state : tuple
        reward : float
        done : bool
        info : dict
        """
        assert not self.done, "Environment is done. Call reset()."

        action_type = action[0]

        if action_type == 'ADD':
            node = action[1]
            self.current_route.append(node)
            # Evaluate current design to get intermediate reward
            all_routes = self._all_routes()
            metrics = evaluate_route_set(all_routes)
            self.last_metrics = metrics
            reward = metrics['reward']
            return self._get_state(), reward, False, metrics

        elif action_type == 'NEW':
            start_node = action[1]
            # Finish current route if it has enough stops
            if len(self.current_route) >= self.min_stops:
                self.completed_routes.append(list(self.current_route))
            # Start new route
            self.current_route = [start_node]
            # Evaluate
            all_routes = self._all_routes()
            metrics = evaluate_route_set(all_routes)
            self.last_metrics = metrics
            reward = metrics['reward']
            return self._get_state(), reward, False, metrics

        elif action_type == 'END':
            # Finish current route if valid
            if len(self.current_route) >= self.min_stops:
                self.completed_routes.append(list(self.current_route))
                self.current_route = []
            self.done = True
            # Final evaluation
            all_routes = [list(r) for r in self.completed_routes]
            metrics = evaluate_route_set(all_routes)
            self.last_metrics = metrics
            reward = metrics['reward']
            return self._get_state(), reward, True, metrics

        else:
            raise ValueError(f"Unknown action type: {action_type}")

    def _all_routes(self):
        """Get all routes including the one under construction (if >= 2 stops)."""
        routes = [list(r) for r in self.completed_routes]
        if len(self.current_route) >= self.min_stops:
            routes.append(list(self.current_route))
        return routes

    def get_final_routes(self):
        """Return the final set of completed routes."""
        return [list(r) for r in self.completed_routes]
