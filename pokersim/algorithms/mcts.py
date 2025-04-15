"""
Monte Carlo Tree Search (MCTS) algorithm for poker simulation.

This module implements the MCTS algorithm for poker, which uses random sampling
to build a search tree for decision making under uncertainty. The implementation
is adapted for imperfect information games like poker.
"""

import numpy as np
import random
import time
import math
from typing import Dict, List, Tuple, Any, Optional, Callable
import copy
import logging


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search.
    
    Attributes:
        state: The game state at this node.
        parent: The parent node.
        children: Dictionary of child nodes mapped by actions.
        visits: Number of times this node has been visited.
        value: Sum of values from rollouts through this node.
        untried_actions: List of actions not yet expanded.
        player_id: ID of the player at this node.
        action: Action that led to this node.
        is_terminal: Whether this node represents a terminal state.
        info_set: Information set identifier for this node.
    """
    
    def __init__(self, state: Any, parent: Optional['MCTSNode'] = None, 
               action: Any = None, player_id: int = 0):
        """
        Initialize a node in the MCTS tree.
        
        Args:
            state: Game state at this node.
            parent (Optional[MCTSNode], optional): Parent node. Defaults to None.
            action (Any, optional): Action leading to this node. Defaults to None.
            player_id (int, optional): Player ID at this node. Defaults to 0.
        """
        self.state = state
        self.parent = parent
        self.children = {}  # Action -> MCTSNode
        self.visits = 0
        self.value = 0.0
        self.player_id = player_id
        self.action = action
        
        # Get legal actions
        self.untried_actions = state.get_legal_actions() if not state.is_terminal() else []
        
        # Terminal state check
        self.is_terminal = state.is_terminal()
        
        # Information set identifier
        self.info_set = self._get_info_set_id()
    
    def _get_info_set_id(self) -> str:
        """
        Get an identifier for the information set of this node.
        
        For poker, this is based on the player's observation of the game,
        which includes their cards, visible community cards, and bet history.
        
        Returns:
            str: Information set identifier.
        """
        # Get observation from the player's perspective
        obs = self.state.get_observation(self.player_id)
        
        # Create a string representation of the observation
        info_set_parts = []
        
        # Add hole cards
        if 'hole_cards' in obs and obs['hole_cards']:
            hole_cards_str = ','.join(str(card) for card in obs['hole_cards'])
            info_set_parts.append(f"cards:{hole_cards_str}")
        
        # Add community cards
        if 'community_cards' in obs and obs['community_cards']:
            community_cards_str = ','.join(str(card) for card in obs['community_cards'])
            info_set_parts.append(f"community:{community_cards_str}")
        
        # Add stage
        if 'stage' in obs:
            info_set_parts.append(f"stage:{obs['stage']}")
        
        # Add action history (in a compressed form)
        if 'history' in obs and obs['history']:
            history_str = ';'.join(f"{p}:{a}" for p, a in obs['history'])
            info_set_parts.append(f"history:{history_str}")
        
        # Combine parts
        return "|".join(info_set_parts)
    
    def is_fully_expanded(self) -> bool:
        """
        Check if all possible actions from this node have been expanded.
        
        Returns:
            bool: Whether the node is fully expanded.
        """
        return len(self.untried_actions) == 0
    
    def select_child(self, exploration_weight: float = 1.0) -> 'MCTSNode':
        """
        Select a child node using the UCB1 formula.
        
        Args:
            exploration_weight (float, optional): Weight for exploration term. Defaults to 1.0.
        
        Returns:
            MCTSNode: Selected child node.
        """
        # UCB1 selection formula: v_i + C * sqrt(ln(N) / n_i)
        # where v_i is the value estimate, N is parent visits, n_i is child visits,
        # and C is the exploration weight
        
        log_visits = math.log(self.visits)
        
        def ucb_score(node: MCTSNode) -> float:
            # Avoid division by zero
            if node.visits == 0:
                return float('inf')
            
            # Calculate exploitation term (value estimate)
            exploitation = node.value / node.visits
            
            # Calculate exploration term
            exploration = exploration_weight * math.sqrt(log_visits / node.visits)
            
            return exploitation + exploration
        
        # Select child with highest UCB score
        return max(self.children.values(), key=ucb_score)
    
    def expand(self) -> Optional['MCTSNode']:
        """
        Expand the tree by adding a child node for an untried action.
        
        Returns:
            Optional[MCTSNode]: The new child node, or None if no untried actions.
        """
        if not self.untried_actions:
            return None
        
        # Choose an unexplored action
        action = self.untried_actions.pop()
        
        # Apply the action to get a new state
        new_state = self.state.apply_action(action)
        
        # Create a new node for the resulting state
        new_player_id = new_state.current_player
        child = MCTSNode(new_state, parent=self, action=action, player_id=new_player_id)
        
        # Add the new node to children
        self.children[action] = child
        
        return child
    
    def update(self, result: float) -> None:
        """
        Update the node statistics with a simulation result.
        
        Args:
            result (float): The result value to propagate up the tree.
        """
        self.visits += 1
        self.value += result
    
    def get_best_action(self, exploration: bool = False) -> Any:
        """
        Get the best action from this node.
        
        Args:
            exploration (bool, optional): Whether to consider exploration. Defaults to False.
        
        Returns:
            Any: The best action.
        """
        if not self.children:
            if self.untried_actions:
                return random.choice(self.untried_actions)
            return None
        
        if exploration:
            # Use UCB for selection during search
            return self.select_child().action
        else:
            # Use most visited child for final selection
            return max(self.children.items(), key=lambda x: x[1].visits)[0]
    
    def __str__(self) -> str:
        """String representation of the node."""
        return f"MCTSNode(visits={self.visits}, value={self.value:.2f}, actions={len(self.untried_actions)})"


class MCTSSolver:
    """
    Monte Carlo Tree Search solver for poker games.
    
    This class implements MCTS for poker decision making, with adaptations
    for imperfect information games.
    
    Attributes:
        max_iterations (int): Maximum number of MCTS iterations.
        max_time (float): Maximum time for search in seconds.
        exploration_weight (float): Exploration weight in UCB formula.
        random_rollout_depth (int): Maximum depth for random rollouts.
        info_set_grouping (bool): Whether to group nodes by information sets.
        info_set_nodes (Dict[str, List[MCTSNode]]): Nodes grouped by information sets.
    """
    
    def __init__(self, max_iterations: int = 1000, max_time: float = 5.0,
               exploration_weight: float = 1.0, random_rollout_depth: int = 10,
               info_set_grouping: bool = True):
        """
        Initialize the MCTS solver.
        
        Args:
            max_iterations (int, optional): Maximum iterations. Defaults to 1000.
            max_time (float, optional): Maximum search time in seconds. Defaults to 5.0.
            exploration_weight (float, optional): Exploration weight. Defaults to 1.0.
            random_rollout_depth (int, optional): Maximum rollout depth. Defaults to 10.
            info_set_grouping (bool, optional): Whether to group by info sets. Defaults to True.
        """
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.exploration_weight = exploration_weight
        self.random_rollout_depth = random_rollout_depth
        self.info_set_grouping = info_set_grouping
        
        # Information set grouping
        self.info_set_nodes = {}  # Info set -> List of nodes
        
        # Tree statistics
        self.total_nodes = 0
        self.max_depth = 0
    
    def search(self, root_state: Any, player_id: int) -> Tuple[Any, MCTSNode]:
        """
        Perform MCTS to find the best action.
        
        Args:
            root_state: The current game state.
            player_id (int): The player making the decision.
        
        Returns:
            Tuple[Any, MCTSNode]: The best action and the root node.
        """
        # Create root node
        root = MCTSNode(root_state, player_id=player_id)
        
        # Reset statistics
        self.total_nodes = 1
        self.max_depth = 0
        self.info_set_nodes = {}
        
        # Add root to info set grouping
        if self.info_set_grouping:
            info_set = root.info_set
            if info_set not in self.info_set_nodes:
                self.info_set_nodes[info_set] = []
            self.info_set_nodes[info_set].append(root)
        
        # MCTS iterations
        iterations = 0
        start_time = time.time()
        
        while (iterations < self.max_iterations and 
              time.time() - start_time < self.max_time):
            # Selection phase
            selected_node = self._select(root)
            
            # If node is terminal, use its rewards directly
            if selected_node.is_terminal:
                rewards = selected_node.state.get_rewards()
                result = rewards[player_id]
            else:
                # Expansion phase (if not terminal)
                if not selected_node.is_fully_expanded():
                    expanded_node = selected_node.expand()
                    if expanded_node:
                        selected_node = expanded_node
                        self.total_nodes += 1
                        
                        # Add to info set grouping
                        if self.info_set_grouping:
                            info_set = expanded_node.info_set
                            if info_set not in self.info_set_nodes:
                                self.info_set_nodes[info_set] = []
                            self.info_set_nodes[info_set].append(expanded_node)
                
                # Simulation phase
                result = self._simulate(selected_node, player_id)
            
            # Backpropagation phase
            self._backpropagate(selected_node, result)
            
            iterations += 1
        
        # Get statistics
        search_time = time.time() - start_time
        logging.debug(f"MCTS completed {iterations} iterations in {search_time:.3f}s")
        logging.debug(f"Total nodes: {self.total_nodes}, Info sets: {len(self.info_set_nodes)}")
        
        # Select best action
        best_action = root.get_best_action(exploration=False)
        
        return best_action, root
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a node for expansion using the tree policy.
        
        Args:
            node (MCTSNode): Starting node.
        
        Returns:
            MCTSNode: Selected node.
        """
        # Keep track of selection path depth
        depth = 0
        
        # Selection phase - traverse tree until we find a node that isn't fully expanded
        # or a terminal node
        while not node.is_terminal and node.is_fully_expanded():
            node = node.select_child(self.exploration_weight)
            depth += 1
        
        # Update max depth statistic
        self.max_depth = max(self.max_depth, depth)
        
        return node
    
    def _simulate(self, node: MCTSNode, player_id: int) -> float:
        """
        Simulate a game from the selected node to estimate the value.
        
        Args:
            node (MCTSNode): Node to simulate from.
            player_id (int): ID of the player from whose perspective to evaluate.
        
        Returns:
            float: Estimated value.
        """
        # Start with the current state
        state = copy.deepcopy(node.state)
        depth = 0
        
        # Rollout until terminal state or depth limit
        while not state.is_terminal() and depth < self.random_rollout_depth:
            # Get legal actions
            actions = state.get_legal_actions()
            
            if not actions:
                break
            
            # For simulation, choose action uniformly randomly
            action = random.choice(actions)
            
            # Apply action
            state = state.apply_action(action)
            depth += 1
        
        # Get rewards
        rewards = state.get_rewards()
        
        # Return the reward for the player of interest
        return rewards[player_id]
    
    def _backpropagate(self, node: MCTSNode, result: float) -> None:
        """
        Backpropagate the simulation result up the tree.
        
        Args:
            node (MCTSNode): Node to start backpropagation from.
            result (float): Simulation result.
        """
        # Update nodes on the path from the expanded node to the root
        while node:
            node.update(result)
            node = node.parent
    
    def get_action_probabilities(self, state: Any, player_id: int, temperature: float = 1.0) -> np.ndarray:
        """
        Get a probability distribution over actions based on MCTS visit counts.
        
        Args:
            state: The current game state.
            player_id (int): The player making the decision.
            temperature (float, optional): Temperature parameter for softmax. Defaults to 1.0.
        
        Returns:
            np.ndarray: Probability distribution over actions.
        """
        # Run MCTS
        _, root = self.search(state, player_id)
        
        # Get legal actions
        actions = state.get_legal_actions()
        
        # If no actions, return uniform distribution
        if not actions:
            return np.ones(1) / 1
        
        # Initialize counts
        counts = np.zeros(len(actions))
        
        # Get visit counts for each action
        for i, action in enumerate(actions):
            if action in root.children:
                counts[i] = root.children[action].visits
        
        # Apply temperature
        if temperature != 0:
            counts = np.power(counts, 1.0 / temperature)
        
        # Avoid division by zero
        if np.sum(counts) == 0:
            return np.ones(len(actions)) / len(actions)
        
        # Normalize to create probability distribution
        probs = counts / np.sum(counts)
        
        return probs
    
    def evaluate_against(self, player_state_fn: Callable, opponent_fn: Callable, 
                       num_games: int = 100) -> Dict[str, float]:
        """
        Evaluate MCTS against another agent or strategy.
        
        Args:
            player_state_fn: Function to create an initial state with MCTS as player.
            opponent_fn: Function to get opponent actions.
            num_games (int, optional): Number of games to play. Defaults to 100.
        
        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        wins = 0
        total_rewards = 0.0
        
        for _ in range(num_games):
            # Create initial state
            state, player_id = player_state_fn()
            
            # Play game
            while not state.is_terminal():
                current_player = state.current_player
                
                if current_player == player_id:
                    # MCTS player's turn
                    action, _ = self.search(state, player_id)
                else:
                    # Opponent's turn
                    action = opponent_fn(state, current_player)
                
                # Apply action
                state = state.apply_action(action)
            
            # Get rewards
            rewards = state.get_rewards()
            player_reward = rewards[player_id]
            
            # Update statistics
            total_rewards += player_reward
            if player_reward > 0:
                wins += 1
        
        # Calculate metrics
        win_rate = wins / num_games
        avg_reward = total_rewards / num_games
        
        return {
            'win_rate': win_rate,
            'avg_reward': avg_reward
        }
    
    def get_info_set_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about information sets.
        
        Returns:
            Dict[str, Dict[str, Any]]: Statistics for each information set.
        """
        statistics = {}
        
        for info_set, nodes in self.info_set_nodes.items():
            visits = sum(node.visits for node in nodes)
            avg_value = sum(node.value for node in nodes) / max(1, visits)
            
            statistics[info_set] = {
                'num_nodes': len(nodes),
                'total_visits': visits,
                'avg_value': avg_value,
                'max_depth': max(len(self._get_path_to_root(node)) for node in nodes)
            }
        
        return statistics
    
    def _get_path_to_root(self, node: MCTSNode) -> List[MCTSNode]:
        """
        Get the path from a node to the root.
        
        Args:
            node (MCTSNode): Starting node.
        
        Returns:
            List[MCTSNode]: Path to root.
        """
        path = []
        while node:
            path.append(node)
            node = node.parent
        return path