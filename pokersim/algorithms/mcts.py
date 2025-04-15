"""
Implementation of Monte Carlo Tree Search (MCTS) algorithm for poker.

This module provides an implementation of the MCTS algorithm adapted for imperfect
information games like poker. MCTS is a heuristic search algorithm that builds
a search tree using random simulations to evaluate nodes.
"""

import numpy as np
import random
import math
from typing import List, Dict, Tuple, Any, Optional, Set

from pokersim.game.state import GameState
from pokersim.game.spingo import SpinGoGame


class Node:
    """
    Node in the MCTS tree.
    
    Each node represents a game state and stores statistics about
    simulations that have passed through it.
    
    Attributes:
        game_state (GameState): The game state this node represents.
        parent (Optional[Node]): Parent node in the tree.
        children (Dict[Any, Node]): Child nodes keyed by action.
        visits (int): Number of times this node has been visited.
        value (float): Total value accumulated from simulations.
        player_id (int): ID of the player making the decision at this node.
        available_actions (List): List of legal actions from this state.
        untried_actions (List): Actions that have not been tried yet.
        is_terminal (bool): Whether this node represents a terminal state.
    """
    
    def __init__(self, game_state: GameState, parent: Optional['Node'] = None, 
                action_taken: Any = None, player_id: int = 0):
        """
        Initialize a node in the MCTS tree.
        
        Args:
            game_state (GameState): The game state this node represents.
            parent (Optional[Node], optional): Parent node. Defaults to None.
            action_taken (Any, optional): Action that led to this state. Defaults to None.
            player_id (int, optional): Player ID. Defaults to 0.
        """
        self.game_state = game_state
        self.parent = parent
        self.action_taken = action_taken
        self.children = {}  # Maps actions to child nodes
        self.visits = 0
        self.value = 0.0
        self.player_id = player_id
        
        # Get legal actions
        if not game_state.is_terminal():
            self.available_actions = game_state.get_legal_actions()
            self.untried_actions = game_state.get_legal_actions().copy()
        else:
            self.available_actions = []
            self.untried_actions = []
        
        self.is_terminal = game_state.is_terminal()
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible child nodes have been expanded."""
        return len(self.untried_actions) == 0
    
    def best_child(self, exploration_weight: float = 1.0) -> 'Node':
        """
        Select the best child node according to the UCT formula.
        
        Args:
            exploration_weight (float, optional): Exploration parameter. Defaults to 1.0.
        
        Returns:
            Node: The best child node.
        """
        # UCT (Upper Confidence Bound for Trees) formula
        def uct_score(child: 'Node') -> float:
            # Exploitation term: average value of child
            exploitation = child.value / child.visits if child.visits > 0 else 0
            
            # Exploration term: uncertainty in child's value
            exploration = exploration_weight * math.sqrt(
                2 * math.log(self.visits) / child.visits) if child.visits > 0 else float('inf')
            
            return exploitation + exploration
        
        # Return the child with the highest UCT score
        return max(self.children.values(), key=uct_score)
    
    def expand(self) -> 'Node':
        """
        Expand the tree by adding a new child node.
        
        Returns:
            Node: The newly created child node.
        """
        # Choose an untried action
        action = self.untried_actions.pop()
        
        # Create new game state by applying the action
        new_state = self.game_state.apply_action(action)
        
        # Create and store a new child node
        child = Node(
            game_state=new_state,
            parent=self,
            action_taken=action,
            player_id=new_state.current_player
        )
        
        self.children[action] = child
        return child
    
    def update(self, result: float) -> None:
        """
        Update node statistics with a simulation result.
        
        Args:
            result (float): The result of the simulation (+1 for win, 0 for loss).
        """
        self.visits += 1
        self.value += result


class MCTS:
    """
    Monte Carlo Tree Search algorithm for poker decision making.
    
    This implementation is adapted for imperfect information games and uses
    determinization (sampling of hidden information) to handle uncertainty.
    
    Attributes:
        num_simulations (int): Number of simulations to run per decision.
        exploration_weight (float): UCT exploration parameter.
        max_depth (int): Maximum depth to simulate.
        player_id (int): ID of the player this MCTS controls.
    """
    
    def __init__(self, player_id: int = 0, num_simulations: int = 1000, 
                exploration_weight: float = 1.0, max_depth: int = 50):
        """
        Initialize MCTS algorithm.
        
        Args:
            player_id (int, optional): Player ID. Defaults to 0.
            num_simulations (int, optional): Number of simulations. Defaults to 1000.
            exploration_weight (float, optional): Exploration parameter. Defaults to 1.0.
            max_depth (int, optional): Maximum simulation depth. Defaults to 50.
        """
        self.player_id = player_id
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth
        
        # Keep track of seen information
        self.cards_seen = set()
    
    def choose_action(self, game_state: GameState) -> Any:
        """
        Choose the best action using MCTS.
        
        Args:
            game_state (GameState): Current game state.
        
        Returns:
            Any: The chosen action.
        """
        # Create root node
        root = Node(game_state, player_id=self.player_id)
        
        # Update seen cards
        if hasattr(game_state, 'hole_cards') and game_state.hole_cards:
            for card in game_state.hole_cards.get(self.player_id, []):
                self.cards_seen.add(card)
        
        if hasattr(game_state, 'community_cards') and game_state.community_cards:
            for card in game_state.community_cards:
                self.cards_seen.add(card)
        
        # If only one legal action, return it immediately
        if len(root.available_actions) == 1:
            return root.available_actions[0]
        
        # Run simulations
        for _ in range(self.num_simulations):
            # Selection and expansion phase
            node = self._select_and_expand(root)
            
            # Simulation phase
            # For imperfect information games, we need to sample a possible state
            sampled_state = self._sample_determinization(node.game_state)
            result = self._simulate(sampled_state)
            
            # Backpropagation phase
            self._backpropagate(node, result)
        
        # Return the action with the most visits
        return max(root.children.keys(), 
                  key=lambda action: root.children[action].visits)
    
    def _select_and_expand(self, node: Node) -> Node:
        """
        Select a leaf node to expand using UCT.
        
        Args:
            node (Node): Starting node.
        
        Returns:
            Node: Selected node for expansion or simulation.
        """
        # Traverse the tree to find a node to expand
        while not node.is_terminal:
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.best_child(self.exploration_weight)
        
        return node
    
    def _sample_determinization(self, game_state: GameState) -> GameState:
        """
        Create a sampled determinization of the game state.
        
        For imperfect information games, this samples the hidden information
        (like opponent's cards) to create a concrete game state.
        
        Args:
            game_state (GameState): Current game state with hidden information.
        
        Returns:
            GameState: A sampled complete game state.
        """
        # This is a simplified version - in a real implementation, we'd need to:
        # 1. Get all known cards (player's hole cards, community cards)
        # 2. Sample possible opponent hole cards from remaining deck
        # 3. Create a new determinized game state
        
        # For now, we'll just clone the state
        return game_state.clone()
    
    def _simulate(self, game_state: GameState, max_depth: Optional[int] = None) -> float:
        """
        Run a random simulation from a state to a terminal state.
        
        Args:
            game_state (GameState): Starting game state.
            max_depth (Optional[int], optional): Maximum simulation depth. 
                                               Defaults to None.
        
        Returns:
            float: Simulation result from player's perspective.
        """
        if max_depth is None:
            max_depth = self.max_depth
        
        # Clone the state to avoid modifying the original
        current_state = game_state.clone()
        depth = 0
        
        # Simulate until terminal state or max depth
        while not current_state.is_terminal() and depth < max_depth:
            # Choose a random action
            legal_actions = current_state.get_legal_actions()
            action = random.choice(legal_actions)
            
            # Apply the action
            current_state = current_state.apply_action(action)
            depth += 1
        
        # Return the result
        if current_state.is_terminal():
            return current_state.get_utility(self.player_id)
        else:
            # If we reached max depth, use a heuristic evaluation
            return self._evaluate_heuristic(current_state)
    
    def _backpropagate(self, node: Node, result: float) -> None:
        """
        Backpropagate the simulation result up the tree.
        
        Args:
            node (Node): Leaf node where simulation started.
            result (float): Simulation result.
        """
        # Update statistics for all nodes in the path
        while node is not None:
            node.update(result)
            node = node.parent
    
    def _evaluate_heuristic(self, game_state: GameState) -> float:
        """
        Heuristic evaluation function for non-terminal states.
        
        Args:
            game_state (GameState): Game state to evaluate.
        
        Returns:
            float: Estimated value of the state.
        """
        # This is a simple heuristic - in practice, you'd want something more sophisticated
        # For example, you could use a trained value network to estimate the value
        
        # For now, just return a random value between -1 and 1
        return random.uniform(-1, 1)


class ISMCTSPlayer:
    """
    Information Set Monte Carlo Tree Search (ISMCTS) player for poker.
    
    This is an extension of MCTS that handles imperfect information games
    by reasoning about information sets rather than individual states.
    
    Attributes:
        player_id (int): Player ID.
        mcts (MCTS): MCTS algorithm instance.
    """
    
    def __init__(self, player_id: int, num_simulations: int = 1000):
        """
        Initialize ISMCTS player.
        
        Args:
            player_id (int): Player ID.
            num_simulations (int, optional): Number of simulations. Defaults to 1000.
        """
        self.player_id = player_id
        self.mcts = MCTS(
            player_id=player_id,
            num_simulations=num_simulations,
            exploration_weight=1.4  # Slightly more exploration
        )
    
    def act(self, game_state: GameState) -> Any:
        """
        Choose an action using ISMCTS.
        
        Args:
            game_state (GameState): Current game state.
        
        Returns:
            Any: The chosen action.
        """
        return self.mcts.choose_action(game_state)
    
    def observe_action(self, action: Any, player_id: int) -> None:
        """
        Observe an action taken by a player.
        
        This is used to update the player's information set.
        
        Args:
            action (Any): The action taken.
            player_id (int): ID of the player who took the action.
        """
        # Update internal state based on observed action
        pass