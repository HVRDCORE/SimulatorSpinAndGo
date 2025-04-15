"""
Implementation of Counterfactual Regret Minimization (CFR) algorithm for poker.

This module provides a general-purpose implementation of the CFR algorithm
for computing approximate Nash equilibrium strategies in extensive-form games
like poker. CFR is one of the most successful algorithms for solving large
imperfect information games.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict

from pokersim.game.state import GameState
from pokersim.game.spingo import SpinGoGame


class InformationSet:
    """
    Information set for a player in an imperfect information game.
    
    An information set represents states that are indistinguishable to a player
    due to hidden information (e.g., opponent's cards in poker).
    
    Attributes:
        key (str): Unique identifier for this information set.
        num_actions (int): Number of available actions in this information set.
        regret_sum (np.ndarray): Accumulated regrets for each action.
        strategy_sum (np.ndarray): Accumulated strategy probabilities for each action.
        current_strategy (np.ndarray): Current strategy probabilities.
    """
    
    def __init__(self, key: str, num_actions: int):
        """
        Initialize an information set.
        
        Args:
            key (str): Unique identifier for this information set.
            num_actions (int): Number of available actions in this information set.
        """
        self.key = key
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.current_strategy = np.zeros(num_actions)
        
        # Initialize to uniform strategy
        self.current_strategy.fill(1.0 / num_actions)
    
    def get_strategy(self, reach_prob: float) -> np.ndarray:
        """
        Get the current strategy for this information set.
        
        Args:
            reach_prob (float): The probability of reaching this information set.
        
        Returns:
            np.ndarray: Strategy probabilities for each action.
        """
        # Calculate strategy using regret matching
        strategy = np.maximum(self.regret_sum, 0)
        total = np.sum(strategy)
        
        if total > 0:
            strategy /= total
        else:
            # If all regrets are non-positive, use uniform strategy
            strategy.fill(1.0 / self.num_actions)
        
        # Accumulate the strategy weighted by reach probability
        self.strategy_sum += reach_prob * strategy
        self.current_strategy = strategy
        
        return strategy
    
    def get_average_strategy(self) -> np.ndarray:
        """
        Get the average strategy over all iterations.
        
        Returns:
            np.ndarray: Average strategy probabilities for each action.
        """
        avg_strategy = np.zeros(self.num_actions)
        total = np.sum(self.strategy_sum)
        
        if total > 0:
            avg_strategy = self.strategy_sum / total
        else:
            # If no accumulated strategy, use uniform
            avg_strategy.fill(1.0 / self.num_actions)
        
        return avg_strategy
    
    def update_regrets(self, action_utils: np.ndarray, node_util: float) -> None:
        """
        Update the accumulated regrets.
        
        Args:
            action_utils (np.ndarray): Utility of each action.
            node_util (float): Actual utility of the node.
        """
        self.regret_sum += action_utils - node_util


class CFR:
    """
    Counterfactual Regret Minimization algorithm.
    
    This class implements the CFR algorithm for computing approximate Nash
    equilibrium strategies in extensive-form games like poker.
    
    Attributes:
        game_type (str): Type of game ('holdem', 'spingo', etc.).
        num_players (int): Number of players in the game.
        info_sets (Dict[str, InformationSet]): Information sets for all players.
        iterations (int): Number of iterations performed.
    """
    
    def __init__(self, game_type: str = 'spingo', num_players: int = 3):
        """
        Initialize the CFR algorithm.
        
        Args:
            game_type (str, optional): Type of game. Defaults to 'spingo'.
            num_players (int, optional): Number of players. Defaults to 3.
        """
        self.game_type = game_type
        self.num_players = num_players
        self.info_sets = {}  # Maps info set keys to InformationSet objects
        self.iterations = 0
    
    def train(self, num_iterations: int = 1000, pruning: bool = True) -> None:
        """
        Run CFR training for a specified number of iterations.
        
        Args:
            num_iterations (int, optional): Number of iterations. Defaults to 1000.
            pruning (bool, optional): Whether to use pruning. Defaults to True.
        """
        utility = 0.0
        
        for i in range(num_iterations):
            # Create a new game state
            if self.game_type == 'spingo':
                game = SpinGoGame(num_players=self.num_players)
                game_state = game.start_new_hand()
            else:
                game_state = GameState(num_players=self.num_players)
                game_state.deal_hole_cards()
            
            # Run CFR for each player
            for p in range(self.num_players):
                # Initialize reach probabilities
                reach_probs = np.ones(self.num_players)
                
                if pruning:
                    utility += self._cfr_with_pruning(game_state, p, reach_probs)
                else:
                    utility += self._cfr(game_state, p, reach_probs)
            
            self.iterations += 1
            
            # Optionally print progress
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1} iterations, average game value: {utility / (i + 1)}")
    
    def _cfr(self, game_state: GameState, traverser: int, reach_probs: np.ndarray) -> float:
        """
        Run one iteration of the CFR algorithm.
        
        Args:
            game_state (GameState): Current game state.
            traverser (int): Player ID of the traverser.
            reach_probs (np.ndarray): Reach probabilities for each player.
        
        Returns:
            float: Expected utility for the traverser.
        """
        # Return payoff at terminal states
        if game_state.is_terminal():
            return game_state.get_utility(traverser)
        
        current_player = game_state.current_player
        
        # Chance node (e.g., dealing cards)
        if current_player == -1:  # Convention for chance player
            outcome_probs = game_state.get_chance_probabilities()
            utility = 0.0
            
            for outcome, prob in outcome_probs.items():
                next_state = game_state.apply_chance_outcome(outcome)
                utility += prob * self._cfr(next_state, traverser, reach_probs)
            
            return utility
        
        # Get information set key for the current player
        info_set_key = self._get_info_set_key(game_state, current_player)
        legal_actions = game_state.get_legal_actions()
        num_actions = len(legal_actions)
        
        # Create a new information set if we haven't seen this one before
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = InformationSet(info_set_key, num_actions)
        
        info_set = self.info_sets[info_set_key]
        
        # Get strategy for this information set
        strategy = info_set.get_strategy(reach_probs[current_player])
        
        # Compute utility for each action
        action_utils = np.zeros(num_actions)
        
        for a, action in enumerate(legal_actions):
            # Create new reach probabilities
            new_reach_probs = reach_probs.copy()
            new_reach_probs[current_player] *= strategy[a]
            
            # Recursive call
            next_state = game_state.apply_action(action)
            action_utils[a] = self._cfr(next_state, traverser, new_reach_probs)
        
        # Compute expected utility
        node_util = np.sum(strategy * action_utils)
        
        # Update regrets if this is the traversing player
        if current_player == traverser:
            # Calculate counterfactual regrets
            info_set.update_regrets(action_utils, node_util)
        
        return node_util
    
    def _cfr_with_pruning(self, game_state: GameState, traverser: int, 
                         reach_probs: np.ndarray) -> float:
        """
        Run one iteration of CFR with regret-based pruning.
        
        This version of CFR skips updating paths that are unlikely to be reached,
        which can significantly speed up the algorithm.
        
        Args:
            game_state (GameState): Current game state.
            traverser (int): Player ID of the traverser.
            reach_probs (np.ndarray): Reach probabilities for each player.
        
        Returns:
            float: Expected utility for the traverser.
        """
        # Return payoff at terminal states
        if game_state.is_terminal():
            return game_state.get_utility(traverser)
        
        current_player = game_state.current_player
        
        # Chance node (e.g., dealing cards)
        if current_player == -1:  # Convention for chance player
            outcome_probs = game_state.get_chance_probabilities()
            utility = 0.0
            
            for outcome, prob in outcome_probs.items():
                next_state = game_state.apply_chance_outcome(outcome)
                utility += prob * self._cfr_with_pruning(next_state, traverser, reach_probs)
            
            return utility
        
        # Check if we can prune this subtree
        if current_player != traverser:
            # Calculate probability of reaching this node for the traverser
            traverser_reach = reach_probs[traverser]
            
            # Prune if the probability is too low
            if traverser_reach < 1e-8:  # Pruning threshold
                return 0.0
        
        # Get information set key for the current player
        info_set_key = self._get_info_set_key(game_state, current_player)
        legal_actions = game_state.get_legal_actions()
        num_actions = len(legal_actions)
        
        # Create a new information set if we haven't seen this one before
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = InformationSet(info_set_key, num_actions)
        
        info_set = self.info_sets[info_set_key]
        
        # Get strategy for this information set
        strategy = info_set.get_strategy(reach_probs[current_player])
        
        # Compute utility for each action
        action_utils = np.zeros(num_actions)
        
        for a, action in enumerate(legal_actions):
            # Skip actions with very low probability for non-traversers
            if current_player != traverser and strategy[a] < 1e-8:
                continue
            
            # Create new reach probabilities
            new_reach_probs = reach_probs.copy()
            new_reach_probs[current_player] *= strategy[a]
            
            # Recursive call
            next_state = game_state.apply_action(action)
            action_utils[a] = self._cfr_with_pruning(next_state, traverser, new_reach_probs)
        
        # Compute expected utility
        node_util = np.sum(strategy * action_utils)
        
        # Update regrets if this is the traversing player
        if current_player == traverser:
            # Calculate counterfactual regrets
            info_set.update_regrets(action_utils, node_util)
        
        return node_util
    
    def _get_info_set_key(self, game_state: GameState, player_id: int) -> str:
        """
        Generate a unique key for an information set.
        
        This key should encapsulate all information visible to the player.
        
        Args:
            game_state (GameState): Current game state.
            player_id (int): Player ID.
        
        Returns:
            str: A unique identifier for the information set.
        """
        # This is a simplified implementation - in practice, you would want
        # to include all visible information:
        #   - Player's hole cards
        #   - Community cards
        #   - Betting history
        #   - Player positions
        
        # Get visible cards
        visible_cards = []
        
        if hasattr(game_state, 'hole_cards') and game_state.hole_cards:
            if player_id in game_state.hole_cards:
                visible_cards.extend(sorted(game_state.hole_cards[player_id]))
        
        if hasattr(game_state, 'community_cards') and game_state.community_cards:
            visible_cards.extend(sorted(game_state.community_cards))
        
        # Get betting history
        betting_history = []
        if hasattr(game_state, 'betting_history'):
            betting_history = game_state.betting_history
        
        # Combine all information into a string
        key_parts = [
            f"P{player_id}",
            f"Cards:{'-'.join(str(c) for c in visible_cards)}",
            f"History:{'-'.join(str(a) for a in betting_history)}",
            f"Stage:{game_state.stage if hasattr(game_state, 'stage') else 'Unknown'}"
        ]
        
        return "|".join(key_parts)
    
    def get_strategy(self, game_state: GameState, player_id: int) -> Dict[Any, float]:
        """
        Get the strategy for a player in a given game state.
        
        Args:
            game_state (GameState): Current game state.
            player_id (int): Player ID.
        
        Returns:
            Dict[Any, float]: Mapping from actions to probabilities.
        """
        # Get information set key
        info_set_key = self._get_info_set_key(game_state, player_id)
        
        # If we haven't seen this information set, return uniform strategy
        if info_set_key not in self.info_sets:
            legal_actions = game_state.get_legal_actions()
            probs = np.ones(len(legal_actions)) / len(legal_actions)
            return {action: prob for action, prob in zip(legal_actions, probs)}
        
        # Get the average strategy for this information set
        info_set = self.info_sets[info_set_key]
        avg_strategy = info_set.get_average_strategy()
        
        # Map strategy probabilities to actions
        legal_actions = game_state.get_legal_actions()
        return {action: prob for action, prob in zip(legal_actions, avg_strategy)}
    
    def act(self, game_state: GameState, player_id: int) -> Any:
        """
        Choose an action using the learned strategy.
        
        Args:
            game_state (GameState): Current game state.
            player_id (int): Player ID.
        
        Returns:
            Any: The chosen action.
        """
        # Get strategy for this state
        strategy = self.get_strategy(game_state, player_id)
        
        # Get legal actions and their probabilities
        actions = list(strategy.keys())
        probs = list(strategy.values())
        
        # Choose an action according to the strategy
        return np.random.choice(actions, p=probs)
    
    def evaluate(self, num_games: int = 100, opponent_type: str = 'rule_based') -> Dict[str, Any]:
        """
        Evaluate the agent against opponents.
        
        Args:
            num_games (int, optional): Number of games to play. Defaults to 100.
            opponent_type (str, optional): Type of opponents. Defaults to 'rule_based'.
            
        Returns:
            Dict[str, Any]: Evaluation metrics.
        """
        # Placeholder for evaluation logic
        results = {
            'win_rate': 0.0,
            'avg_utility': 0.0,
            'games_played': num_games
        }
        
        return results


class CFRPlayer:
    """
    Player that uses CFR to make decisions.
    
    Attributes:
        player_id (int): Player ID.
        cfr (CFR): CFR algorithm instance.
    """
    
    def __init__(self, player_id: int, cfr: CFR):
        """
        Initialize CFR player.
        
        Args:
            player_id (int): Player ID.
            cfr (CFR): Trained CFR instance.
        """
        self.player_id = player_id
        self.cfr = cfr
    
    def act(self, game_state: GameState) -> Any:
        """
        Choose an action using CFR.
        
        Args:
            game_state (GameState): Current game state.
        
        Returns:
            Any: The chosen action.
        """
        return self.cfr.act(game_state, self.player_id)