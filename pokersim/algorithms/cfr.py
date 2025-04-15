"""
Counterfactual Regret Minimization (CFR) algorithm for the poker simulator.

This module implements the vanilla CFR algorithm for solving imperfect
information games like poker. It includes functionality for computing
Nash equilibrium strategies through iterative self-play.
"""

import os
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
import time
import random
from collections import defaultdict

from pokersim.config.config_manager import get_config
from pokersim.logging.game_logger import get_logger

# Configure logging
logger = logging.getLogger("pokersim.algorithms.cfr")


class CFRSolver:
    """
    Counterfactual Regret Minimization (CFR) solver for poker games.
    
    This class implements vanilla CFR for computing approximate Nash equilibrium
    strategies in extensive-form games like poker. It tracks regrets and strategies
    for each information set and updates them through iterative self-play.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings.
        game_logger: Game logger instance.
        info_sets (Dict[str, Dict]): Information sets with regrets and strategies.
        iterations (int): Number of iterations performed.
        game_state_class: Class for the game state (e.g., PokerState).
    """
    
    def __init__(self, game_state_class: Any):
        """
        Initialize the CFR solver.
        
        Args:
            game_state_class: Class for the game state.
        """
        # Get configuration
        config = get_config()
        self.config = config.to_dict()
        
        # Set up logging
        self.game_logger = get_logger()
        
        # Initialize solver state
        self.info_sets = {}  # Information set -> {regrets, strategy, cumulative_strategy}
        self.iterations = 0
        self.game_state_class = game_state_class
    
    def get_strategy(self, info_set_key: str, num_actions: int) -> np.ndarray:
        """
        Get current strategy for an information set.
        
        Args:
            info_set_key (str): Information set key.
            num_actions (int): Number of possible actions.
        
        Returns:
            np.ndarray: Probability distribution over actions.
        """
        # Create information set if it doesn't exist
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = {
                "regrets": np.zeros(num_actions),
                "strategy": np.ones(num_actions) / num_actions,  # Uniform initial strategy
                "cumulative_strategy": np.zeros(num_actions)
            }
        
        # Get information set data
        info_set = self.info_sets[info_set_key]
        
        # Compute strategy from regrets using Regret Matching
        regrets = np.maximum(0, info_set["regrets"])  # Only consider positive regrets
        regret_sum = np.sum(regrets)
        
        if regret_sum > 0:
            # Normalize regrets to get strategy
            strategy = regrets / regret_sum
        else:
            # Use uniform strategy if all regrets are non-positive
            strategy = np.ones(num_actions) / num_actions
        
        # Update strategy in the information set
        info_set["strategy"] = strategy
        
        return strategy
    
    def update_cumulative_strategy(self, info_set_key: str, strategy: np.ndarray, 
                                  reach_prob: float):
        """
        Update cumulative strategy for an information set.
        
        Args:
            info_set_key (str): Information set key.
            strategy (np.ndarray): Current strategy.
            reach_prob (float): Reach probability for the player.
        """
        self.info_sets[info_set_key]["cumulative_strategy"] += reach_prob * strategy
    
    def get_average_strategy(self, info_set_key: str) -> np.ndarray:
        """
        Get average strategy for an information set.
        
        The average strategy approaches a Nash equilibrium as the number of
        iterations increases.
        
        Args:
            info_set_key (str): Information set key.
        
        Returns:
            np.ndarray: Average strategy (probability distribution over actions).
        """
        if info_set_key not in self.info_sets:
            return None
        
        cumulative_strategy = self.info_sets[info_set_key]["cumulative_strategy"]
        total = np.sum(cumulative_strategy)
        
        if total > 0:
            return cumulative_strategy / total
        else:
            # Uniform strategy if no cumulative strategy
            return np.ones(len(cumulative_strategy)) / len(cumulative_strategy)
    
    def update_regrets(self, info_set_key: str, action_regrets: np.ndarray):
        """
        Update regrets for an information set.
        
        Args:
            info_set_key (str): Information set key.
            action_regrets (np.ndarray): Regrets for each action.
        """
        self.info_sets[info_set_key]["regrets"] += action_regrets
    
    def cfr(self, state: Any, reach_probs: List[float]) -> float:
        """
        Run Counterfactual Regret Minimization recursively on a game state.
        
        Args:
            state: Current game state.
            reach_probs (List[float]): Reach probabilities for each player.
        
        Returns:
            float: Expected value for the active player.
        """
        # Return terminal utility
        if state.is_terminal():
            return state.get_utility()
        
        # Return expected value for chance nodes
        if state.is_chance_node():
            outcomes = state.get_chance_outcomes()
            expected_value = 0.0
            
            for action, prob in outcomes:
                next_state = state.apply_action(action)
                expected_value += prob * self.cfr(next_state, reach_probs)
            
            return expected_value
        
        # Get current player and information set
        player = state.get_current_player()
        info_set_key = state.get_info_set_key()
        legal_actions = state.get_legal_actions()
        num_actions = len(legal_actions)
        
        # Get strategy for this information set
        strategy = self.get_strategy(info_set_key, num_actions)
        
        # Update cumulative strategy
        self.update_cumulative_strategy(info_set_key, strategy, reach_probs[player])
        
        # Recursively call CFR for each action and compute expected value
        action_values = np.zeros(num_actions)
        
        for i, action in enumerate(legal_actions):
            # Create new reach probabilities with this action
            new_reach_probs = reach_probs.copy()
            new_reach_probs[player] *= strategy[i]
            
            # Apply action and recurse
            next_state = state.apply_action(action)
            action_values[i] = self.cfr(next_state, new_reach_probs)
        
        # Compute counterfactual value
        cf_value = np.sum(strategy * action_values)
        
        # Compute counterfactual regrets
        regrets = action_values - cf_value
        
        # Update regrets scaled by counterfactual reach probability
        cf_reach_prob = np.prod(reach_probs) / reach_probs[player]  # Reach probability of other players
        self.update_regrets(info_set_key, regrets * cf_reach_prob)
        
        return cf_value
    
    def train(self, num_iterations: int, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Train the CFR solver for a specified number of iterations.
        
        Args:
            num_iterations (int): Number of iterations to run.
            callback (Optional[Callable], optional): Callback function called
                after each iteration. Defaults to None.
        
        Returns:
            Dict[str, Any]: Training results.
        """
        start_time = time.time()
        metrics = []
        
        for i in range(num_iterations):
            # Initialize game state
            initial_state = self.game_state_class()
            
            # Determine initial reach probabilities
            num_players = initial_state.get_num_players()
            reach_probs = [1.0] * num_players
            
            # Run CFR
            utility = self.cfr(initial_state, reach_probs)
            
            # Update iteration counter
            self.iterations += 1
            
            # Log progress
            if (i + 1) % max(1, num_iterations // 10) == 0:
                elapsed = time.time() - start_time
                exploitability = self.compute_exploitability() if hasattr(self, "compute_exploitability") else None
                
                logger.info(f"Iteration {i+1}/{num_iterations} - Utility: {utility:.6f}" + 
                           (f" - Exploitability: {exploitability:.6f}" if exploitability is not None else ""))
                
                # Record metrics
                metric = {
                    "iteration": i + 1,
                    "utility": float(utility),
                    "time": elapsed
                }
                
                if exploitability is not None:
                    metric["exploitability"] = float(exploitability)
                
                metrics.append(metric)
                
                # Log to game logger
                self.game_logger.log_training_metrics("CFR", metric, i + 1)
            
            # Call callback if provided
            if callback is not None:
                callback(i, utility, self)
        
        total_time = time.time() - start_time
        logger.info(f"CFR training completed in {total_time:.2f} seconds")
        
        # Return results
        return {
            "algorithm": "CFR",
            "iterations": num_iterations,
            "total_time": total_time,
            "info_sets_count": len(self.info_sets),
            "metrics": metrics
        }
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the CFR solver state to a file.
        
        Args:
            filepath (Optional[str], optional): Path to save the state. Defaults to None.
        
        Returns:
            str: Path to the saved file.
        """
        if filepath is None:
            # Generate default filepath
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filepath = os.path.join(self.config["agents"]["save_dir"], f"cfr_model_{timestamp}.json")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare data for saving
        data = {
            "algorithm": "CFR",
            "iterations": self.iterations,
            "timestamp": time.time(),
            "info_sets": {}
        }
        
        # Convert information sets to serializable format
        for info_set_key, info_set in self.info_sets.items():
            data["info_sets"][info_set_key] = {
                "regrets": info_set["regrets"].tolist(),
                "strategy": info_set["strategy"].tolist(),
                "cumulative_strategy": info_set["cumulative_strategy"].tolist()
            }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved CFR model to {filepath}")
        return filepath
    
    def load(self, filepath: str) -> bool:
        """
        Load the CFR solver state from a file.
        
        Args:
            filepath (str): Path to the saved state.
        
        Returns:
            bool: Whether the load was successful.
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Check algorithm
            if data.get("algorithm") != "CFR":
                logger.error(f"Invalid algorithm in {filepath}, expected CFR")
                return False
            
            # Load iterations
            self.iterations = data.get("iterations", 0)
            
            # Load information sets
            self.info_sets = {}
            for info_set_key, info_set in data.get("info_sets", {}).items():
                self.info_sets[info_set_key] = {
                    "regrets": np.array(info_set["regrets"]),
                    "strategy": np.array(info_set["strategy"]),
                    "cumulative_strategy": np.array(info_set["cumulative_strategy"])
                }
            
            logger.info(f"Loaded CFR model from {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading CFR model from {filepath}: {e}")
            return False
    
    def get_best_response(self, info_set_key: str) -> int:
        """
        Get the best response action for an information set.
        
        Args:
            info_set_key (str): Information set key.
        
        Returns:
            int: Best response action index.
        """
        if info_set_key not in self.info_sets:
            # Return random action if information set not found
            return 0
        
        # Get average strategy
        strategy = self.get_average_strategy(info_set_key)
        
        # Return action with highest probability
        return np.argmax(strategy)
    
    def compute_exploitability(self, num_samples: int = 1000) -> float:
        """
        Compute the exploitability of the current strategy.
        
        This is a simplified implementation that estimates exploitability
        by sampling states. For a full implementation, see the CFR+ paper.
        
        Args:
            num_samples (int, optional): Number of states to sample. Defaults to 1000.
        
        Returns:
            float: Estimated exploitability.
        """
        # This is a simplified placeholder for exploitability computation
        # A full implementation would require best response computation
        # which is complex and depends on the specific game
        
        # For simplicity, we return a proxy measure based on regret
        total_regret = 0.0
        count = 0
        
        for info_set in self.info_sets.values():
            regrets = np.maximum(0, info_set["regrets"])
            if np.sum(regrets) > 0:
                total_regret += np.max(regrets)
                count += 1
        
        if count > 0:
            return total_regret / (count * self.iterations)
        else:
            return 0.0
    
    def get_action_probabilities(self, state: Any) -> np.ndarray:
        """
        Get action probabilities for a given state.
        
        Args:
            state: Game state.
        
        Returns:
            np.ndarray: Probability distribution over actions.
        """
        info_set_key = state.get_info_set_key()
        legal_actions = state.get_legal_actions()
        num_actions = len(legal_actions)
        
        if info_set_key in self.info_sets:
            return self.get_average_strategy(info_set_key)
        else:
            # Return uniform distribution if information set not found
            return np.ones(num_actions) / num_actions