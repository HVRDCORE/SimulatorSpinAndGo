"""
Implementation of Deep Counterfactual Regret Minimization (Deep CFR).

This module provides an implementation of the Deep CFR algorithm for learning
approximate Nash equilibrium strategies in imperfect information games like poker.
Deep CFR uses deep neural networks to approximate the advantages and strategy
of the game.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

from pokersim.game.state import GameState
from pokersim.game.spingo import SpinGoGame


class ReplayMemory:
    """
    Replay memory for storing advantage samples.
    
    Attributes:
        capacity (int): Maximum size of the memory.
        memory (deque): The internal storage.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize replay memory.
        
        Args:
            capacity (int): Maximum size of the memory.
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, advantage: float):
        """
        Add a sample to memory.
        
        Args:
            state (np.ndarray): State representation.
            advantage (float): Advantage value.
        """
        self.memory.append((state, advantage))
    
    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, float]]:
        """
        Sample a batch from memory.
        
        Args:
            batch_size (int): Size of the batch to sample.
            
        Returns:
            List[Tuple[np.ndarray, float]]: Batch of samples.
        """
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    
    def __len__(self) -> int:
        """
        Get the current size of the memory.
        
        Returns:
            int: Number of samples in memory.
        """
        return len(self.memory)


class DeepCFR:
    """
    Deep Counterfactual Regret Minimization algorithm.
    
    This class implements the Deep CFR algorithm for approximating Nash equilibrium
    strategies in poker games.
    
    Attributes:
        value_network (nn.Module): Neural network for advantage estimation.
        optimizer (optim.Optimizer): Optimizer for the value network.
        memory_size (int): Maximum size of the replay memories.
        batch_size (int): Batch size for training.
        game_type (str): Type of game to play ('holdem' or 'spingo').
        num_players (int): Number of players in the game.
        memories (List[ReplayMemory]): Replay memories for each player.
    """
    
    def __init__(self, value_network: nn.Module, input_dim: int, action_dim: int, 
                game_type: str = 'spingo', num_players: int = 3,
                memory_size: int = 10000, batch_size: int = 128,
                learning_rate: float = 0.001):
        """
        Initialize Deep CFR algorithm.
        
        Args:
            value_network (nn.Module): Neural network for advantage estimation.
            input_dim (int): Input dimension for state representation.
            action_dim (int): Dimension of the action space.
            game_type (str, optional): Type of game. Defaults to 'spingo'.
            num_players (int, optional): Number of players. Defaults to 3.
            memory_size (int, optional): Replay memory size. Defaults to 10000.
            batch_size (int, optional): Batch size for training. Defaults to 128.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
        """
        self.value_network = value_network
        self.optimizer = optim.Adam(value_network.parameters(), lr=learning_rate)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.game_type = game_type
        self.num_players = num_players
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        # Initialize a replay memory for each player
        self.memories = [ReplayMemory(memory_size) for _ in range(num_players)]
        
        # Create device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.value_network.to(self.device)
        
        # Training metrics
        self.iteration = 0
        self.advantage_losses = []
        
    def train_iteration(self):
        """Run a single training iteration."""
        self.iteration += 1
        
        # Create a new game
        if self.game_type == 'spingo':
            game = SpinGoGame(num_players=self.num_players)
            game_state = game.start_new_hand()
        else:
            game_state = GameState(num_players=self.num_players)
            game_state.deal_hole_cards()
        
        # Collect samples through traversal
        for player_id in range(self.num_players):
            self._traverse_game_tree(game_state, player_id)
        
        # Train value network on each player's samples
        total_loss = 0
        for player_id in range(self.num_players):
            if len(self.memories[player_id]) >= self.batch_size:
                loss = self._train_value_network(player_id)
                total_loss += loss
        
        self.advantage_losses.append(total_loss / self.num_players)
        
    def _traverse_game_tree(self, game_state: GameState, traverser_id: int, 
                          iteration: int = None, reach_prob: float = 1.0):
        """
        Traverse the game tree to collect advantage samples.
        
        Args:
            game_state (GameState): Current game state.
            traverser_id (int): ID of the traversing player.
            iteration (int, optional): Current iteration. Defaults to None.
            reach_prob (float, optional): Reach probability. Defaults to 1.0.
        
        Returns:
            float: The expected utility of the state.
        """
        if iteration is None:
            iteration = self.iteration
        
        if game_state.is_terminal():
            # Return utility at terminal state
            return game_state.get_utility(traverser_id)
        
        current_player = game_state.current_player
        
        if current_player == traverser_id:
            # It's the traverser's turn - compute advantages
            legal_actions = game_state.get_legal_actions()
            state_representation = self._get_state_representation(game_state, traverser_id)
            
            # Get advantages using value network
            advantages = self._compute_advantages(state_representation, legal_actions)
            
            # Use regret matching to select an action
            strategy = self._compute_strategy(advantages)
            
            # Sample an action from the strategy
            action_idx = np.random.choice(len(legal_actions), p=strategy)
            action = legal_actions[action_idx]
            
            # Recursively traverse with the selected action
            next_state = game_state.apply_action(action)
            value = self._traverse_game_tree(next_state, traverser_id, iteration, reach_prob)
            
            # Update advantages and store samples
            for i, a in enumerate(legal_actions):
                next_state = game_state.apply_action(a)
                if i == action_idx:
                    # We already computed this value
                    a_value = value
                else:
                    # Compute counterfactual value
                    a_value = self._traverse_game_tree(next_state, traverser_id, iteration, 0)
                
                # Compute advantage
                advantage = a_value - value
                
                # Store the sample for later training
                self.memories[traverser_id].push(state_representation, advantage)
            
            return value
        else:
            # It's another player's turn - use strategy from value network
            legal_actions = game_state.get_legal_actions()
            state_representation = self._get_state_representation(game_state, current_player)
            
            # Get advantages using value network
            advantages = self._compute_advantages(state_representation, legal_actions)
            
            # Compute strategy from advantages
            strategy = self._compute_strategy(advantages)
            
            # Sample an action from the strategy
            action_idx = np.random.choice(len(legal_actions), p=strategy)
            action = legal_actions[action_idx]
            
            # Recursively traverse with the selected action
            next_state = game_state.apply_action(action)
            return self._traverse_game_tree(next_state, traverser_id, iteration, reach_prob)
    
    def _compute_advantages(self, state: np.ndarray, legal_actions: List) -> np.ndarray:
        """
        Compute advantages for each legal action.
        
        Args:
            state (np.ndarray): State representation.
            legal_actions (List): List of legal actions.
            
        Returns:
            np.ndarray: Advantages for each action.
        """
        # Convert state to torch tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get predictions from value network
        with torch.no_grad():
            advantages = self.value_network(state_tensor).cpu().numpy().flatten()
        
        # Mask illegal actions with large negative values
        action_mask = np.zeros(self.action_dim)
        for i, action in enumerate(legal_actions):
            action_type = action.action_type.value - 1  # Convert enum to 0-indexed
            action_mask[action_type] = 1
        
        # Apply mask (set illegal actions to large negative values)
        masked_advantages = advantages * action_mask + (1 - action_mask) * (-1e9)
        
        return masked_advantages
    
    def _compute_strategy(self, advantages: np.ndarray) -> np.ndarray:
        """
        Compute a strategy from advantages using regret matching.
        
        Args:
            advantages (np.ndarray): Action advantages.
            
        Returns:
            np.ndarray: Strategy (action probabilities).
        """
        # Convert advantages to regrets (only positive regrets matter)
        regrets = np.maximum(advantages, 0)
        
        # If all regrets are 0, use uniform strategy
        if np.sum(regrets) <= 0:
            # Find indices of valid actions (not extremely negative)
            valid_indices = np.where(advantages > -1e8)[0]
            strategy = np.zeros_like(advantages)
            strategy[valid_indices] = 1.0 / len(valid_indices)
            return strategy
        
        # Normalize to get a proper strategy
        strategy = regrets / np.sum(regrets)
        return strategy
    
    def _get_state_representation(self, game_state: GameState, player_id: int) -> np.ndarray:
        """
        Convert a game state to a vector representation for the neural network.
        
        Args:
            game_state (GameState): The game state.
            player_id (int): The player ID.
            
        Returns:
            np.ndarray: Vector representation of the state.
        """
        # This is a simplified representation - in practice, you would want a more
        # sophisticated state encoding that captures the relevant game information
        
        # Basic features:
        # - Player's cards
        # - Community cards
        # - Pot size
        # - Player positions
        # - Betting history
        # - Stack sizes
        
        # Placeholder implementation - replace with actual feature extraction
        feature_vector = np.zeros(self.input_dim)
        
        # Return the feature vector
        return feature_vector
    
    def _train_value_network(self, player_id: int) -> float:
        """
        Train the value network on a batch of samples for a player.
        
        Args:
            player_id (int): The player ID.
            
        Returns:
            float: The training loss.
        """
        # Sample a batch from memory
        batch = self.memories[player_id].sample(self.batch_size)
        states, advantages = zip(*batch)
        
        # Convert to torch tensors
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        advantages_tensor = torch.tensor(np.array(advantages), dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Forward pass
        predicted_advantages = self.value_network(states_tensor)
        
        # Compute loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(predicted_advantages, advantages_tensor)
        
        # Backward pass and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def act(self, game_state: GameState, player_id: int) -> Any:
        """
        Choose an action using the current policy.
        
        Args:
            game_state (GameState): Current game state.
            player_id (int): The player ID.
            
        Returns:
            Any: The chosen action.
        """
        legal_actions = game_state.get_legal_actions()
        state_representation = self._get_state_representation(game_state, player_id)
        
        # Get advantages using value network
        advantages = self._compute_advantages(state_representation, legal_actions)
        
        # Compute strategy from advantages
        strategy = self._compute_strategy(advantages)
        
        # Sample an action from the strategy
        action_idx = np.random.choice(len(legal_actions), p=strategy)
        return legal_actions[action_idx]
    
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