"""
Implementation of Proximal Policy Optimization (PPO) for poker.

This module provides an implementation of the PPO algorithm for training
agents in the poker domain. PPO is a policy gradient method that uses
a trust region constraint to ensure stable learning.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pokersim.game.state import GameState
from pokersim.game.spingo import SpinGoGame


class PPOMemory:
    """
    Memory buffer for PPO algorithm.
    
    This class stores trajectories of experiences that include states, actions,
    rewards, values, log probs, and episode information.
    
    Attributes:
        states (List): List of state representations.
        actions (List): List of actions taken.
        rewards (List): List of rewards received.
        values (List): List of value estimates.
        log_probs (List): List of log probabilities of actions.
        dones (List): List of episode termination flags.
        size (int): Current size of the memory.
    """
    
    def __init__(self, batch_size: int):
        """
        Initialize PPO memory.
        
        Args:
            batch_size (int): Batch size for training.
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.batch_size = batch_size
        self.size = 0
    
    def store(self, state, action, reward, value, log_prob, done):
        """
        Store a transition in memory.
        
        Args:
            state: State representation.
            action: Action taken.
            reward (float): Reward received.
            value (float): Value estimate.
            log_prob (float): Log probability of the action.
            done (bool): Whether the episode ended.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.size += 1
    
    def clear(self):
        """Clear the memory buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.size = 0
    
    def get_batches(self) -> List:
        """
        Get batches of data for training.
        
        Returns:
            List: Batches of experiences.
        """
        indices = np.arange(self.size)
        np.random.shuffle(indices)
        
        batches = []
        start_idx = 0
        
        while start_idx < self.size:
            end_idx = min(start_idx + self.batch_size, self.size)
            batch_indices = indices[start_idx:end_idx]
            
            batch = {
                'states': [self.states[i] for i in batch_indices],
                'actions': [self.actions[i] for i in batch_indices],
                'rewards': [self.rewards[i] for i in batch_indices],
                'values': [self.values[i] for i in batch_indices],
                'log_probs': [self.log_probs[i] for i in batch_indices],
                'dones': [self.dones[i] for i in batch_indices]
            }
            
            batches.append(batch)
            start_idx = end_idx
        
        return batches


class PPO:
    """
    Proximal Policy Optimization algorithm for poker.
    
    This class implements the PPO algorithm for training agents in poker games.
    It includes methods for collecting trajectories, computing advantages, and
    updating the policy network.
    
    Attributes:
        policy_network (nn.Module): Neural network for policy approximation.
        optimizer (optim.Optimizer): Optimizer for the policy network.
        game_type (str): Type of game to play ('holdem' or 'spingo').
        num_players (int): Number of players in the game.
        memory (PPOMemory): Memory buffer for storing experiences.
        clip_ratio (float): PPO clipping parameter.
        value_coef (float): Value loss coefficient.
        entropy_coef (float): Entropy loss coefficient.
    """
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer,
                game_type: str = 'spingo', num_players: int = 3,
                learning_rate: float = 0.0003, clip_ratio: float = 0.2,
                value_coef: float = 0.5, entropy_coef: float = 0.01,
                batch_size: int = 64, gae_lambda: float = 0.95, gamma: float = 0.99):
        """
        Initialize PPO algorithm.
        
        Args:
            model (nn.Module): Neural network model.
            optimizer (optim.Optimizer): Optimizer for the model.
            game_type (str, optional): Type of game. Defaults to 'spingo'.
            num_players (int, optional): Number of players. Defaults to 3.
            learning_rate (float, optional): Learning rate. Defaults to 0.0003.
            clip_ratio (float, optional): PPO clipping ratio. Defaults to 0.2.
            value_coef (float, optional): Value loss coefficient. Defaults to 0.5.
            entropy_coef (float, optional): Entropy loss coefficient. Defaults to 0.01.
            batch_size (int, optional): Batch size for training. Defaults to 64.
            gae_lambda (float, optional): GAE lambda parameter. Defaults to 0.95.
            gamma (float, optional): Discount factor. Defaults to 0.99.
        """
        self.model = model
        self.optimizer = optimizer
        self.game_type = game_type
        self.num_players = num_players
        
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        
        self.memory = PPOMemory(batch_size)
        
        # Create device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Training metrics
        self.episode = 0
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
    
    def train_episode(self):
        """Run a single training episode."""
        self.episode += 1
        
        # Create a new game
        if self.game_type == 'spingo':
            game = SpinGoGame(num_players=self.num_players)
            game_state = game.start_new_hand()
        else:
            game_state = GameState(num_players=self.num_players)
            game_state.deal_hole_cards()
        
        # Clear the memory
        self.memory.clear()
        
        # Run episode
        terminated = False
        while not terminated:
            self._collect_experience(game_state)
            
            if game_state.is_terminal():
                # If the current hand is terminal, get a new hand or end the episode
                if hasattr(game, 'is_tournament_over') and game.is_tournament_over():
                    terminated = True
                else:
                    try:
                        game_state = game.start_new_hand()
                    except:
                        terminated = True
        
        # Update the policy network
        if self.memory.size > 0:
            self._update_policy()
    
    def _collect_experience(self, game_state: GameState):
        """
        Collect experience by playing a hand.
        
        Args:
            game_state (GameState): Current game state.
        """
        # Current player is the agent being trained (assume 0)
        agent_id = 0
        
        while not game_state.is_terminal():
            current_player = game_state.current_player
            
            if current_player == agent_id:
                # Agent's turn - use the policy network
                state_rep = self._get_state_representation(game_state, agent_id)
                legal_actions = game_state.get_legal_actions()
                
                # Convert state to torch tensor
                state_tensor = torch.tensor(state_rep, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get action probabilities and value estimate
                with torch.no_grad():
                    policy_logits, value = self.model(state_tensor)
                    action_probs = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
                
                # Mask illegal actions
                action_mask = np.zeros_like(action_probs)
                for i, action in enumerate(legal_actions):
                    action_type = action.action_type.value - 1  # Convert enum to 0-indexed
                    action_mask[action_type] = 1
                
                # Apply mask and normalize probabilities
                masked_probs = action_probs * action_mask
                if np.sum(masked_probs) > 0:
                    masked_probs = masked_probs / np.sum(masked_probs)
                else:
                    # Fallback to uniform distribution over legal actions
                    masked_probs = action_mask / np.sum(action_mask)
                
                # Sample action
                action_idx = np.random.choice(len(masked_probs), p=masked_probs)
                action = None
                for a in legal_actions:
                    if a.action_type.value - 1 == action_idx:
                        action = a
                        break
                
                if action is None:
                    # Fallback if no matching action type
                    action = legal_actions[0]
                
                # Compute log probability
                log_prob = np.log(masked_probs[action_idx] + 1e-10)
                
                # Execute action
                next_state = game_state.apply_action(action)
                
                # Get immediate reward (0 for non-terminal states)
                reward = 0.0
                if next_state.is_terminal():
                    reward = next_state.get_utility(agent_id)
                
                # Store experience
                self.memory.store(
                    state=state_rep,
                    action=action_idx,
                    reward=reward,
                    value=value.item(),
                    log_prob=log_prob,
                    done=next_state.is_terminal()
                )
                
                game_state = next_state
            else:
                # Other player's turn - use a simple rule-based policy
                legal_actions = game_state.get_legal_actions()
                
                # Simple strategy: uniformly random choice
                action = random.choice(legal_actions)
                
                # Execute action
                game_state = game_state.apply_action(action)
    
    def _update_policy(self):
        """Update the policy network using PPO algorithm."""
        # Compute advantages and returns
        returns, advantages = self._compute_advantages()
        
        # Convert to torch tensors
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        # Get batches from memory
        batches = self.memory.get_batches()
        
        # Train for multiple epochs
        for _ in range(3):  # Number of PPO epochs
            for batch in batches:
                # Process batch data
                batch_states = np.array(batch['states'])
                batch_actions = np.array(batch['actions'])
                batch_log_probs = np.array(batch['log_probs'])
                
                batch_indices = np.arange(len(batch_states))
                
                # Convert to torch tensors
                states_tensor = torch.tensor(batch_states, dtype=torch.float32).to(self.device)
                actions_tensor = torch.tensor(batch_actions, dtype=torch.long).to(self.device)
                old_log_probs_tensor = torch.tensor(batch_log_probs, dtype=torch.float32).to(self.device)
                
                # Forward pass
                policy_logits, values = self.model(states_tensor)
                
                # Get new probabilities
                policy_distribution = F.softmax(policy_logits, dim=-1)
                new_probs = policy_distribution.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                new_log_probs = torch.log(new_probs + 1e-10)
                
                # Compute entropy
                entropy = -(policy_distribution * torch.log(policy_distribution + 1e-10)).sum(dim=1).mean()
                
                # Compute the policy ratio and clipped objective
                ratio = torch.exp(new_log_probs - old_log_probs_tensor)
                
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Normalize advantages for stable training
                if len(batch_advantages) > 1:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
                
                # PPO objectives
                obj = ratio * batch_advantages
                obj_clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Record metrics
                self.policy_losses.append(policy_loss.item())
                self.value_losses.append(value_loss.item())
                self.entropy_losses.append(entropy.item())
    
    def _compute_advantages(self) -> Tuple[List[float], List[float]]:
        """
        Compute advantages and returns using GAE.
        
        Returns:
            Tuple[List[float], List[float]]: Returns and advantages.
        """
        rewards = np.array(self.memory.rewards)
        values = np.array(self.memory.values)
        dones = np.array(self.memory.dones).astype(np.float32)
        
        returns = []
        advantages = []
        
        # Initialize with zeros
        gae = 0
        next_value = 0  # Assume terminal states have value 0
        
        # Compute returns and advantages in reverse order
        for t in reversed(range(len(rewards))):
            # For terminal states, next_value is 0, otherwise it's the value of the next state
            if t < len(rewards) - 1:
                next_value = values[t + 1]
            
            # Compute temporal difference
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # Compute GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            # Add to lists
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return returns, advantages
    
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
        feature_vector = np.zeros(128)  # Assuming input dimension is 128
        
        # Return the feature vector
        return feature_vector
    
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
        
        # Convert state to torch tensor
        state_tensor = torch.tensor(state_representation, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get action probabilities
        with torch.no_grad():
            policy_logits, _ = self.model(state_tensor)
            action_probs = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
        
        # Mask illegal actions
        action_mask = np.zeros_like(action_probs)
        for i, action in enumerate(legal_actions):
            action_type = action.action_type.value - 1  # Convert enum to 0-indexed
            action_mask[action_type] = 1
        
        # Apply mask and normalize probabilities
        masked_probs = action_probs * action_mask
        if np.sum(masked_probs) > 0:
            masked_probs = masked_probs / np.sum(masked_probs)
        else:
            # Fallback to uniform distribution over legal actions
            masked_probs = action_mask / np.sum(action_mask)
        
        # Sample action
        action_idx = np.random.choice(len(masked_probs), p=masked_probs)
        
        # Find the corresponding action
        for action in legal_actions:
            if action.action_type.value - 1 == action_idx:
                return action
        
        # Fallback if no matching action type found
        return legal_actions[0]
    
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