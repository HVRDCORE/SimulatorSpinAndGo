"""
Implementation of machine learning-based agents for poker simulations.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional

from pokersim.agents.base_agent import Agent
from pokersim.game.state import GameState, Action, ActionType
from pokersim.ml.models import PokerMLP, PokerCNN, PokerActorCritic
from pokersim.ml.torch_integration import TorchAgent, ExperienceBuffer


class MLAgent(Agent):
    """
    Base class for machine learning-based poker agents.
    
    Attributes:
        player_id (int): The ID of the player controlled by this agent.
    """
    
    def __init__(self, player_id: int):
        """
        Initialize an ML agent.
        
        Args:
            player_id (int): The ID of the player controlled by this agent.
        """
        super().__init__(player_id)
    
    def act(self, game_state: GameState) -> Action:
        """
        Choose an action based on the current game state.
        
        Args:
            game_state (GameState): The current game state.
            
        Returns:
            Action: The chosen action.
        """
        raise NotImplementedError
    
    def observe(self, game_state: GameState) -> None:
        """
        Update the agent's internal state based on the current game state.
        
        Args:
            game_state (GameState): The current game state.
        """
        pass
    
    def reset(self) -> None:
        """Reset the agent's internal state."""
        pass
    
    def end_hand(self, game_state: GameState) -> None:
        """
        Update the agent's internal state at the end of a hand.
        
        Args:
            game_state (GameState): The final game state.
        """
        pass


class RandomMLAgent(MLAgent):
    """
    A random machine learning agent for testing.
    
    Attributes:
        player_id (int): The ID of the player controlled by this agent.
    """
    
    def act(self, game_state: GameState) -> Action:
        """
        Choose an action based on the current game state.
        
        Args:
            game_state (GameState): The current game state.
            
        Returns:
            Action: The chosen action.
        """
        legal_actions = game_state.get_legal_actions()
        if not legal_actions:
            raise ValueError("No legal actions available")
        
        return random.choice(legal_actions)


class TorchMLAgent(MLAgent, TorchAgent):
    """
    A PyTorch-based machine learning agent.
    
    Attributes:
        player_id (int): The ID of the player controlled by this agent.
        model (nn.Module): The PyTorch model.
        optimizer (optim.Optimizer): The optimizer.
        device (torch.device): The device (CPU or GPU).
    """
    
    def __init__(self, player_id: int, model: nn.Module, optimizer: optim.Optimizer, 
                device: Optional[torch.device] = None, epsilon: float = 0.1):
        """
        Initialize a PyTorch-based ML agent.
        
        Args:
            player_id (int): The ID of the player controlled by this agent.
            model (nn.Module): The PyTorch model.
            optimizer (optim.Optimizer): The optimizer.
            device (torch.device, optional): The device (CPU or GPU). Defaults to CPU.
            epsilon (float, optional): The exploration rate. Defaults to 0.1.
        """
        MLAgent.__init__(self, player_id)
        TorchAgent.__init__(self, model, optimizer, device)
        self.epsilon = epsilon
    
    def act(self, game_state: GameState) -> Action:
        """
        Choose an action based on the current game state.
        
        Args:
            game_state (GameState): The current game state.
            
        Returns:
            Action: The chosen action.
        """
        legal_actions = game_state.get_legal_actions()
        if not legal_actions:
            raise ValueError("No legal actions available")
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        # Convert state to tensor
        state_tensor = self.state_to_tensor(game_state, self.player_id)
        
        # Get model prediction
        with torch.no_grad():
            # Handle different model types
            if isinstance(self.model, PokerActorCritic):
                policy, _ = self.model(state_tensor.unsqueeze(0))
                policy = policy.squeeze(0)
            else:
                policy = self.model(state_tensor.unsqueeze(0)).squeeze(0)
            
            # Filter out illegal actions and get probabilities
            action_probs = torch.softmax(policy, dim=0).cpu().numpy()
            
            # Choose action with highest probability
            action_idx = np.argmax(action_probs[:len(legal_actions)])
            
            return legal_actions[action_idx]


class DQNAgent(MLAgent):
    """
    A Deep Q-Network agent for poker.
    
    Attributes:
        player_id (int): The ID of the player controlled by this agent.
        model (nn.Module): The Q-network model.
        target_model (nn.Module): The target Q-network model.
        optimizer (optim.Optimizer): The optimizer.
        buffer (ExperienceBuffer): The experience buffer.
        device (torch.device): The device (CPU or GPU).
        epsilon (float): The exploration rate.
        gamma (float): The discount factor.
        batch_size (int): The batch size for training.
        target_update_freq (int): The frequency of target network updates.
        step_counter (int): Counter for target network updates.
    """
    
    def __init__(self, player_id: int, model: nn.Module, optimizer: optim.Optimizer, 
                device: Optional[torch.device] = None, epsilon: float = 0.1, gamma: float = 0.99,
                batch_size: int = 64, target_update_freq: int = 100):
        """
        Initialize a DQN agent.
        
        Args:
            player_id (int): The ID of the player controlled by this agent.
            model (nn.Module): The Q-network model.
            optimizer (optim.Optimizer): The optimizer.
            device (torch.device, optional): The device (CPU or GPU). Defaults to CPU.
            epsilon (float, optional): The exploration rate. Defaults to 0.1.
            gamma (float, optional): The discount factor. Defaults to 0.99.
            batch_size (int, optional): The batch size for training. Defaults to 64.
            target_update_freq (int, optional): The frequency of target network updates. Defaults to 100.
        """
        super().__init__(player_id)
        self.model = model
        self.target_model = type(model)(*model.__init__.__defaults__).to(device)
        self.target_model.load_state_dict(model.state_dict())
        self.optimizer = optimizer
        self.buffer = ExperienceBuffer()
        self.device = device if device is not None else torch.device("cpu")
        self.model.to(self.device)
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_counter = 0
        
        # For storing the current state and action
        self.current_state = None
        self.current_action = None
    
    def state_to_tensor(self, game_state: GameState) -> torch.Tensor:
        """
        Convert a game state to a PyTorch tensor.
        
        Args:
            game_state (GameState): The game state.
            
        Returns:
            torch.Tensor: The tensor representation of the game state.
        """
        # Get feature vector
        features = game_state.to_feature_vector(self.player_id)
        
        # Convert to tensor
        tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        return tensor
    
    def act(self, game_state: GameState) -> Action:
        """
        Choose an action based on the current game state.
        
        Args:
            game_state (GameState): The current game state.
            
        Returns:
            Action: The chosen action.
        """
        legal_actions = game_state.get_legal_actions()
        if not legal_actions:
            raise ValueError("No legal actions available")
        
        # Store current state
        self.current_state = game_state.to_feature_vector(self.player_id)
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            action = random.choice(legal_actions)
            self.current_action = legal_actions.index(action)
            return action
        
        # Convert state to tensor
        state_tensor = self.state_to_tensor(game_state)
        
        # Get model prediction
        with torch.no_grad():
            q_values = self.model(state_tensor.unsqueeze(0)).squeeze(0)
            
            # Filter out illegal actions
            legal_q_values = q_values[:len(legal_actions)]
            
            # Choose action with highest Q-value
            action_idx = torch.argmax(legal_q_values).item()
            
            self.current_action = action_idx
            return legal_actions[action_idx]
    
    def observe(self, game_state: GameState) -> None:
        """
        Update the agent's internal state based on the current game state.
        
        Args:
            game_state (GameState): The current game state.
        """
        # Skip if we don't have a current state or action
        if self.current_state is None or self.current_action is None:
            return
        
        # Get reward
        reward = game_state.get_rewards()[self.player_id]
        
        # Get next state
        next_state = game_state.to_feature_vector(self.player_id)
        
        # Check if the episode is done
        done = game_state.is_terminal()
        
        # Add experience to buffer
        self.buffer.add(self.current_state, self.current_action, reward, next_state, done)
        
        # Train the model
        self.train()
        
        # Update current state and action
        self.current_state = next_state
        self.current_action = None
    
    def train(self) -> None:
        """Train the model on a batch of experiences."""
        # Skip if the buffer doesn't have enough experiences
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Compute current Q-values
        current_q = self.model(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_model(next_states)
            max_next_q = next_q.max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Compute loss
        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
    
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path (str): The file path.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_counter': self.step_counter
        }, path)
    
    def load(self, path: str) -> None:
        """
        Load the model from a file.
        
        Args:
            path (str): The file path.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_counter = checkpoint['step_counter']
