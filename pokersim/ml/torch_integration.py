"""
Integration with PyTorch for poker simulations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from pokersim.game.state import GameState, Action
from pokersim.ml.models import PokerMLP, PokerCNN, PokerActorCritic


class TorchAgent:
    """
    Base class for PyTorch-based poker agents.
    
    Attributes:
        model (nn.Module): The PyTorch model.
        optimizer (optim.Optimizer): The optimizer.
        device (torch.device): The device (CPU or GPU).
    """
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, device: Optional[torch.device] = None):
        """
        Initialize a PyTorch agent.
        
        Args:
            model (nn.Module): The PyTorch model.
            optimizer (optim.Optimizer): The optimizer.
            device (torch.device, optional): The device (CPU or GPU). Defaults to CPU.
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device if device is not None else torch.device("cpu")
        self.model.to(self.device)
    
    def state_to_tensor(self, game_state: GameState, player_id: int) -> torch.Tensor:
        """
        Convert a game state to a PyTorch tensor.
        
        Args:
            game_state (GameState): The game state.
            player_id (int): The player ID.
            
        Returns:
            torch.Tensor: The tensor representation of the game state.
        """
        # Get feature vector
        features = game_state.to_feature_vector(player_id)
        
        # Convert to tensor
        tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        return tensor
    
    def tensor_to_action(self, tensor: torch.Tensor, game_state: GameState) -> Action:
        """
        Convert a PyTorch tensor to an action.
        
        Args:
            tensor (torch.Tensor): The tensor.
            game_state (GameState): The game state.
            
        Returns:
            Action: The action.
        """
        # Get legal actions
        legal_actions = game_state.get_legal_actions()
        
        # If no legal actions, return None
        if not legal_actions:
            return None
        
        # Get action probabilities
        probs = torch.softmax(tensor, dim=0).cpu().detach().numpy()
        
        # Choose action with highest probability
        action_idx = np.argmax(probs)
        
        # Map to legal action
        return legal_actions[min(action_idx, len(legal_actions) - 1)]
    
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path (str): The file path.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str) -> None:
        """
        Load the model from a file.
        
        Args:
            path (str): The file path.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class ExperienceBuffer:
    """
    Buffer for storing experiences during training.
    
    Attributes:
        states (List[np.ndarray]): The states.
        actions (List[int]): The actions.
        rewards (List[float]): The rewards.
        next_states (List[np.ndarray]): The next states.
        dones (List[bool]): Whether the episodes are done.
        capacity (int): The capacity of the buffer.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize an experience buffer.
        
        Args:
            capacity (int, optional): The capacity. Defaults to 10000.
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.capacity = capacity
    
    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            state (np.ndarray): The state.
            action (int): The action.
            reward (float): The reward.
            next_state (np.ndarray): The next state.
            done (bool): Whether the episode is done.
        """
        # If buffer is full, remove oldest experience
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
        
        # Add new experience
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample experiences from the buffer.
        
        Args:
            batch_size (int): The batch size.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The sampled experiences.
        """
        # Sample indices
        indices = np.random.choice(len(self.states), min(batch_size, len(self.states)), replace=False)
        
        # Get sampled experiences
        states = np.array([self.states[i] for i in indices])
        actions = np.array([self.actions[i] for i in indices])
        rewards = np.array([self.rewards[i] for i in indices])
        next_states = np.array([self.next_states[i] for i in indices])
        dones = np.array([self.dones[i] for i in indices])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Get the current size of the buffer."""
        return len(self.states)


class ActorCriticTrainer:
    """
    Trainer for actor-critic models.
    
    Attributes:
        model (PokerActorCritic): The actor-critic model.
        optimizer (optim.Optimizer): The optimizer.
        device (torch.device): The device (CPU or GPU).
        gamma (float): The discount factor.
    """
    
    def __init__(self, model: PokerActorCritic, optimizer: optim.Optimizer, device: torch.device, gamma: float = 0.99):
        """
        Initialize an actor-critic trainer.
        
        Args:
            model (PokerActorCritic): The actor-critic model.
            optimizer (optim.Optimizer): The optimizer.
            device (torch.device): The device (CPU or GPU).
            gamma (float, optional): The discount factor. Defaults to 0.99.
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.gamma = gamma
    
    def train_step(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, 
                next_states: torch.Tensor, dones: torch.Tensor) -> Dict[str, float]:
        """
        Perform a training step.
        
        Args:
            states (torch.Tensor): The states.
            actions (torch.Tensor): The actions.
            rewards (torch.Tensor): The rewards.
            next_states (torch.Tensor): The next states.
            dones (torch.Tensor): Whether the episodes are done.
            
        Returns:
            Dict[str, float]: The training metrics.
        """
        # Get policy and value predictions
        policy, values = self.model(states)
        _, next_values = self.model(next_states)
        
        # Compute advantages and target values
        target_values = rewards + self.gamma * next_values * (1 - dones)
        advantages = target_values - values
        
        # Compute policy loss
        policy_probs = torch.softmax(policy, dim=1)
        log_probs = torch.log(policy_probs + 1e-10)
        action_log_probs = torch.gather(log_probs, 1, actions.unsqueeze(1)).squeeze(1)
        policy_loss = -torch.mean(action_log_probs * advantages.detach())
        
        # Compute value loss
        value_loss = torch.mean(torch.square(advantages))
        
        # Compute entropy loss (for exploration)
        entropy = -torch.mean(torch.sum(policy_probs * log_probs, dim=1))
        
        # Compute total loss
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Return metrics
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
