"""
Neural network models for poker AI agents.

This module provides neural network model definitions for value networks, 
policy networks, and other components needed for reinforcement learning
algorithms in the poker domain.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    """
    Neural network for value function approximation.
    
    This network is used in algorithms like Deep CFR to estimate the value
    of a particular game state or information set.
    
    Attributes:
        input_size (int): Size of the input feature vector.
        hidden_layers (List[int]): Sizes of hidden layers.
        output_size (int): Size of the output (typically 1 for value estimation).
    """
    
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int = 1):
        """
        Initialize a value network.
        
        Args:
            input_size (int): Size of the input feature vector.
            hidden_layers (List[int]): Sizes of hidden layers.
            output_size (int, optional): Size of the output. Defaults to 1.
        """
        super(ValueNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.network(x)


class PolicyNetwork(nn.Module):
    """
    Neural network for policy function approximation.
    
    This network is used in algorithms like PPO to estimate action probabilities
    for a given game state.
    
    Attributes:
        input_size (int): Size of the input feature vector.
        hidden_layers (List[int]): Sizes of hidden layers.
        action_space_size (int): Size of the action space.
    """
    
    def __init__(self, input_size: int, hidden_layers: List[int], action_space_size: int):
        """
        Initialize a policy network.
        
        Args:
            input_size (int): Size of the input feature vector.
            hidden_layers (List[int]): Sizes of hidden layers.
            action_space_size (int): Size of the action space.
        """
        super(PolicyNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.action_space_size = action_space_size
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Policy head
        self.policy_head = nn.Linear(prev_size, action_space_size)
        
        # Value head for actor-critic methods
        self.value_head = nn.Linear(prev_size, 1)
        
        self.shared_network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                Policy logits and value estimate.
        """
        features = self.shared_network(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value
    
    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities from the policy.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Action probabilities.
        """
        policy_logits, _ = self.forward(x)
        return F.softmax(policy_logits, dim=-1)
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate from the policy.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Value estimate.
        """
        _, value = self.forward(x)
        return value


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network for deep reinforcement learning.
    
    This network architecture separates the state value and advantage functions,
    which can improve learning efficiency for algorithms like DQN.
    
    Attributes:
        input_size (int): Size of the input feature vector.
        hidden_layers (List[int]): Sizes of hidden layers.
        action_space_size (int): Size of the action space.
    """
    
    def __init__(self, input_size: int, hidden_layers: List[int], action_space_size: int):
        """
        Initialize a dueling Q-network.
        
        Args:
            input_size (int): Size of the input feature vector.
            hidden_layers (List[int]): Sizes of hidden layers.
            action_space_size (int): Size of the action space.
        """
        super(DuelingQNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.action_space_size = action_space_size
        
        # Build shared layers
        shared_layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers[:-1]:  # All but last hidden layer
            shared_layers.append(nn.Linear(prev_size, hidden_size))
            shared_layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], action_space_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Q-values for each action.
        """
        features = self.shared_network(x)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages using the dueling architecture
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


class PokerTransformer(nn.Module):
    """
    Transformer-based model for poker decision-making.
    
    This model uses a transformer architecture to process the sequential nature
    of poker games, tracking betting patterns and action histories.
    
    Attributes:
        input_size (int): Size of the input feature vector.
        hidden_size (int): Size of the transformer hidden layers.
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        action_space_size (int): Size of the action space.
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                num_heads: int, action_space_size: int):
        """
        Initialize a poker transformer model.
        
        Args:
            input_size (int): Size of the input feature vector.
            hidden_size (int): Size of the transformer hidden layers.
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            action_space_size (int): Size of the action space.
        """
        super(PokerTransformer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.action_space_size = action_space_size
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Transformer layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_layers
        )
        
        # Output heads
        self.policy_head = nn.Linear(hidden_size, action_space_size)
        self.value_head = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, feature_dim).
            mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                Policy logits and value estimate.
        """
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Apply transformer
        features = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Use the output corresponding to the current state (last in sequence)
        current_features = features[:, -1, :]
        
        # Get policy and value outputs
        policy_logits = self.policy_head(current_features)
        value = self.value_head(current_features)
        
        return policy_logits, value


def create_value_network(input_size: int, hidden_layers: List[int], output_size: int = 1) -> ValueNetwork:
    """
    Create a value network.
    
    Args:
        input_size (int): Size of the input feature vector.
        hidden_layers (List[int]): Sizes of hidden layers.
        output_size (int, optional): Size of the output. Defaults to 1.
        
    Returns:
        ValueNetwork: Initialized value network.
    """
    return ValueNetwork(input_size, hidden_layers, output_size)


def create_policy_network(input_size: int, hidden_layers: List[int], action_space_size: int) -> PolicyNetwork:
    """
    Create a policy network.
    
    Args:
        input_size (int): Size of the input feature vector.
        hidden_layers (List[int]): Sizes of hidden layers.
        action_space_size (int): Size of the action space.
        
    Returns:
        PolicyNetwork: Initialized policy network.
    """
    return PolicyNetwork(input_size, hidden_layers, action_space_size)


def create_dueling_q_network(input_size: int, hidden_layers: List[int], action_space_size: int) -> DuelingQNetwork:
    """
    Create a dueling Q-network.
    
    Args:
        input_size (int): Size of the input feature vector.
        hidden_layers (List[int]): Sizes of hidden layers.
        action_space_size (int): Size of the action space.
        
    Returns:
        DuelingQNetwork: Initialized dueling Q-network.
    """
    return DuelingQNetwork(input_size, hidden_layers, action_space_size)


def create_poker_transformer(input_size: int, hidden_size: int, num_layers: int, 
                           num_heads: int, action_space_size: int) -> PokerTransformer:
    """
    Create a poker transformer model.
    
    Args:
        input_size (int): Size of the input feature vector.
        hidden_size (int): Size of the transformer hidden layers.
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        action_space_size (int): Size of the action space.
        
    Returns:
        PokerTransformer: Initialized poker transformer model.
    """
    return PokerTransformer(input_size, hidden_size, num_layers, num_heads, action_space_size)