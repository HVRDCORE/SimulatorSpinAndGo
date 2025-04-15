"""
Module for saving and loading neural network models in the poker AI framework.

This module provides utility functions for serializing and deserializing
trained models, which is essential for long-running training processes
and for deploying trained agents.
"""

import os
import torch
import json
import numpy as np
from typing import Dict, Any, Optional, Union
from datetime import datetime

from pokersim.ml.models import (
    ValueNetwork, PolicyNetwork, DuelingQNetwork, PokerTransformer,
    create_value_network, create_policy_network, create_dueling_q_network, create_poker_transformer
)


def save_model(model: torch.nn.Module, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save a PyTorch model to disk.
    
    Args:
        model (torch.nn.Module): The model to save.
        path (str): Path where the model will be saved.
        metadata (Optional[Dict[str, Any]], optional): Additional metadata to save. 
                                                      Defaults to None.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Default metadata if none provided
    if metadata is None:
        metadata = {}
    
    # Add timestamp
    metadata['timestamp'] = datetime.now().isoformat()
    metadata['model_type'] = model.__class__.__name__
    
    # Save model state
    save_dict = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }
    
    torch.save(save_dict, path)
    
    # Save metadata separately as JSON for easy inspection
    json_path = path + '.json'
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_model(path: str, model_type: Optional[str] = None) -> tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load a PyTorch model from disk.
    
    Args:
        path (str): Path where the model is saved.
        model_type (Optional[str], optional): Type of model to load. 
                                            If None, will be inferred from metadata. 
                                            Defaults to None.
    
    Returns:
        tuple[torch.nn.Module, Dict[str, Any]]: The loaded model and its metadata.
        
    Raises:
        ValueError: If model_type is not specified and cannot be inferred.
        FileNotFoundError: If the model file doesn't exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    
    # Load saved data
    saved_data = torch.load(path, map_location=torch.device('cpu'))
    model_state_dict = saved_data['model_state_dict']
    metadata = saved_data.get('metadata', {})
    
    # Determine model type
    if model_type is None:
        model_type = metadata.get('model_type')
        if model_type is None:
            raise ValueError("Model type not specified and cannot be inferred from metadata")
    
    # Create a new model instance based on the type
    if model_type == 'ValueNetwork':
        input_size = metadata.get('input_size', 128)
        hidden_layers = metadata.get('hidden_layers', [256, 256])
        output_size = metadata.get('output_size', 1)
        model = create_value_network(input_size, hidden_layers, output_size)
    
    elif model_type == 'PolicyNetwork':
        input_size = metadata.get('input_size', 128)
        hidden_layers = metadata.get('hidden_layers', [256, 256])
        action_space_size = metadata.get('action_space_size', 5)
        model = create_policy_network(input_size, hidden_layers, action_space_size)
    
    elif model_type == 'DuelingQNetwork':
        input_size = metadata.get('input_size', 128)
        hidden_layers = metadata.get('hidden_layers', [256, 256])
        action_space_size = metadata.get('action_space_size', 5)
        model = create_dueling_q_network(input_size, hidden_layers, action_space_size)
    
    elif model_type == 'PokerTransformer':
        input_size = metadata.get('input_size', 128)
        hidden_size = metadata.get('hidden_size', 256)
        num_layers = metadata.get('num_layers', 4)
        num_heads = metadata.get('num_heads', 8)
        action_space_size = metadata.get('action_space_size', 5)
        model = create_poker_transformer(input_size, hidden_size, num_layers, num_heads, action_space_size)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load the state dict
    model.load_state_dict(model_state_dict)
    
    return model, metadata


def save_agent(agent: Any, path: str, agent_type: str, extras: Optional[Dict[str, Any]] = None) -> None:
    """
    Save a poker AI agent to disk, including its model and algorithm state.
    
    Args:
        agent (Any): The agent to save (DeepCFR, PPO, etc.)
        path (str): Path where the agent will be saved.
        agent_type (str): Type of the agent (e.g., 'deep_cfr', 'ppo')
        extras (Optional[Dict[str, Any]], optional): Additional data to save.
                                                    Defaults to None.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Default extras if none provided
    if extras is None:
        extras = {}
    
    # Create metadata
    metadata = {
        'agent_type': agent_type,
        'timestamp': datetime.now().isoformat(),
        **extras
    }
    
    # Create agent-specific save dictionary
    if agent_type == 'deep_cfr':
        # For DeepCFR, save the value network
        model_path = f"{path}_value_network.pt"
        save_model(
            agent.value_network, 
            model_path, 
            {
                'input_dim': agent.input_dim,
                'action_dim': agent.action_dim,
                'game_type': agent.game_type,
                'num_players': agent.num_players,
                'iteration': agent.iteration
            }
        )
        metadata['model_path'] = model_path
        
        # Save additional algorithm state
        algorithm_state = {
            'memory_size': agent.memory_size,
            'batch_size': agent.batch_size,
            'iteration': agent.iteration,
            'advantage_losses': agent.advantage_losses
        }
        metadata['algorithm_state'] = algorithm_state
    
    elif agent_type == 'ppo':
        # For PPO, save the policy network
        model_path = f"{path}_policy_network.pt"
        save_model(
            agent.model, 
            model_path, 
            {
                'game_type': agent.game_type,
                'num_players': agent.num_players,
                'episode': agent.episode
            }
        )
        metadata['model_path'] = model_path
        
        # Save additional algorithm state
        algorithm_state = {
            'clip_ratio': agent.clip_ratio,
            'value_coef': agent.value_coef,
            'entropy_coef': agent.entropy_coef,
            'batch_size': agent.batch_size,
            'gae_lambda': agent.gae_lambda,
            'gamma': agent.gamma,
            'episode': agent.episode,
            'policy_losses': agent.policy_losses,
            'value_losses': agent.value_losses,
            'entropy_losses': agent.entropy_losses
        }
        metadata['algorithm_state'] = algorithm_state
    
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    # Save agent metadata
    with open(f"{path}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def load_agent(path: str, agent_type: Optional[str] = None) -> Any:
    """
    Load a poker AI agent from disk.
    
    Args:
        path (str): Base path where the agent is saved.
        agent_type (Optional[str], optional): Type of agent to load.
                                           If None, will be inferred from metadata.
                                           Defaults to None.
    
    Returns:
        Any: The loaded agent.
        
    Raises:
        ValueError: If agent_type is not specified and cannot be inferred.
        FileNotFoundError: If the agent files don't exist.
    """
    metadata_path = f"{path}_metadata.json"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Agent metadata not found at {metadata_path}")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Determine agent type
    if agent_type is None:
        agent_type = metadata.get('agent_type')
        if agent_type is None:
            raise ValueError("Agent type not specified and cannot be inferred from metadata")
    
    # Load the agent based on type
    if agent_type == 'deep_cfr':
        from pokersim.algorithms.deep_cfr import DeepCFR
        
        # Load the value network
        model_path = metadata.get('model_path')
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Value network model not found at {model_path}")
        
        value_network, model_metadata = load_model(model_path)
        
        # Create a new DeepCFR instance
        algorithm_state = metadata.get('algorithm_state', {})
        agent = DeepCFR(
            value_network=value_network,
            input_dim=model_metadata.get('input_dim', 128),
            action_dim=model_metadata.get('action_dim', 5),
            game_type=model_metadata.get('game_type', 'spingo'),
            num_players=model_metadata.get('num_players', 3),
            memory_size=algorithm_state.get('memory_size', 10000),
            batch_size=algorithm_state.get('batch_size', 128)
        )
        
        # Restore algorithm state
        agent.iteration = algorithm_state.get('iteration', 0)
        agent.advantage_losses = algorithm_state.get('advantage_losses', [])
    
    elif agent_type == 'ppo':
        from pokersim.algorithms.ppo import PPO
        
        # Load the policy network
        model_path = metadata.get('model_path')
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Policy network model not found at {model_path}")
        
        policy_network, model_metadata = load_model(model_path)
        
        # Create optimizer for the policy network
        optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.0001)
        
        # Create a new PPO instance
        algorithm_state = metadata.get('algorithm_state', {})
        agent = PPO(
            model=policy_network,
            optimizer=optimizer,
            game_type=model_metadata.get('game_type', 'spingo'),
            num_players=model_metadata.get('num_players', 3),
            clip_ratio=algorithm_state.get('clip_ratio', 0.2),
            value_coef=algorithm_state.get('value_coef', 0.5),
            entropy_coef=algorithm_state.get('entropy_coef', 0.01),
            batch_size=algorithm_state.get('batch_size', 64),
            gae_lambda=algorithm_state.get('gae_lambda', 0.95),
            gamma=algorithm_state.get('gamma', 0.99)
        )
        
        # Restore algorithm state
        agent.episode = algorithm_state.get('episode', 0)
        agent.policy_losses = algorithm_state.get('policy_losses', [])
        agent.value_losses = algorithm_state.get('value_losses', [])
        agent.entropy_losses = algorithm_state.get('entropy_losses', [])
    
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    return agent


def list_saved_models(directory: str) -> Dict[str, Any]:
    """
    List all saved models in a directory.
    
    Args:
        directory (str): Directory to search for saved models.
    
    Returns:
        Dict[str, Any]: Dictionary of model files and their metadata.
    """
    result = {}
    
    if not os.path.exists(directory):
        return result
    
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            path = os.path.join(directory, filename)
            json_path = path + '.json'
            
            if os.path.exists(json_path):
                # Read metadata
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                
                result[filename] = metadata
    
    return result


def list_saved_agents(directory: str) -> Dict[str, Any]:
    """
    List all saved agents in a directory.
    
    Args:
        directory (str): Directory to search for saved agents.
    
    Returns:
        Dict[str, Any]: Dictionary of agent files and their metadata.
    """
    result = {}
    
    if not os.path.exists(directory):
        return result
    
    for filename in os.listdir(directory):
        if filename.endswith("_metadata.json"):
            path = os.path.join(directory, filename)
            
            # Read metadata
            with open(path, 'r') as f:
                metadata = json.load(f)
            
            result[filename.replace("_metadata.json", "")] = metadata
    
    return result