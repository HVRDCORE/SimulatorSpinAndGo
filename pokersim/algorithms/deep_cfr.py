"""
Deep Counterfactual Regret Minimization (Deep CFR) algorithm for the poker simulator.

This module implements the Deep CFR algorithm, which uses neural networks to
approximate the advantage functions in CFR, allowing for more efficient computation
and better generalization across game states.
"""

import os
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
import time
import random
from collections import defaultdict, deque

from pokersim.config.config_manager import get_config
from pokersim.logging.game_logger import get_logger
from pokersim.ml.model_io import get_model_io
from pokersim.utils.gpu_optimization import optimize_tensor_for_gpu

# Conditional imports for PyTorch and TensorFlow
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Configure logging
logger = logging.getLogger("pokersim.algorithms.deep_cfr")


class DeepCFRSolver:
    """
    Deep Counterfactual Regret Minimization (Deep CFR) solver for poker games.
    
    This class implements Deep CFR, a variant of CFR that uses neural networks
    to approximate the advantage functions, allowing for better scaling and
    generalization. It maintains separate networks for each player's advantage
    function and a policy network for the final strategy.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings.
        game_logger: Game logger instance.
        model_io: Model I/O manager.
        game_state_class: Class for the game state.
        advantage_networks (Dict[int, Any]): Advantage networks for each player.
        strategy_network (Any): Strategy network (policy).
        advantage_buffers (Dict[int, List]): Advantage memories for each player.
        strategy_buffer (List): Strategy memory.
        iterations (int): Number of iterations performed.
        framework (str): Deep learning framework being used ('pytorch' or 'tensorflow').
    """
    
    def __init__(self, game_state_class: Any, framework: str = "auto"):
        """
        Initialize the Deep CFR solver.
        
        Args:
            game_state_class: Class for the game state.
            framework (str, optional): Deep learning framework to use. Defaults to "auto".
        """
        # Get configuration
        config = get_config()
        self.config = config.to_dict()
        
        # Set up logging and model I/O
        self.game_logger = get_logger()
        self.model_io = get_model_io()
        
        # Determine framework
        if framework == "auto":
            if TORCH_AVAILABLE:
                self.framework = "pytorch"
            elif TF_AVAILABLE:
                self.framework = "tensorflow"
            else:
                raise ImportError("No supported deep learning framework found")
        else:
            if framework == "pytorch" and not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
            elif framework == "tensorflow" and not TF_AVAILABLE:
                raise ImportError("TensorFlow not available")
            self.framework = framework
        
        # Initialize solver state
        self.game_state_class = game_state_class
        self.advantage_networks = {}
        self.strategy_network = None
        self.advantage_buffers = {}
        self.strategy_buffer = []
        self.iterations = 0
        
        # Get neural network parameters from config
        self.batch_size = self.config["training"]["batch_size"]
        self.learning_rate = self.config["training"]["learning_rate"]
        
        # Create an initial game state to determine input dimensions
        initial_state = self.game_state_class()
        
        # Determine number of players and input dimensions
        self.num_players = initial_state.get_num_players()
        info_set_dim = initial_state.get_info_set_dimension()
        num_actions = len(initial_state.get_legal_actions())
        
        # Initialize networks and buffers for each player
        for player in range(self.num_players):
            # Create advantage network
            self.advantage_networks[player] = self._create_advantage_network(
                input_dim=info_set_dim,
                output_dim=num_actions
            )
            
            # Create advantage buffer
            self.advantage_buffers[player] = deque(maxlen=1000000)  # Limit buffer size
        
        # Create strategy network
        self.strategy_network = self._create_strategy_network(
            input_dim=info_set_dim,
            output_dim=num_actions
        )
    
    def _create_advantage_network(self, input_dim: int, output_dim: int) -> Any:
        """
        Create an advantage network.
        
        Args:
            input_dim (int): Input dimension (info set dimension).
            output_dim (int): Output dimension (number of actions).
        
        Returns:
            Any: Advantage network.
        """
        if self.framework == "pytorch":
            # Create PyTorch network
            network = AdvantageNetworkPyTorch(
                input_dim=input_dim,
                hidden_dim=self.config["model"]["value_network"]["hidden_layers"][0],
                output_dim=output_dim
            )
            
            # Define optimizer
            optimizer = optim.Adam(network.parameters(), lr=self.learning_rate)
            
            return {
                "network": network,
                "optimizer": optimizer
            }
        
        elif self.framework == "tensorflow":
            # Create TensorFlow network
            network = AdvantageNetworkTensorFlow(
                input_dim=input_dim,
                hidden_dim=self.config["model"]["value_network"]["hidden_layers"][0],
                output_dim=output_dim
            )
            
            # Define optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            
            return {
                "network": network,
                "optimizer": optimizer
            }
    
    def _create_strategy_network(self, input_dim: int, output_dim: int) -> Any:
        """
        Create a strategy network.
        
        Args:
            input_dim (int): Input dimension (info set dimension).
            output_dim (int): Output dimension (number of actions).
        
        Returns:
            Any: Strategy network.
        """
        if self.framework == "pytorch":
            # Create PyTorch network
            network = StrategyNetworkPyTorch(
                input_dim=input_dim,
                hidden_dim=self.config["model"]["policy_network"]["hidden_layers"][0],
                output_dim=output_dim
            )
            
            # Define optimizer
            optimizer = optim.Adam(network.parameters(), lr=self.learning_rate)
            
            return {
                "network": network,
                "optimizer": optimizer
            }
        
        elif self.framework == "tensorflow":
            # Create TensorFlow network
            network = StrategyNetworkTensorFlow(
                input_dim=input_dim,
                hidden_dim=self.config["model"]["policy_network"]["hidden_layers"][0],
                output_dim=output_dim
            )
            
            # Define optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            
            return {
                "network": network,
                "optimizer": optimizer
            }
    
    def _forward(self, network: Any, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through a network.
        
        Args:
            network (Any): Neural network.
            inputs (np.ndarray): Input data.
        
        Returns:
            np.ndarray: Network output.
        """
        if self.framework == "pytorch":
            # Convert inputs to PyTorch tensor
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
            
            # Optimize for GPU if available
            inputs_tensor = inputs_tensor.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = network["network"](inputs_tensor)
            
            # Convert back to numpy
            return outputs.cpu().numpy()
        
        elif self.framework == "tensorflow":
            # Convert inputs to TensorFlow tensor
            inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
            
            # Optimize for GPU if available
            inputs_tensor = tf.convert_to_tensor(inputs_tensor, dtype=tf.float32)
            
            # Forward pass
            outputs = network["network"](inputs_tensor, training=False)
            
            # Convert back to numpy
            return outputs.numpy()
    
    def _train_network(self, network: Any, inputs: np.ndarray, targets: np.ndarray, 
                     weights: Optional[np.ndarray] = None) -> float:
        """
        Train a network on a batch of data.
        
        Args:
            network (Any): Neural network and optimizer.
            inputs (np.ndarray): Input data.
            targets (np.ndarray): Target values.
            weights (Optional[np.ndarray], optional): Sample weights. Defaults to None.
        
        Returns:
            float: Loss value.
        """
        if self.framework == "pytorch":
            # Convert data to PyTorch tensors
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
            targets_tensor = torch.tensor(targets, dtype=torch.float32)
            
            # Add weights if provided
            if weights is not None:
                weights_tensor = torch.tensor(weights, dtype=torch.float32)
            else:
                weights_tensor = torch.ones_like(targets_tensor[:, 0])
            
            # Optimize for GPU if available
            inputs_tensor = optimize_tensor_for_gpu(inputs_tensor)
            targets_tensor = optimize_tensor_for_gpu(targets_tensor)
            weights_tensor = optimize_tensor_for_gpu(weights_tensor)
            
            # Zero gradients
            network["optimizer"].zero_grad()
            
            # Forward pass
            outputs = network["network"](inputs_tensor)
            
            # Compute weighted MSE loss
            losses = F.mse_loss(outputs, targets_tensor, reduction='none')
            loss = torch.mean(torch.sum(losses, dim=1) * weights_tensor)
            
            # Backward pass and optimization
            loss.backward()
            network["optimizer"].step()
            
            return loss.item()
        
        elif self.framework == "tensorflow":
            # Convert data to TensorFlow tensors
            inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
            targets_tensor = tf.convert_to_tensor(targets, dtype=tf.float32)
            
            # Add weights if provided
            if weights is not None:
                weights_tensor = tf.convert_to_tensor(weights, dtype=tf.float32)
            else:
                weights_tensor = tf.ones_like(targets_tensor[:, 0])
            
            # Optimize for GPU if available
            inputs_tensor = optimize_tensor_for_gpu(inputs_tensor)
            targets_tensor = optimize_tensor_for_gpu(targets_tensor)
            weights_tensor = optimize_tensor_for_gpu(weights_tensor)
            
            # Training step
            with tf.GradientTape() as tape:
                outputs = network["network"](inputs_tensor, training=True)
                losses = tf.reduce_sum(tf.keras.losses.mean_squared_error(
                    targets_tensor, outputs), axis=1)
                loss = tf.reduce_mean(losses * weights_tensor)
            
            # Compute gradients and apply
            gradients = tape.gradient(loss, network["network"].trainable_variables)
            network["optimizer"].apply_gradients(zip(gradients, network["network"].trainable_variables))
            
            return loss.numpy()
    
    def traverse_game_tree(self, state: Any, reach_probs: List[float], player: int, 
                         iteration: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Traverse the game tree and collect advantage samples for a player.
        
        Args:
            state: Current game state.
            reach_probs (List[float]): Reach probabilities for each player.
            player (int): Player to collect samples for.
            iteration (int): Current iteration.
        
        Returns:
            List[Tuple[np.ndarray, np.ndarray, float]]: Collected samples.
        """
        if state.is_terminal():
            return []
        
        if state.is_chance_node():
            outcomes = state.get_chance_outcomes()
            samples = []
            
            for action, prob in outcomes:
                next_state = state.apply_action(action)
                child_samples = self.traverse_game_tree(next_state, reach_probs, player, iteration)
                samples.extend(child_samples)
            
            return samples
        
        current_player = state.get_current_player()
        info_set_key = state.get_info_set_key()
        info_set_vector = state.get_info_set_vector()
        legal_actions = state.get_legal_actions()
        num_actions = len(legal_actions)
        samples = []
        
        # Skip if not a player's turn
        if current_player != player:
            # Use current strategy to make a decision
            action_probs = self.get_action_probabilities(state)
            action_idx = np.random.choice(len(action_probs), p=action_probs)
            action = legal_actions[action_idx]
            
            # Update reach probability
            new_reach_probs = reach_probs.copy()
            new_reach_probs[current_player] *= action_probs[action_idx]
            
            # Recurse on the new state
            next_state = state.apply_action(action)
            child_samples = self.traverse_game_tree(next_state, new_reach_probs, player, iteration)
            samples.extend(child_samples)
            
            return samples
        
        # For player's turn, compute advantages
        # Get values for each action using the advantage network
        action_values = np.zeros(num_actions)
        
        for i, action in enumerate(legal_actions):
            # Apply action
            next_state = state.apply_action(action)
            
            if next_state.is_terminal():
                # Get immediate value for terminal states
                action_values[i] = next_state.get_utility()
            else:
                # Recurse for non-terminal states
                new_reach_probs = reach_probs.copy()
                new_reach_probs[current_player] = 1.0  # Set to 1 since we're computing counterfactual values
                
                # Get child samples (will be added to samples later)
                child_samples = self.traverse_game_tree(next_state, new_reach_probs, player, iteration)
                samples.extend(child_samples)
                
                # Compute value for this action
                action_values[i] = next_state.get_expected_value(player)
        
        # Compute average value
        cf_value = np.mean(action_values)
        
        # Compute advantages
        advantages = action_values - cf_value
        
        # Create one-hot target vector
        target = np.zeros(num_actions)
        
        # Fill non-zero values for legal actions
        for i in range(num_actions):
            target[i] = advantages[i]
        
        # Add sample: (info_set_vector, target, weight)
        weight = np.prod(reach_probs) / reach_probs[player]  # Reach probability of other players
        sample = (info_set_vector, target, weight)
        samples.append(sample)
        
        # Add sample to advantage buffer
        self.advantage_buffers[player].append((info_set_vector, target, weight, iteration))
        
        # Add to strategy buffer if using average strategy
        strategy_target = np.zeros(num_actions)
        
        # Use regret matching to compute strategy
        positive_advantages = np.maximum(0, advantages)
        advantage_sum = np.sum(positive_advantages)
        
        if advantage_sum > 0:
            strategy_target = positive_advantages / advantage_sum
        else:
            strategy_target = np.ones(num_actions) / num_actions  # Uniform strategy
        
        # Add strategy sample
        self.strategy_buffer.append((info_set_vector, strategy_target, weight, iteration))
        
        return samples
    
    def train_advantage_network(self, player: int, batch_size: Optional[int] = None) -> float:
        """
        Train the advantage network for a player.
        
        Args:
            player (int): Player to train.
            batch_size (Optional[int], optional): Batch size. Defaults to None.
        
        Returns:
            float: Training loss.
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Skip if not enough samples
        if len(self.advantage_buffers[player]) < batch_size:
            return 0.0
        
        # Sample from the buffer
        samples = random.sample(self.advantage_buffers[player], batch_size)
        
        # Unpack samples
        info_sets = []
        advantages = []
        weights = []
        iterations = []
        
        for info_set, advantage, weight, iteration in samples:
            info_sets.append(info_set)
            advantages.append(advantage)
            weights.append(weight)
            iterations.append(iteration)
        
        # Convert to numpy arrays
        info_sets = np.array(info_sets)
        advantages = np.array(advantages)
        weights = np.array(weights)
        iterations = np.array(iterations)
        
        # Apply iteration weighting (more recent iterations have higher weight)
        iteration_weights = (iterations / self.iterations) ** 2
        combined_weights = weights * iteration_weights
        
        # Train the network
        loss = self._train_network(
            network=self.advantage_networks[player],
            inputs=info_sets,
            targets=advantages,
            weights=combined_weights
        )
        
        return loss
    
    def train_strategy_network(self, batch_size: Optional[int] = None) -> float:
        """
        Train the strategy network.
        
        Args:
            batch_size (Optional[int], optional): Batch size. Defaults to None.
        
        Returns:
            float: Training loss.
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Skip if not enough samples
        if len(self.strategy_buffer) < batch_size:
            return 0.0
        
        # Sample from the buffer
        samples = random.sample(self.strategy_buffer, batch_size)
        
        # Unpack samples
        info_sets = []
        strategies = []
        weights = []
        iterations = []
        
        for info_set, strategy, weight, iteration in samples:
            info_sets.append(info_set)
            strategies.append(strategy)
            weights.append(weight)
            iterations.append(iteration)
        
        # Convert to numpy arrays
        info_sets = np.array(info_sets)
        strategies = np.array(strategies)
        weights = np.array(weights)
        iterations = np.array(iterations)
        
        # Apply iteration weighting (more recent iterations have higher weight)
        iteration_weights = iterations / self.iterations
        combined_weights = weights * iteration_weights
        
        # Train the network
        loss = self._train_network(
            network=self.strategy_network,
            inputs=info_sets,
            targets=strategies,
            weights=combined_weights
        )
        
        return loss
    
    def train(self, num_iterations: int, num_traversals: int = 100, 
            callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Train the Deep CFR solver.
        
        Args:
            num_iterations (int): Number of iterations.
            num_traversals (int, optional): Number of tree traversals per iteration. Defaults to 100.
            callback (Optional[Callable], optional): Callback function. Defaults to None.
        
        Returns:
            Dict[str, Any]: Training results.
        """
        start_time = time.time()
        metrics = []
        
        # Training loop
        for i in range(num_iterations):
            # Update iteration counter
            self.iterations += 1
            
            # Traverse game tree for each player
            for player in range(self.num_players):
                for _ in range(num_traversals):
                    # Initialize game state
                    initial_state = self.game_state_class()
                    
                    # Determine initial reach probabilities
                    reach_probs = [1.0] * self.num_players
                    
                    # Traverse game tree
                    self.traverse_game_tree(initial_state, reach_probs, player, self.iterations)
                
                # Train advantage network for this player
                advantage_loss = self.train_advantage_network(player)
                
                # Log progress
                logger.debug(f"Iteration {i+1}/{num_iterations} - Player {player} - Advantage Loss: {advantage_loss:.6f}")
            
            # Train strategy network
            strategy_loss = self.train_strategy_network()
            
            # Log progress
            if (i + 1) % max(1, num_iterations // 10) == 0:
                elapsed = time.time() - start_time
                logger.info(f"Iteration {i+1}/{num_iterations} - Strategy Loss: {strategy_loss:.6f} - Time: {elapsed:.2f}s")
                
                # Record metrics
                metric = {
                    "iteration": i + 1,
                    "strategy_loss": float(strategy_loss),
                    "time": elapsed
                }
                
                metrics.append(metric)
                
                # Log to game logger
                self.game_logger.log_training_metrics("DeepCFR", metric, i + 1)
            
            # Call callback if provided
            if callback is not None:
                callback(i, strategy_loss, self)
        
        total_time = time.time() - start_time
        logger.info(f"Deep CFR training completed in {total_time:.2f} seconds")
        
        # Return results
        return {
            "algorithm": "DeepCFR",
            "iterations": num_iterations,
            "total_time": total_time,
            "advantage_buffer_sizes": {p: len(self.advantage_buffers[p]) for p in range(self.num_players)},
            "strategy_buffer_size": len(self.strategy_buffer),
            "metrics": metrics
        }
    
    def get_action_probabilities(self, state: Any) -> np.ndarray:
        """
        Get action probabilities for a given state.
        
        Args:
            state: Game state.
        
        Returns:
            np.ndarray: Probability distribution over actions.
        """
        # Get information set
        info_set_vector = state.get_info_set_vector()
        
        # Reshape for batch prediction
        inputs = np.array([info_set_vector])
        
        # Forward pass through strategy network
        outputs = self._forward(self.strategy_network, inputs)[0]
        
        # Get valid actions mask
        legal_actions = state.get_legal_actions()
        action_mask = np.zeros(outputs.shape)
        
        for i, _ in enumerate(legal_actions):
            action_mask[i] = 1
        
        # Apply mask and normalize
        masked_outputs = outputs * action_mask
        action_sum = np.sum(masked_outputs)
        
        if action_sum > 0:
            return masked_outputs / action_sum
        else:
            # Uniform distribution if all outputs are zero
            return action_mask / np.sum(action_mask)
    
    def get_advantage_values(self, state: Any, player: int) -> np.ndarray:
        """
        Get advantage values for a given state and player.
        
        Args:
            state: Game state.
            player (int): Player index.
        
        Returns:
            np.ndarray: Advantage values for each action.
        """
        # Get information set
        info_set_vector = state.get_info_set_vector()
        
        # Reshape for batch prediction
        inputs = np.array([info_set_vector])
        
        # Forward pass through advantage network
        outputs = self._forward(self.advantage_networks[player], inputs)[0]
        
        return outputs
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the Deep CFR model.
        
        Args:
            filepath (Optional[str], optional): Directory to save the model. Defaults to None.
        
        Returns:
            str: Path to the saved model.
        """
        if filepath is None:
            # Generate default filepath
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filepath = os.path.join(self.config["agents"]["save_dir"], f"deep_cfr_model_{timestamp}")
        
        # Create directory if it doesn't exist
        os.makedirs(filepath, exist_ok=True)
        
        # Save metadata
        metadata = {
            "algorithm": "DeepCFR",
            "framework": self.framework,
            "iterations": self.iterations,
            "timestamp": time.time(),
            "num_players": self.num_players
        }
        
        with open(os.path.join(filepath, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save advantage networks
        for player in range(self.num_players):
            self._save_network(
                os.path.join(filepath, f"advantage_network_{player}"),
                self.advantage_networks[player]["network"]
            )
        
        # Save strategy network
        self._save_network(
            os.path.join(filepath, "strategy_network"),
            self.strategy_network["network"]
        )
        
        logger.info(f"Saved Deep CFR model to {filepath}")
        return filepath
    
    def load(self, filepath: str) -> bool:
        """
        Load the Deep CFR model.
        
        Args:
            filepath (str): Path to the saved model.
        
        Returns:
            bool: Whether the load was successful.
        """
        try:
            # Load metadata
            with open(os.path.join(filepath, "metadata.json"), 'r') as f:
                metadata = json.load(f)
            
            # Check algorithm
            if metadata.get("algorithm") != "DeepCFR":
                logger.error(f"Invalid algorithm in {filepath}, expected DeepCFR")
                return False
            
            # Check framework
            if metadata.get("framework") != self.framework:
                logger.error(f"Framework mismatch: model {metadata.get('framework')}, current {self.framework}")
                return False
            
            # Load iterations
            self.iterations = metadata.get("iterations", 0)
            
            # Load advantage networks
            for player in range(self.num_players):
                self._load_network(
                    os.path.join(filepath, f"advantage_network_{player}"),
                    self.advantage_networks[player]["network"]
                )
            
            # Load strategy network
            self._load_network(
                os.path.join(filepath, "strategy_network"),
                self.strategy_network["network"]
            )
            
            logger.info(f"Loaded Deep CFR model from {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading Deep CFR model from {filepath}: {e}")
            return False
    
    def _save_network(self, filepath: str, network: Any):
        """
        Save a neural network.
        
        Args:
            filepath (str): Path to save the network.
            network: Neural network.
        """
        if self.framework == "pytorch":
            # Save PyTorch model
            torch.save(network.state_dict(), f"{filepath}.pt")
        
        elif self.framework == "tensorflow":
            # Save TensorFlow model
            network.save(f"{filepath}.keras")
    
    def _load_network(self, filepath: str, network: Any):
        """
        Load a neural network.
        
        Args:
            filepath (str): Path to the saved network.
            network: Neural network.
        """
        if self.framework == "pytorch":
            # Load PyTorch model
            state_dict = torch.load(f"{filepath}.pt")
            network.load_state_dict(state_dict)
        
        elif self.framework == "tensorflow":
            # Load TensorFlow model
            loaded_model = tf.keras.models.load_model(f"{filepath}.keras")
            network.set_weights(loaded_model.get_weights())


# Neural network implementations for PyTorch
class AdvantageNetworkPyTorch(nn.Module):
    """PyTorch implementation of the advantage network."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize the advantage network.
        
        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class StrategyNetworkPyTorch(nn.Module):
    """PyTorch implementation of the strategy network."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize the strategy network.
        
        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


# Neural network implementations for TensorFlow
class AdvantageNetworkTensorFlow(tf.keras.Model):
    """TensorFlow implementation of the advantage network."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize the advantage network.
        
        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
        """
        super().__init__()
        
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc3 = tf.keras.layers.Dense(output_dim)
    
    def call(self, inputs, training=False):
        """Forward pass."""
        x = self.fc1(inputs)
        x = self.fc2(x)
        return self.fc3(x)


class StrategyNetworkTensorFlow(tf.keras.Model):
    """TensorFlow implementation of the strategy network."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize the strategy network.
        
        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
        """
        super().__init__()
        
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc3 = tf.keras.layers.Dense(output_dim)
        self.softmax = tf.keras.layers.Softmax()
    
    def call(self, inputs, training=False):
        """Forward pass."""
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.softmax(x)