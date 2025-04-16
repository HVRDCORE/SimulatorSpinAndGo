"""
Proximal Policy Optimization (PPO) algorithm for poker simulation.

This module implements PPO, a state-of-the-art reinforcement learning algorithm,
for training poker agents. PPO offers stable learning with good sample efficiency
and is effective for environments with continuous action spaces.
"""

import numpy as np
import random
import time
from typing import Dict, List, Tuple, Any, Optional, Callable
import os
import copy

# ML framework imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
    HAS_TF = True
except ImportError:
    HAS_TF = False

# Make sure at least one framework is available
if not HAS_TORCH and not HAS_TF:
    raise ImportError("Either PyTorch or TensorFlow is required for PPO implementation")


class PPOSolver:
    """
    Proximal Policy Optimization (PPO) solver for poker games.

    This class implements PPO for poker, using neural networks for policy and value
    functions. It supports both PyTorch and TensorFlow backends.

    Attributes:
        config (Dict[str, Any]): Configuration settings.
        game_state_class: Class for creating game states.
        framework (str): Deep learning framework being used ('pytorch' or 'tensorflow').
        actor_network: Policy network.
        critic_network: Value network.
        actor_optimizer: Optimizer for the policy network.
        critic_optimizer: Optimizer for the value network.
        trajectory_buffer: Storage for experience data.
        device: Device to run computations (CPU/GPU).
        iterations (int): Number of training iterations performed.
    """

    def __init__(self, game_state_class: Any, num_players: int = 2, framework: str = "auto"):
        """
        Initialize the PPO solver.

        Args:
            game_state_class: Class for creating game states.
            num_players (int, optional): Number of players in the game. Defaults to 2.
            framework (str, optional): Deep learning framework to use. Defaults to "auto".
        """
        self.game_state_class = game_state_class
        self.num_players = num_players

        # Determine framework
        if framework == "auto":
            if HAS_TORCH:
                self.framework = "pytorch"
            elif HAS_TF:
                self.framework = "tensorflow"
            else:
                raise ImportError("No supported ML framework found")
        else:
            if framework == "pytorch" and not HAS_TORCH:
                raise ImportError("PyTorch requested but not installed")
            elif framework == "tensorflow" and not HAS_TF:
                raise ImportError("TensorFlow requested but not installed")
            self.framework = framework

        # Create a temporary game state to determine dimensions
        temp_state = self.game_state_class(num_players=self.num_players, small_blind=1, big_blind=2)
        player_id = 0  # Perspective of first player
        obs = temp_state.get_observation(player_id)

        # Set up input and output dimensions
        feature_vector = temp_state.to_feature_vector(player_id)
        self.input_dim = feature_vector.shape[0]
        self.output_dim = len(temp_state.get_legal_actions())

        # Create networks and optimizers based on framework
        if self.framework == "pytorch":
            self._init_pytorch_components()
        else:  # tensorflow
            self._init_tensorflow_components()

        # PPO hyperparameters
        self.clip_ratio = 0.2  # PPO clipping parameter
        self.entropy_coef = 0.01  # Entropy coefficient
        self.value_coef = 0.5  # Value loss coefficient
        self.max_grad_norm = 0.5  # Gradient clipping

        # Setup trajectory buffer for experience collection
        self.trajectory_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }

        # Training stats
        self.iterations = 0
        self.metrics = []

    def _init_pytorch_components(self):
        """Initialize PyTorch networks and optimizer."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed")

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create networks
        self.actor_network = ActorNetworkPyTorch(self.input_dim, 128, self.output_dim).to(self.device)
        self.critic_network = CriticNetworkPyTorch(self.input_dim, 128).to(self.device)

        # Set up optimizers
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=1e-3)

    def _init_tensorflow_components(self):
        """Initialize TensorFlow networks and optimizer."""
        if not HAS_TF:
            raise ImportError("TensorFlow is not installed")

        # Create networks
        self.actor_network = ActorNetworkTensorFlow(self.input_dim, 128, self.output_dim)
        self.critic_network = CriticNetworkTensorFlow(self.input_dim, 128)

        # Set up optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # Use GPU if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.device = "/GPU:0"
            except RuntimeError as e:
                print(f"GPU error: {e}")
                self.device = "/CPU:0"
        else:
            self.device = "/CPU:0"

    def _forward_actor(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the actor network.

        Args:
            states (np.ndarray): State inputs.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Action probabilities and log probabilities.
        """
        if self.framework == "pytorch":
            # PyTorch forward pass
            states_tensor = torch.FloatTensor(states).to(self.device)
            action_probs = self.actor_network(states_tensor)
            dist = Categorical(action_probs)
            log_probs = dist.logits

            return action_probs.detach().cpu().numpy(), log_probs.detach().cpu().numpy()
        else:
            # TensorFlow forward pass
            with tf.device(self.device):
                states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
                action_probs = self.actor_network(states_tensor)
                log_probs = tf.math.log(action_probs + 1e-10)

                return action_probs.numpy(), log_probs.numpy()

    def _forward_critic(self, states: np.ndarray) -> np.ndarray:
        """
        Forward pass through the critic network.

        Args:
            states (np.ndarray): State inputs.

        Returns:
            np.ndarray: Value predictions.
        """
        if self.framework == "pytorch":
            # PyTorch forward pass
            states_tensor = torch.FloatTensor(states).to(self.device)
            values = self.critic_network(states_tensor)

            return values.detach().cpu().numpy().flatten()
        else:
            # TensorFlow forward pass
            with tf.device(self.device):
                states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
                values = self.critic_network(states_tensor)

                return values.numpy().flatten()

    def get_action(self, state: Any, player_id: int, deterministic: bool = False) -> Tuple[Any, float, float]:
        """
        Get an action for a given state.

        Args:
            state: Game state.
            player_id (int): Player ID.
            deterministic (bool, optional): Whether to use deterministic action selection. Defaults to False.

        Returns:
            Tuple[Any, float, float]: Selected action, action log probability, and value estimate.
        """
        # Convert state to feature vector
        state_vec = state.to_feature_vector(player_id).reshape(1, -1)

        # Get action probabilities and value
        action_probs, log_probs = self._forward_actor(state_vec)
        value = self._forward_critic(state_vec)[0]

        # Get legal actions
        legal_actions = state.get_legal_actions()
        if len(legal_actions) > self.output_dim:
            raise ValueError(
                f"Number of legal actions ({len(legal_actions)}) exceeds output dimension ({self.output_dim})")

        # Set probabilities of illegal actions to zero and renormalize
        legal_mask = np.zeros(self.output_dim)
        for i, action in enumerate(legal_actions):
            if i < len(legal_mask):
                legal_mask[i] = 1

        masked_probs = action_probs[0] * legal_mask
        if np.sum(masked_probs) > 0:
            masked_probs = masked_probs / np.sum(masked_probs)
        else:
            # Fallback to uniform distribution over legal actions
            masked_probs = legal_mask / np.sum(legal_mask)

        # Select action
        if deterministic:
            action_idx = np.argmax(masked_probs)
        else:
            action_idx = np.random.choice(len(masked_probs), p=masked_probs)

        # Get corresponding action object and log probability
        action = legal_actions[action_idx]
        log_prob = log_probs[0][action_idx]

        return action, log_prob, value

    def collect_trajectories(self, num_trajectories: int = 1000, max_steps: int = 100) -> Dict[str, List]:
        """
        Collect trajectories by playing games.

        Args:
            num_trajectories (int, optional): Number of trajectories to collect. Defaults to 1000.
            max_steps (int, optional): Maximum steps per trajectory. Defaults to 100.

        Returns:
            Dict[str, List]: Collected trajectories.
        """
        # Reset trajectory buffer
        trajectory_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }

        # Collect trajectories
        for _ in range(num_trajectories):
            # Initialize game state
            state = self.game_state_class(num_players=self.num_players, small_blind=1, big_blind=2)

            for step in range(max_steps):
                # Get current player
                player_id = state.current_player

                # Check if game is terminal
                if state.is_terminal():
                    break

                # Convert state to feature vector
                state_vec = state.to_feature_vector(player_id)

                # Get action from policy
                action, log_prob, value = self.get_action(state, player_id)

                # Apply action to get next state
                next_state = state.apply_action(action)

                # Get reward (only non-zero at terminal states)
                reward = 0.0
                if next_state.is_terminal():
                    rewards = next_state.get_rewards()
                    reward = rewards[player_id]

                # Store transition
                trajectory_buffer['states'].append(state_vec)
                trajectory_buffer['actions'].append(action)
                trajectory_buffer['rewards'].append(reward)
                trajectory_buffer['values'].append(value)
                trajectory_buffer['log_probs'].append(log_prob)
                trajectory_buffer['dones'].append(next_state.is_terminal())

                # Update state
                state = next_state

                # Break if terminal
                if state.is_terminal():
                    break

        # Convert lists to numpy arrays
        for key in trajectory_buffer:
            if key != 'actions':  # Keep actions as objects
                trajectory_buffer[key] = np.array(trajectory_buffer[key])

        return trajectory_buffer

    def compute_advantages(self, rewards: np.ndarray, values: np.ndarray,
                          dones: np.ndarray, gamma: float = 0.99,
                          lambda_gae: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        Args:
            rewards (np.ndarray): Rewards from trajectories.
            values (np.ndarray): Value estimates from trajectories.
            dones (np.ndarray): Terminal state flags.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            lambda_gae (float, optional): GAE parameter. Defaults to 0.95.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Returns and advantages.
        """
        # Initialize arrays
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        last_value = 0

        # Compute advantages in reverse order
        for t in reversed(range(len(rewards))):
            # Handle terminal states
            mask = 1.0 - dones[t]

            # Compute delta (TD error)
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * mask - values[t]

            # Compute advantage using GAE
            advantages[t] = delta + gamma * lambda_gae * last_advantage * mask
            last_advantage = advantages[t]

        # Compute returns
        returns = advantages + values

        return returns, advantages

    def update_policy(self, trajectories: Dict[str, Any],
                     epochs: int = 10, batch_size: int = 64) -> Dict[str, float]:
        """
        Update policy and value networks using PPO.

        Args:
            trajectories (Dict[str, Any]): Collected trajectories.
            epochs (int, optional): Number of update epochs. Defaults to 10.
            batch_size (int, optional): Batch size. Defaults to 64.

        Returns:
            Dict[str, float]: Training metrics.
        """
        # Extract data from trajectories
        states = np.array(trajectories['states'])
        rewards = np.array(trajectories['rewards'])
        values = np.array(trajectories['values'])
        log_probs_old = np.array(trajectories['log_probs'])
        dones = np.array(trajectories['dones'])

        # Get actions - these are complex objects
        actions = trajectories['actions']

        # Compute advantages and returns
        returns, advantages = self.compute_advantages(rewards, values, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training loop
        metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0
        }

        # Generate indices for all data points
        indices = np.arange(len(states))

        # Run multiple epochs
        for epoch in range(epochs):
            # Shuffle indices
            np.random.shuffle(indices)

            # Iterate over mini-batches
            for start_idx in range(0, len(indices), batch_size):
                # Extract mini-batch
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_states = states[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]

                # Use indices to select the correct actions
                batch_actions = [actions[i] for i in batch_indices]

                # Perform update based on framework
                if self.framework == "pytorch":
                    update_metrics = self._update_pytorch(
                        batch_states, batch_actions, batch_returns,
                        batch_advantages, batch_log_probs_old
                    )
                else:
                    update_metrics = self._update_tensorflow(
                        batch_states, batch_actions, batch_returns,
                        batch_advantages, batch_log_probs_old
                    )

                # Accumulate metrics
                for key, value in update_metrics.items():
                    metrics[key] += value / (len(indices) // batch_size + 1) / epochs

        # Update iteration counter
        self.iterations += 1

        # Store metrics
        metrics['iteration'] = self.iterations
        self.metrics.append(metrics)

        return metrics

    def _update_pytorch(self, states: np.ndarray, actions: List, returns: np.ndarray,
                      advantages: np.ndarray, old_log_probs: np.ndarray) -> Dict[str, float]:
        """
        Update policy and value networks using PyTorch.

        Args:
            states (np.ndarray): State inputs.
            actions (List): Action objects.
            returns (np.ndarray): Return targets.
            advantages (np.ndarray): Advantage estimates.
            old_log_probs (np.ndarray): Old log probabilities.

        Returns:
            Dict[str, float]: Training metrics.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for this method")

        # Convert numpy arrays to PyTorch tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)

        # Forward pass through actor network
        action_probs = self.actor_network(states_tensor)
        dist = Categorical(action_probs)

        # Get indices from actions (simplified)
        action_indices = torch.tensor([self._action_to_idx(action) for action in actions], device=self.device)

        # Get new log probabilities
        new_log_probs = dist.log_prob(action_indices)

        # Compute ratio for PPO
        ratio = torch.exp(new_log_probs - old_log_probs_tensor)

        # Compute surrogate objectives
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages_tensor
        actor_loss = -torch.min(surr1, surr2).mean()

        # Compute entropy bonus
        entropy = dist.entropy().mean()

        # Forward pass through critic network
        values_pred = self.critic_network(states_tensor).squeeze()

        # Compute value loss
        critic_loss = F.mse_loss(values_pred, returns_tensor)

        # Compute total loss
        total_loss = actor_loss - self.entropy_coef * entropy + self.value_coef * critic_loss

        # Compute approximate KL divergence
        approx_kl = ((old_log_probs_tensor - new_log_probs)**2).mean().item()

        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # Update critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # Return metrics
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'approx_kl': approx_kl
        }

    def _update_tensorflow(self, states: np.ndarray, actions: List, returns: np.ndarray,
                         advantages: np.ndarray, old_log_probs: np.ndarray) -> Dict[str, float]:
        """
        Update policy and value networks using TensorFlow.

        Args:
            states (np.ndarray): State inputs.
            actions (List): Action objects.
            returns (np.ndarray): Return targets.
            advantages (np.ndarray): Advantage estimates.
            old_log_probs (np.ndarray): Old log probabilities.

        Returns:
            Dict[str, float]: Training metrics.
        """
        if not HAS_TF:
            raise ImportError("TensorFlow is required for this method")

        with tf.device(self.device):
            # Convert numpy arrays to TensorFlow tensors
            states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
            returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
            advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
            old_log_probs_tensor = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)

            # Create action indices tensor (simplified)
            action_indices = tf.range(len(actions)) % self.output_dim

            metrics = {}

            # Update actor
            with tf.GradientTape() as tape:
                # Forward pass
                action_probs = self.actor_network(states_tensor)

                # Get log probabilities
                action_mask = tf.one_hot(action_indices, self.output_dim)
                selected_action_probs = tf.reduce_sum(action_probs * action_mask, axis=1)
                new_log_probs = tf.math.log(selected_action_probs + 1e-10)

                # Compute ratio for PPO
                ratio = tf.exp(new_log_probs - old_log_probs_tensor)

                # Compute surrogate objectives
                surr1 = ratio * advantages_tensor
                surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages_tensor
                actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

                # Compute entropy bonus
                entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1))

                # Compute total actor loss
                total_actor_loss = actor_loss - self.entropy_coef * entropy

            # Apply gradients
            actor_gradients = tape.gradient(total_actor_loss, self.actor_network.trainable_variables)
            actor_gradients, _ = tf.clip_by_global_norm(actor_gradients, self.max_grad_norm)
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor_network.trainable_variables))

            # Update critic
            with tf.GradientTape() as tape:
                # Forward pass
                values_pred = self.critic_network(states_tensor)

                # Compute value loss
                critic_loss = tf.reduce_mean(tf.square(returns_tensor - tf.squeeze(values_pred)))

                # Apply value coefficient
                total_critic_loss = self.value_coef * critic_loss

            # Apply gradients
            critic_gradients = tape.gradient(total_critic_loss, self.critic_network.trainable_variables)
            critic_gradients, _ = tf.clip_by_global_norm(critic_gradients, self.max_grad_norm)
            self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic_network.trainable_variables))

            # Compute approximate KL divergence
            approx_kl = tf.reduce_mean(tf.square(old_log_probs_tensor - new_log_probs))

            # Store metrics
            metrics = {
                'actor_loss': float(actor_loss),
                'critic_loss': float(critic_loss),
                'entropy': float(entropy),
                'approx_kl': float(approx_kl)
            }

        return metrics

    def train(self, num_iterations: int = 100, num_trajectories: int = 1000,
             epochs: int = 10, batch_size: int = 64,
             callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Train the PPO agent.

        Args:
            num_iterations (int, optional): Number of iterations. Defaults to 100.
            num_trajectories (int, optional): Number of trajectories to collect per iteration. Defaults to 1000.
            epochs (int, optional): Number of update epochs per iteration. Defaults to 10.
            batch_size (int, optional): Batch size. Defaults to 64.
            callback (Optional[Callable], optional): Callback function. Defaults to None.

        Returns:
            Dict[str, Any]: Training results.
        """
        # Start time
        start_time = time.time()

        # Initialize metrics storage
        training_metrics = []

        # Training loop
        for iteration in range(num_iterations):
            # Collect trajectories
            trajectories = self.collect_trajectories(num_trajectories)

            # Update policy
            metrics = self.update_policy(trajectories, epochs, batch_size)

            # Store metrics
            training_metrics.append(metrics)

            # Calculate time
            elapsed_time = time.time() - start_time
            metrics['time'] = elapsed_time

            # Call callback if provided
            if callback:
                callback(iteration, metrics, self)

        # Return results
        results = {
            'algorithm': 'PPO',
            'framework': self.framework,
            'iterations': num_iterations,
            'training_time': time.time() - start_time,
            'metrics': training_metrics
        }

        return results

    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the PPO model.

        Args:
            filepath (Optional[str], optional): Directory to save the model. Defaults to None.

        Returns:
            str: Path to the saved model.
        """
        # Create a default filename if none provided
        if filepath is None:
            filepath = f"./saved_models/ppo_model_{int(time.time())}"

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save model based on framework
        if self.framework == "pytorch":
            actor_path = f"{filepath}_actor.pt"
            critic_path = f"{filepath}_critic.pt"

            # Save networks
            torch.save(self.actor_network.state_dict(), actor_path)
            torch.save(self.critic_network.state_dict(), critic_path)
        else:
            actor_path = f"{filepath}_actor"
            critic_path = f"{filepath}_critic"

            # Save networks
            self.actor_network.save(actor_path)
            self.critic_network.save(critic_path)

        # Save metadata
        metadata = {
            'algorithm': 'PPO',
            'framework': self.framework,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'iterations': self.iterations
        }

        np.save(f"{filepath}_metadata.npy", metadata)

        return filepath

    def load(self, filepath: str) -> bool:
        """
        Load a PPO model.

        Args:
            filepath (str): Path to the saved model.

        Returns:
            bool: Whether the load was successful.
        """
        try:
            # Load metadata
            metadata = np.load(f"{filepath}_metadata.npy", allow_pickle=True).item()

            # Check if compatible
            if metadata['algorithm'] != 'PPO':
                print(f"Error: Model is not a PPO model (found {metadata['algorithm']})")
                return False

            if metadata['framework'] != self.framework:
                print(f"Error: Framework mismatch (model: {metadata['framework']}, current: {self.framework})")
                return False

            if metadata['input_dim'] != self.input_dim or metadata['output_dim'] != self.output_dim:
                print(f"Error: Dimension mismatch (model: {metadata['input_dim']}x{metadata['output_dim']}, current: {self.input_dim}x{self.output_dim})")
                return False

            # Load model based on framework
            if self.framework == "pytorch":
                actor_path = f"{filepath}_actor.pt"
                critic_path = f"{filepath}_critic.pt"

                # Load networks
                self.actor_network.load_state_dict(torch.load(actor_path))
                self.critic_network.load_state_dict(torch.load(critic_path))
            else:
                actor_path = f"{filepath}_actor"
                self.actor_path = f"{filepath}_critic"

                # Load networks
                self.actor_network = tf.keras.models.load_model(actor_path)
                self.critic_network = tf.keras.models.load_model(critic_path)

            # Update iterations
            self.iterations = metadata['iterations']

            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def get_action_probabilities(self, state: Any, player_id: int) -> np.ndarray:
        """
        Get action probabilities for a given state.

        Args:
            state: Game state.
            player_id (int): Player ID.

        Returns:
            np.ndarray: Probability distribution over actions.
        """
        # Convert state to feature vector
        state_vec = state.to_feature_vector(player_id).reshape(1, -1)

        # Get action probabilities
        action_probs, _ = self._forward_actor(state_vec)

        # Get legal actions
        legal_actions = state.get_legal_actions()
        if not legal_actions and not state.is_terminal():
            raise ValueError("No legal actions available in non-terminal state")

        # Set probabilities of illegal actions to zero and renormalize
        legal_mask = np.zeros(self.output_dim)
        for i, action in enumerate(legal_actions):
            if i < len(legal_mask):
                legal_mask[i] = 1

        masked_probs = action_probs[0] * legal_mask
        if np.sum(masked_probs) > 0:
            masked_probs = masked_probs / np.sum(masked_probs)
        else:
            # Fallback to uniform distribution over legal actions
            masked_probs = legal_mask / np.sum(legal_mask)

        return masked_probs


class ActorNetworkPyTorch(nn.Module):
    """PyTorch implementation of the actor network."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize the actor network.
        
        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
        """
        super(ActorNetworkPyTorch, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x


class CriticNetworkPyTorch(nn.Module):
    """PyTorch implementation of the critic network."""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize the critic network.
        
        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
        """
        super(CriticNetworkPyTorch, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorNetworkTensorFlow(tf.keras.Model):
    """TensorFlow implementation of the actor network."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize the actor network.
        
        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
        """
        super(ActorNetworkTensorFlow, self).__init__()
        
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc3 = tf.keras.layers.Dense(output_dim, activation='softmax')
    
    def call(self, inputs, training=False):
        """Forward pass."""
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class CriticNetworkTensorFlow(tf.keras.Model):
    """TensorFlow implementation of the critic network."""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize the critic network.
        
        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
        """
        super(CriticNetworkTensorFlow, self).__init__()
        
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc3 = tf.keras.layers.Dense(1)
    
    def call(self, inputs, training=False):
        """Forward pass."""
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x