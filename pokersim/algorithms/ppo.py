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
    HAS_TF = True
except ImportError:
    HAS_TF = False

if not HAS_TORCH and not HAS_TF:
    raise ImportError("Either PyTorch or TensorFlow is required for PPO implementation")


class PPOSolver:
    """
    Proximal Policy Optimization (PPO) solver for poker games.

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
        self.game_state_class = game_state_class
        self.num_players = num_players

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

        temp_state = self.game_state_class(
            num_players=self.num_players,
            small_blind=1,
            big_blind=2,
            initial_stacks=[500] * num_players,
            button=0,
            player_ids=list(range(num_players))
        )
        player_id = 0
        feature_vector = temp_state.to_feature_vector(player_id)
        self.input_dim = feature_vector.shape[0]
        self.output_dim = len(temp_state.get_legal_actions())

        if self.framework == "pytorch":
            self._init_pytorch_components()
        else:
            self._init_tensorflow_components()

        self.clip_ratio = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5

        self.trajectory_buffer = {
            'states': [],
            'actions': [],
            'action_indices': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }

        self.iterations = 0
        self.metrics = []

    def _init_pytorch_components(self):
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_network = ActorNetworkPyTorch(self.input_dim, 128, self.output_dim).to(self.device)
        self.critic_network = CriticNetworkPyTorch(self.input_dim, 128).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=1e-3)

    def _init_tensorflow_components(self):
        if not HAS_TF:
            raise ImportError("TensorFlow is not installed")
        self.actor_network = ActorNetworkTensorFlow(self.input_dim, 128, self.output_dim)
        self.critic_network = CriticNetworkTensorFlow(self.input_dim, 128)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        gpus = tf.config.list_physical_devices('GPU')
        self.device = "/GPU:0" if gpus else "/CPU:0"

    def _forward_actor(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.framework == "pytorch":
            states_tensor = torch.FloatTensor(states).to(self.device)
            action_probs = self.actor_network(states_tensor)
            dist = Categorical(action_probs)
            log_probs = dist.logits
            return action_probs.detach().cpu().numpy(), log_probs.detach().cpu().numpy()
        else:
            with tf.device(self.device):
                states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
                action_probs = self.actor_network(states_tensor)
                action_probs = tf.clip_by_value(action_probs, 1e-10, 1.0)
                log_probs = tf.math.log(action_probs)
                return action_probs.numpy(), log_probs.numpy()

    def _forward_critic(self, states: np.ndarray) -> np.ndarray:
        if self.framework == "pytorch":
            states_tensor = torch.FloatTensor(states).to(self.device)
            values = self.critic_network(states_tensor)
            return values.detach().cpu().numpy().flatten()
        else:
            with tf.device(self.device):
                states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
                values = self.critic_network(states_tensor)
                return values.numpy().flatten()

    def _action_to_idx(self, action: Any, legal_actions: List[Any]) -> int:
        try:
            return legal_actions.index(action)
        except ValueError:
            return 0  # Fallback to first action

    def get_action(self, state: Any, player_id: int, deterministic: bool = False) -> Tuple[Any, float, float]:
        state_vec = state.to_feature_vector(player_id).reshape(1, -1)
        action_probs, log_probs = self._forward_actor(state_vec)
        value = self._forward_critic(state_vec)[0]

        legal_actions = state.get_legal_actions()
        if len(legal_actions) > self.output_dim:
            raise ValueError(f"Number of legal actions ({len(legal_actions)}) exceeds output dimension ({self.output_dim})")

        legal_mask = np.zeros(self.output_dim)
        for i in range(min(len(legal_actions), len(legal_mask))):
            legal_mask[i] = 1

        masked_probs = action_probs[0] * legal_mask
        if np.sum(masked_probs) > 0:
            masked_probs = masked_probs / np.sum(masked_probs)
        else:
            masked_probs = legal_mask / np.sum(legal_mask)

        if deterministic:
            action_idx = np.argmax(masked_probs)
        else:
            action_idx = np.random.choice(len(masked_probs), p=masked_probs)

        action = legal_actions[action_idx]
        log_prob = log_probs[0][action_idx]
        return action, log_prob, value

    def collect_trajectories(self, num_trajectories: int = 1000, max_steps: int = 100) -> Dict[str, List]:
        trajectory_buffer = {
            'states': [],
            'actions': [],
            'action_indices': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }

        for _ in range(num_trajectories):
            state = self.game_state_class(
                num_players=self.num_players,
                small_blind=1,
                big_blind=2,
                initial_stacks=[500] * self.num_players,
                button=0,
                player_ids=list(range(self.num_players))
            )
            for step in range(max_steps):
                player_id = state.current_player
                if state.is_terminal():
                    break

                state_vec = state.to_feature_vector(player_id)
                legal_actions = state.get_legal_actions()
                action, log_prob, value = self.get_action(state, player_id)
                action_idx = self._action_to_idx(action, legal_actions)

                next_state = state.apply_action(action)
                reward = 0.0
                if next_state.is_terminal():
                    rewards = next_state.get_rewards()
                    reward = rewards[player_id]

                trajectory_buffer['states'].append(state_vec)
                trajectory_buffer['actions'].append(action)
                trajectory_buffer['action_indices'].append(action_idx)
                trajectory_buffer['rewards'].append(reward)
                trajectory_buffer['values'].append(value)
                trajectory_buffer['log_probs'].append(log_prob)
                trajectory_buffer['dones'].append(next_state.is_terminal())

                state = next_state
                if state.is_terminal():
                    break

        for key in trajectory_buffer:
            if key not in ['actions', 'action_indices']:
                trajectory_buffer[key] = np.array(trajectory_buffer[key])

        return trajectory_buffer

    def compute_advantages(self, rewards: np.ndarray, values: np.ndarray,
                           dones: np.ndarray, gamma: float = 0.99,
                           lambda_gae: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        last_value = 0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_value * mask - values[t]
            advantages[t] = delta + gamma * lambda_gae * last_advantage * mask
            last_advantage = advantages[t]

        returns = advantages + values
        return returns, advantages

    def update_policy(self, trajectories: Dict[str, Any],
                      epochs: int = 10, batch_size: int = 64) -> Dict[str, float]:
        states = np.array(trajectories['states'])
        rewards = np.array(trajectories['rewards'])
        values = np.array(trajectories['values'])
        log_probs_old = np.array(trajectories['log_probs'])
        dones = np.array(trajectories['dones'])
        action_indices = np.array(trajectories['action_indices'])

        returns, advantages = self.compute_advantages(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0
        }

        indices = np.arange(len(states))
        for epoch in range(epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, len(indices), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_states = states[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_action_indices = action_indices[batch_indices]

                if self.framework == "pytorch":
                    update_metrics = self._update_pytorch(
                        batch_states, batch_action_indices, batch_returns,
                        batch_advantages, batch_log_probs_old
                    )
                else:
                    update_metrics = self._update_tensorflow(
                        batch_states, batch_action_indices, batch_returns,
                        batch_advantages, batch_log_probs_old
                    )

                for key, value in update_metrics.items():
                    metrics[key] += value / (len(indices) // batch_size + 1) / epochs

        self.iterations += 1
        metrics['iteration'] = self.iterations
        self.metrics.append(metrics)
        return metrics

    def _update_pytorch(self, states: np.ndarray, action_indices: np.ndarray,
                        returns: np.ndarray, advantages: np.ndarray,
                        old_log_probs: np.ndarray) -> Dict[str, float]:
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for this method")

        states_tensor = torch.FloatTensor(states).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        action_indices_tensor = torch.LongTensor(action_indices).to(self.device)

        action_probs = self.actor_network(states_tensor)
        dist = Categorical(action_probs)
        new_log_probs = dist.log_prob(action_indices_tensor)

        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages_tensor
        actor_loss = -torch.min(surr1, surr2).mean()

        entropy = dist.entropy().mean()
        values_pred = self.critic_network(states_tensor).squeeze()
        critic_loss = F.mse_loss(values_pred, returns_tensor)

        total_loss = actor_loss - self.entropy_coef * entropy + self.value_coef * critic_loss
        approx_kl = ((old_log_probs_tensor - new_log_probs)**2).mean().item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'approx_kl': approx_kl
        }

    def _update_tensorflow(self, states: np.ndarray, action_indices: np.ndarray,
                           returns: np.ndarray, advantages: np.ndarray,
                           old_log_probs: np.ndarray) -> Dict[str, float]:
        if not HAS_TF:
            raise ImportError("TensorFlow is required for this method")

        with tf.device(self.device):
            states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
            returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
            advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
            old_log_probs_tensor = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
            action_indices_tensor = tf.convert_to_tensor(action_indices, dtype=tf.int32)

            with tf.GradientTape() as tape:
                action_probs = self.actor_network(states_tensor)
                action_mask = tf.one_hot(action_indices_tensor, self.output_dim)
                selected_action_probs = tf.reduce_sum(action_probs * action_mask, axis=1)
                new_log_probs = tf.math.log(tf.clip_by_value(selected_action_probs, 1e-10, 1.0))

                ratio = tf.exp(new_log_probs - old_log_probs_tensor)
                surr1 = ratio * advantages_tensor
                surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages_tensor
                actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

                entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(tf.clip_by_value(action_probs, 1e-10, 1.0)), axis=1))
                total_actor_loss = actor_loss - self.entropy_coef * entropy

            actor_gradients = tape.gradient(total_actor_loss, self.actor_network.trainable_variables)
            actor_gradients, _ = tf.clip_by_global_norm(actor_gradients, self.max_grad_norm)
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor_network.trainable_variables))

            with tf.GradientTape() as tape:
                values_pred = self.critic_network(states_tensor)
                critic_loss = tf.reduce_mean(tf.square(returns_tensor - tf.squeeze(values_pred)))
                total_critic_loss = self.value_coef * critic_loss

            critic_gradients = tape.gradient(total_critic_loss, self.critic_network.trainable_variables)
            critic_gradients, _ = tf.clip_by_global_norm(critic_gradients, self.max_grad_norm)
            self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic_network.trainable_variables))

            approx_kl = tf.reduce_mean(tf.square(old_log_probs_tensor - new_log_probs))

            return {
                'actor_loss': float(actor_loss),
                'critic_loss': float(critic_loss),
                'entropy': float(entropy),
                'approx_kl': float(approx_kl)
            }

    def train(self, num_iterations: int = 100, num_trajectories: int = 1000,
              epochs: int = 10, batch_size: int = 64,
              callback: Optional[Callable] = None) -> Dict[str, Any]:
        start_time = time.time()
        training_metrics = []

        for iteration in range(num_iterations):
            trajectories = self.collect_trajectories(num_trajectories)
            metrics = self.update_policy(trajectories, epochs, batch_size)
            training_metrics.append(metrics)
            elapsed_time = time.time() - start_time
            metrics['time'] = elapsed_time
            if callback:
                callback(iteration, metrics, self)

        return {
            'algorithm': 'PPO',
            'framework': self.framework,
            'iterations': num_iterations,
            'training_time': time.time() - start_time,
            'metrics': training_metrics
        }

    def save(self, filepath: Optional[str] = None) -> str:
        if filepath is None:
            filepath = f"./saved_models/ppo_model_{int(time.time())}"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if self.framework == "pytorch":
            actor_path = f"{filepath}_actor.pt"
            critic_path = f"{filepath}_critic.pt"
            torch.save(self.actor_network.state_dict(), actor_path)
            torch.save(self.critic_network.state_dict(), critic_path)
        else:
            actor_path = f"{filepath}_actor"
            critic_path = f"{filepath}_critic"
            self.actor_network.save(actor_path)
            self.critic_network.save(critic_path)

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
        try:
            metadata = np.load(f"{filepath}_metadata.npy", allow_pickle=True).item()
            if metadata['algorithm'] != 'PPO' or metadata['framework'] != self.framework or \
               metadata['input_dim'] != self.input_dim or metadata['output_dim'] != self.output_dim:
                print("Error: Incompatible model metadata")
                return False

            if self.framework == "pytorch":
                actor_path = f"{filepath}_actor.pt"
                critic_path = f"{filepath}_critic.pt"
                self.actor_network.load_state_dict(torch.load(actor_path))
                self.critic_network.load_state_dict(torch.load(critic_path))
            else:
                actor_path = f"{filepath}_actor"
                critic_path = f"{filepath}_critic"
                self.actor_network = tf.keras.models.load_model(actor_path)
                self.critic_network = tf.keras.models.load_model(critic_path)

            self.iterations = metadata['iterations']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def get_action_probabilities(self, state: Any, player_id: int) -> np.ndarray:
        state_vec = state.to_feature_vector(player_id).reshape(1, -1)
        action_probs, _ = self._forward_actor(state_vec)
        legal_actions = state.get_legal_actions()
        if not legal_actions and not state.is_terminal():
            raise ValueError("No legal actions available in non-terminal state")

        legal_mask = np.zeros(self.output_dim)
        for i in range(min(len(legal_actions), len(legal_mask))):
            legal_mask[i] = 1

        masked_probs = action_probs[0] * legal_mask
        if np.sum(masked_probs) > 0:
            masked_probs = masked_probs / np.sum(masked_probs)
        else:
            masked_probs = legal_mask / np.sum(legal_mask)
        return masked_probs


class ActorNetworkPyTorch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x


class CriticNetworkPyTorch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorNetworkTensorFlow(tf.keras.Model):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc3 = tf.keras.layers.Dense(output_dim, activation='softmax')

    def call(self, inputs, training=False):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class CriticNetworkTensorFlow(tf.keras.Model):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc3 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x