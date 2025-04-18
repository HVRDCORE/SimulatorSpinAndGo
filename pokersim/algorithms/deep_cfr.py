"""
Алгоритм Deep Counterfactual Regret Minimization (Deep CFR) для покерного симулятора.
"""

import os
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import time
import random
from collections import deque

from pokersim.config.config_manager import get_config
from pokersim.logging.game_logger import get_logger
from pokersim.ml.model_io import get_model_io

# Условные импорты для PyTorch
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

logger = logging.getLogger("pokersim.algorithms.deep_cfr")

class DeepCFRSolver:
    """
    Решатель Deep Counterfactual Regret Minimization (Deep CFR) для покерных игр.
    """
    def __init__(self, game_state_class: Any, num_players: int = 2, framework: str = "auto"):
        config = get_config()
        self.config = config.to_dict()
        self.game_logger = get_logger()
        self.model_io = get_model_io()

        if framework == "auto":
            if TORCH_AVAILABLE:
                self.framework = "pytorch"
            elif TF_AVAILABLE:
                self.framework = "tensorflow"
            else:
                raise ImportError("Не найден поддерживаемый фреймворк глубокого обучения")
        else:
            if framework == "pytorch" and not TORCH_AVAILABLE:
                raise ImportError("PyTorch недоступен")
            elif framework == "tensorflow" and not TF_AVAILABLE:
                raise ImportError("TensorFlow недоступен")
            self.framework = framework

        self.game_state_class = game_state_class
        self.num_players = num_players
        self.advantage_networks = {}
        self.strategy_network = None
        self.advantage_buffers = {}
        self.strategy_buffer = []
        self.iterations = 0
        self.batch_size = self.config["training"]["batch_size"]
        self.learning_rate = self.config["training"]["learning_rate"]
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config["training"]["use_gpu"] else "cpu")

        initial_state = self.game_state_class(
            num_players=self.num_players,
            small_blind=1,
            big_blind=2,
            initial_stacks=[500] * self.num_players,
            button=0,
            player_ids=list(range(self.num_players))
        )

        info_set_dim = initial_state.get_info_set_dimension()
        num_actions = len(initial_state.get_legal_actions())

        for player in range(self.num_players):
            self.advantage_networks[player] = self._create_advantage_network(
                input_dim=info_set_dim,
                output_dim=num_actions
            )
            self.advantage_buffers[player] = deque(maxlen=1000000)

        self.strategy_network = self._create_strategy_network(
            input_dim=info_set_dim,
            output_dim=num_actions
        )

    def _create_advantage_network(self, input_dim: int, output_dim: int) -> Any:
        if self.framework == "pytorch":
            network = AdvantageNetworkPyTorch(
                input_dim=input_dim,
                hidden_dim=self.config["model"]["value_network"]["hidden_layers"][0],
                output_dim=output_dim
            ).to(self.device)
            optimizer = optim.Adam(network.parameters(), lr=self.learning_rate)
            return {"network": network, "optimizer": optimizer}
        elif self.framework == "tensorflow":
            network = AdvantageNetworkTensorFlow(
                input_dim=input_dim,
                hidden_dim=self.config["model"]["value_network"]["hidden_layers"][0],
                output_dim=output_dim
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            return {"network": network, "optimizer": optimizer}

    def _create_strategy_network(self, input_dim: int, output_dim: int) -> Any:
        if self.framework == "pytorch":
            network = StrategyNetworkPyTorch(
                input_dim=input_dim,
                hidden_dim=self.config["model"]["policy_network"]["hidden_layers"][0],
                output_dim=output_dim
            ).to(self.device)
            optimizer = optim.Adam(network.parameters(), lr=self.learning_rate)
            return {"network": network, "optimizer": optimizer}
        elif self.framework == "tensorflow":
            network = StrategyNetworkTensorFlow(
                input_dim=input_dim,
                hidden_dim=self.config["model"]["policy_network"]["hidden_layers"][0],
                output_dim=output_dim
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            return {"network": network, "optimizer": optimizer}

    def _forward(self, network: Any, inputs: np.ndarray) -> np.ndarray:
        if self.framework == "pytorch":
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                outputs = network["network"](inputs_tensor)
            return outputs.cpu().numpy()
        elif self.framework == "tensorflow":
            inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
            outputs = network["network"](inputs_tensor, training=False)
            return outputs.numpy()

    def _train_network(self, network: Any, inputs: np.ndarray, targets: np.ndarray,
                       weights: Optional[np.ndarray] = None) -> float:
        if self.framework == "pytorch":
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            targets_tensor = torch.tensor(targets, dtype=torch.float32).to(self.device)
            weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device) if weights is not None else torch.ones_like(targets_tensor[:, 0]).to(self.device)

            network["optimizer"].zero_grad()
            outputs = network["network"](inputs_tensor)
            losses = F.mse_loss(outputs, targets_tensor, reduction='none')
            loss = torch.mean(torch.sum(losses, dim=1) * weights_tensor)
            loss.backward()
            network["optimizer"].step()
            return loss.item()
        elif self.framework == "tensorflow":
            inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
            targets_tensor = tf.convert_to_tensor(targets, dtype=tf.float32)
            weights_tensor = tf.convert_to_tensor(weights, dtype=tf.float32) if weights is not None else tf.ones_like(targets_tensor[:, 0])

            with tf.GradientTape() as tape:
                outputs = network["network"](inputs_tensor, training=True)
                losses = tf.reduce_sum(tf.keras.losses.mean_squared_error(targets_tensor, outputs), axis=1)
                loss = tf.reduce_mean(losses * weights_tensor)
            gradients = tape.gradient(loss, network["network"].trainable_variables)
            network["optimizer"].apply_gradients(zip(gradients, network["network"].trainable_variables))
            return loss.numpy()

    def traverse_game_tree(self, state: Any, reach_probs: List[float], player: int,
                           iteration: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
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

        if current_player != player:
            action_probs = self.get_action_probabilities(state)
            action_idx = np.random.choice(len(action_probs), p=action_probs)
            action = legal_actions[action_idx]
            new_reach_probs = reach_probs.copy()
            new_reach_probs[current_player] *= action_probs[action_idx]
            next_state = state.apply_action(action)
            child_samples = self.traverse_game_tree(next_state, new_reach_probs, player, iteration)
            samples.extend(child_samples)
            return samples

        action_values = np.zeros(num_actions)
        for i, action in enumerate(legal_actions):
            next_state = state.apply_action(action)
            if next_state.is_terminal():
                action_values[i] = next_state.get_utility()
            else:
                new_reach_probs = reach_probs.copy()
                new_reach_probs[current_player] = 1.0
                child_samples = self.traverse_game_tree(next_state, new_reach_probs, player, iteration)
                samples.extend(child_samples)
                action_values[i] = next_state.get_expected_value(player)

        cf_value = np.mean(action_values)
        advantages = action_values - cf_value
        target = np.zeros(num_actions)

        for i in range(num_actions):
            target[i] = advantages[i]

        weight = np.prod(reach_probs) / reach_probs[player]
        sample = (info_set_vector, target, weight)
        samples.append(sample)

        self.advantage_buffers[player].append((info_set_vector, target, weight, iteration))

        strategy_target = np.zeros(num_actions)
        positive_advantages = np.maximum(0, advantages)
        advantage_sum = np.sum(positive_advantages)

        if advantage_sum > 0:
            strategy_target = positive_advantages / advantage_sum
        else:
            strategy_target = np.ones(num_actions) / num_actions

        self.strategy_buffer.append((info_set_vector, strategy_target, weight, iteration))
        return samples

    def train_advantage_network(self, player: int, batch_size: Optional[int] = None) -> float:
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.advantage_buffers[player]) < batch_size:
            return 0.0

        samples = random.sample(self.advantage_buffers[player], batch_size)
        info_sets = [s[0] for s in samples]
        advantages = [s[1] for s in samples]
        weights = [s[2] for s in samples]
        iterations = [s[3] for s in samples]

        info_sets = np.array(info_sets)
        advantages = np.array(advantages)
        weights = np.array(weights)
        iterations = np.array(iterations)

        iteration_weights = (iterations / self.iterations) ** 2
        combined_weights = weights * iteration_weights

        loss = self._train_network(
            network=self.advantage_networks[player],
            inputs=info_sets,
            targets=advantages,
            weights=combined_weights
        )
        return loss

    def train_strategy_network(self, batch_size: Optional[int] = None) -> float:
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.strategy_buffer) < batch_size:
            return 0.0

        samples = random.sample(self.strategy_buffer, batch_size)
        info_sets = [s[0] for s in samples]
        strategies = [s[1] for s in samples]
        weights = [s[2] for s in samples]
        iterations = [s[3] for s in samples]

        info_sets = np.array(info_sets)
        strategies = np.array(strategies)
        weights = np.array(weights)
        iterations = np.array(iterations)

        iteration_weights = iterations / self.iterations
        combined_weights = weights * iteration_weights

        loss = self._train_network(
            network=self.strategy_network,
            inputs=info_sets,
            targets=strategies,
            weights=combined_weights
        )
        return loss

    def train(self, num_iterations: int, num_traversals: int = 100,
              callback: Optional[Callable] = None) -> Dict[str, Any]:
        start_time = time.time()
        metrics = []

        for i in range(num_iterations):
            self.iterations += 1
            for player in range(self.num_players):
                for _ in range(num_traversals):
                    initial_state = self.game_state_class(
                        num_players=self.num_players,
                        small_blind=1,
                        big_blind=2,
                        initial_stacks=[500] * self.num_players,
                        button=0,
                        player_ids=list(range(self.num_players))
                    )
                    reach_probs = [1.0] * self.num_players
                    self.traverse_game_tree(initial_state, reach_probs, player, self.iterations)
                advantage_loss = self.train_advantage_network(player)
                logger.debug(f"Итерация {i+1}/{num_iterations} - Игрок {player} - Потери преимуществ: {advantage_loss:.6f}")

            strategy_loss = self.train_strategy_network()
            if (i + 1) % max(1, num_iterations // 10) == 0:
                elapsed = time.time() - start_time
                logger.info(f"Итерация {i+1}/{num_iterations} - Потери стратегии: {strategy_loss:.6f} - Время: {elapsed:.2f}с")
                metric = {
                    "iteration": i + 1,
                    "strategy_loss": float(strategy_loss),
                    "time": elapsed
                }
                metrics.append(metric)
                self.game_logger.log_training_metrics("DeepCFR", metric, i + 1)

            if callback is not None:
                callback(i, strategy_loss, self)

        total_time = time.time() - start_time
        logger.info(f"Обучение Deep CFR завершено за {total_time:.2f} секунд")
        return {
            "algorithm": "DeepCFR",
            "iterations": num_iterations,
            "total_time": total_time,
            "advantage_buffer_sizes": {p: len(self.advantage_buffers[p]) for p in range(self.num_players)},
            "strategy_buffer_size": len(self.strategy_buffer),
            "metrics": metrics
        }

    def get_action_probabilities(self, state: Any) -> np.ndarray:
        info_set_vector = state.get_info_set_vector()
        legal_actions = state.get_legal_actions()
        inputs = np.array([info_set_vector])
        outputs = self._forward(self.strategy_network, inputs)[0]
        action_mask = np.zeros(outputs.shape)
        for i, _ in enumerate(legal_actions):
            action_mask[i] = 1
        masked_outputs = outputs * action_mask
        action_sum = np.sum(masked_outputs)
        if action_sum > 0:
            return masked_outputs / action_sum
        return action_mask / np.sum(action_mask)

    def get_advantage_values(self, state: Any, player: int) -> np.ndarray:
        info_set_vector = state.get_info_set_vector()
        inputs = np.array([info_set_vector])
        outputs = self._forward(self.advantage_networks[player], inputs)[0]
        return outputs

    def save(self, filepath: Optional[str] = None) -> str:
        if filepath is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filepath = os.path.join(self.config["agents"]["save_dir"], f"deep_cfr_model_{timestamp}")
        os.makedirs(filepath, exist_ok=True)
        metadata = {
            "algorithm": "DeepCFR",
            "framework": self.framework,
            "iterations": self.iterations,
            "timestamp": time.time(),
            "num_players": self.num_players
        }
        with open(os.path.join(filepath, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        for player in range(self.num_players):
            self._save_network(
                os.path.join(filepath, f"advantage_network_{player}"),
                self.advantage_networks[player]["network"]
            )
        self._save_network(
            os.path.join(filepath, "strategy_network"),
            self.strategy_network["network"]
        )
        logger.info(f"Модель Deep CFR сохранена в {filepath}")
        return filepath

    def load(self, filepath: str) -> bool:
        try:
            with open(os.path.join(filepath, "metadata.json"), 'r') as f:
                metadata = json.load(f)
            if metadata.get("algorithm") != "DeepCFR":
                logger.error(f"Недопустимый алгоритм в {filepath}")
                return False
            if metadata.get("framework") != self.framework:
                logger.error(f"Несоответствие фреймворка: {metadata.get('framework')}, текущий {self.framework}")
                return False
            self.iterations = metadata.get("iterations", 0)
            for player in range(self.num_players):
                self._load_network(
                    os.path.join(filepath, f"advantage_network_{player}"),
                    self.advantage_networks[player]["network"]
                )
            self._load_network(
                os.path.join(filepath, "strategy_network"),
                self.strategy_network["network"]
            )
            logger.info(f"Модель Deep CFR загружена из {filepath}")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки модели Deep CFR: {e}")
            return False

    def _save_network(self, filepath: str, network: Any):
        if self.framework == "pytorch":
            torch.save(network.state_dict(), f"{filepath}.pt")
        elif self.framework == "tensorflow":
            network.save(f"{filepath}.keras")

    def _load_network(self, filepath: str, network: Any):
        if self.framework == "pytorch":
            state_dict = torch.load(f"{filepath}.pt")
            network.load_state_dict(state_dict)
        elif self.framework == "tensorflow":
            loaded_model = tf.keras.models.load_model(f"{filepath}.keras")
            network.set_weights(loaded_model.get_weights())

if TORCH_AVAILABLE:
    class AdvantageNetworkPyTorch(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    class StrategyNetworkPyTorch(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return F.softmax(x, dim=1)

if TF_AVAILABLE:
    class AdvantageNetworkTensorFlow(tf.keras.Model):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
            self.fc2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
            self.fc3 = tf.keras.layers.Dense(output_dim)

        def call(self, inputs, training=False):
            x = self.fc1(inputs)
            x = self.fc2(x)
            return self.fc3(x)

    class StrategyNetworkTensorFlow(tf.keras.Model):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
            self.fc2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
            self.fc3 = tf.keras.layers.Dense(output_dim)
            self.softmax = tf.keras.layers.Softmax()

        def call(self, inputs, training=False):
            x = self.fc1(inputs)
            x = self.fc2(x)
            x = self.fc3(x)
            return self.softmax(x)