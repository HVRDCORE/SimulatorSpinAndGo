"""
Implementation of advanced machine learning-based agents for poker simulations.

This module provides sophisticated agent implementations that leverage neural networks
and reinforcement learning techniques to make poker decisions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import random

from pokersim.agents.base_agent import Agent
from pokersim.ml.agents import MLAgent
from pokersim.game.state import GameState, Action, ActionType, Stage
from pokersim.game.evaluator import HandEvaluator
from pokersim.ml.models import PokerMLP, PokerCNN, PokerActorCritic
from pokersim.algorithms.deep_cfr import DeepCFRSolver
from pokersim.algorithms.ppo import PPOSolver


class DeepCFRAgent(Agent):
    """
    An agent that uses Deep Counterfactual Regret Minimization for decision making.

    This agent uses a neural network trained with Deep CFR to approximate optimal
    strategy in imperfect information games like poker.

    Attributes:
        player_id (int): The ID of the player controlled by this agent.
        deep_cfr (DeepCFRSolver): The Deep CFR algorithm.
        device (torch.device): Device for computation (CPU/GPU).
        epsilon (float): Exploration rate for epsilon-greedy strategy.
    """

    def __init__(self, player_id: int, game_state_class: Any, num_players: int = 2,
                 device: Optional[torch.device] = None, epsilon: float = 0.05, framework: str = "auto"):
        super().__init__(player_id)
        self.deep_cfr = DeepCFRSolver(game_state_class=game_state_class, framework=framework)
        self.device = device if device is not None else torch.device("cpu")
        self.epsilon = epsilon

    def act(self, game_state: GameState) -> Action:
        legal_actions = game_state.get_legal_actions()
        if not legal_actions:
            raise ValueError("No legal actions available")

        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        action_probs = self.deep_cfr.get_action_probabilities(game_state)
        action_idx = np.random.choice(len(legal_actions), p=action_probs)
        return legal_actions[action_idx]

    def observe(self, game_state: GameState) -> None:
        pass

    def train(self, game_state: GameState, num_iterations: int = 100, num_traversals: int = 100) -> Dict[str, float]:
        metrics = self.deep_cfr.train(num_iterations=num_iterations, num_traversals=num_traversals)
        return metrics

    def save(self, path: str) -> None:
        self.deep_cfr.save(path)

    def load(self, path: str) -> None:
        self.deep_cfr.load(path)


class PPOAgent(Agent):
    """
    Агент, использующий Proximal Policy Optimization для принятия решений.

    Этот агент использует нейронную сеть, обученную PPO, для выбора действий
    в покерных играх формата Spin and Go, балансируя между исследованием и эксплуатацией.

    Атрибуты:
        player_id (int): ID игрока, управляемого агентом.
        ppo (PPOSolver): Экземпляр алгоритма PPO.
        device (torch.device): Устройство для вычислений (CPU/GPU).
        epsilon (float): Уровень исследования для стратегии epsilon-greedy.
    """

    def __init__(self, player_id: int, game_state_class: Any, num_players: int = 3,
                 device: Optional[torch.device] = None, epsilon: float = 0.1,
                 framework: str = "auto"):
        """
        Инициализация агента PPO.

        Аргументы:
            player_id (int): ID игрока, управляемого агентом.
            game_state_class: Класс для создания состояний игры.
            num_players (int, optional): Количество игроков. По умолчанию 3.
            device (torch.device, optional): Устройство для вычислений. По умолчанию None.
            epsilon (float, optional): Уровень исследования. По умолчанию 0.1.
            framework (str, optional): Фреймворк ML ('pytorch' или 'tensorflow'). По умолчанию "auto".
        """
        super().__init__(player_id)
        self.ppo = PPOSolver(game_state_class=game_state_class, num_players=num_players, framework=framework)
        self.device = device if device is not None else torch.device("cpu")
        self.epsilon = epsilon

    def act(self, game_state: GameState) -> Action:
        """
        Выбор действия для текущего состояния игры.

        Аргументы:
            game_state (GameState): Текущее состояние игры.

        Возвращает:
            Action: Выбранное действие.
        """
        legal_actions = game_state.get_legal_actions()
        if not legal_actions:
            return None

        if random.random() < self.epsilon:
            action = random.choice(legal_actions)
            print(f"PPOAgent {self.player_id}: Случайное действие {action}")
            return action

        action, _, _ = self.ppo.get_action(game_state, self.player_id)
        if action not in legal_actions:
            print(f"PPOAgent {self.player_id}: Нелегальное действие {action}, выбор случайного")
            action = random.choice(legal_actions)
        print(f"PPOAgent {self.player_id}: Действие модели {action}")
        return action

    def observe(self, game_state: GameState) -> None:
        """
        Наблюдение состояния игры (без операций для PPO).

        Аргументы:
            game_state (GameState): Текущее состояние игры.
        """
        pass

    def train(self, num_iterations: int = 100, num_trajectories: int = 1000) -> Dict[str, float]:
        """
        Обучение агента PPO.

        Аргументы:
            num_iterations (int, optional): Количество итераций. По умолчанию 100.
            num_trajectories (int, optional): Количество траекторий на итерацию. По умолчанию 1000.

        Возвращает:
            Dict[str, float]: Метрики обучения.
        """
        metrics = self.ppo.train(num_iterations=num_iterations, num_trajectories=num_trajectories)
        return metrics

    def save(self, path: str) -> None:
        """
        Сохранение модели PPO.

        Аргументы:
            path (str): Путь для сохранения модели.
        """
        self.ppo.save(path)

    def load(self, path: str) -> None:
        """
        Загрузка модели PPO.

        Аргументы:
            path (str): Путь к сохраненной модели.
        """
        self.ppo.load(path)


class ImitationLearningAgent(Agent):
    """
    An agent that uses Imitation Learning to mimic expert strategies.

    This agent learns to imitate the behavior of expert agents by observing their actions.

    Attributes:
        player_id (int): The ID of the player controlled by this agent.
        model (nn.Module): The neural network model.
        optimizer (optim.Optimizer): The optimizer.
        device (torch.device): Device for computation (CPU/GPU).
        expert (Agent): The expert agent to imitate.
        memory (List): Memory of state-action pairs.
        batch_size (int): Batch size for training.
    """

    def __init__(self, player_id: int, input_dim: int, hidden_dims: List[int], action_dim: int,
                lr: float = 0.001, device: Optional[torch.device] = None, expert: Optional[Agent] = None,
                batch_size: int = 32):
        """
        Initialize an Imitation Learning agent.

        Args:
            player_id (int): The ID of the player controlled by this agent.
            input_dim (int): Input dimension for the neural network.
            hidden_dims (List[int]): Hidden dimensions for the neural network.
            action_dim (int): Output dimension for the neural network.
            lr (float, optional): Learning rate. Defaults to 0.001.
            device (torch.device, optional): Device for computation (CPU/GPU). Defaults to CPU.
            expert (Agent, optional): The expert agent to imitate. Defaults to None.
            batch_size (int, optional): Batch size for training. Defaults to 32.
        """
        super().__init__(player_id)
        self.device = device if device is not None else torch.device("cpu")
        self.model = PokerMLP(input_dim, hidden_dims, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.expert = expert
        self.memory = []
        self.batch_size = batch_size

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

        # If we have an expert, collect demonstration
        if self.expert and game_state.current_player == self.player_id:
            expert_action = self.expert.act(game_state)
            self._add_to_memory(game_state, expert_action, legal_actions)

        # Convert state to tensor
        state_tensor = self._state_to_tensor(game_state)

        # Forward pass
        with torch.no_grad():
            action_logits = self.model(state_tensor).squeeze(0)

            # Filter illegal actions
            legal_mask = torch.zeros(action_logits.size(0), dtype=torch.bool)
            for i in range(min(len(legal_actions), action_logits.size(0))):
                legal_mask[i] = True

            masked_logits = action_logits.clone()
            masked_logits[~legal_mask] = float('-inf')

            # Sample action
            probs = F.softmax(masked_logits, dim=0)
            action_idx = torch.multinomial(probs, 1).item()

            return legal_actions[action_idx]

    def _state_to_tensor(self, game_state: GameState) -> torch.Tensor:
        """
        Convert a game state to a tensor.

        Args:
            game_state (GameState): The game state.

        Returns:
            torch.Tensor: The tensor representation.
        """
        # Get feature vector
        features = game_state.to_feature_vector(self.player_id)

        # Convert to tensor
        tensor = torch.tensor(features, dtype=torch.float32).to(self.device).unsqueeze(0)

        return tensor

    def _add_to_memory(self, game_state: GameState, action: Action, legal_actions: List[Action]) -> None:
        """
        Add a state-action pair to memory.

        Args:
            game_state (GameState): The game state.
            action (Action): The action taken by the expert.
            legal_actions (List[Action]): The legal actions.
        """
        # Get feature vector
        features = game_state.to_feature_vector(self.player_id)

        # Get action index
        action_idx = legal_actions.index(action)

        # Add to memory
        self.memory.append((features, action_idx, len(legal_actions)))

        # Train if we have enough samples
        if len(self.memory) >= self.batch_size:
            self._train_step()

    def _train_step(self) -> None:
        """Train the model on a batch of data."""
        # Skip if we don't have enough samples
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch
        batch_indices = random.sample(range(len(self.memory)), self.batch_size)
        batch = [self.memory[i] for i in batch_indices]

        # Prepare data
        states = torch.tensor([x[0] for x in batch], dtype=torch.float32).to(self.device)
        actions = torch.tensor([x[1] for x in batch], dtype=torch.long).to(self.device)
        action_dims = [x[2] for x in batch]

        # Forward pass
        logits = self.model(states)

        # Compute loss for each sample using appropriate action dimension
        loss = 0
        for i, (logit, action, dim) in enumerate(zip(logits, actions, action_dims)):
            loss += F.cross_entropy(logit[:dim].unsqueeze(0), action.unsqueeze(0))
        loss /= self.batch_size

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear part of the memory to avoid using the same samples repeatedly
        self.memory = self.memory[self.batch_size:]

    def save(self, path: str) -> None:
        """
        Save the agent's model to a file.

        Args:
            path (str): The file path.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path: str) -> None:
        """
        Load the agent's model from a file.

        Args:
            path (str): The file path.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class HeuristicNetwork(nn.Module):
    """
    A neural network that combines heuristic knowledge with learned representations.

    This network has explicit features for poker knowledge like hand strength,
    pot odds, etc. in addition to learned features.

    Attributes:
        input_dim (int): The input dimension.
        hidden_dims (List[int]): The hidden layer dimensions.
        output_dim (int): The output dimension.
        heuristic_dim (int): The dimension of heuristic features.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, heuristic_dim: int = 5):
        """
        Initialize a heuristic network.

        Args:
            input_dim (int): The input dimension.
            hidden_dims (List[int]): The hidden layer dimensions.
            output_dim (int): The output dimension.
            heuristic_dim (int, optional): The dimension of heuristic features. Defaults to 5.
        """
        super(HeuristicNetwork, self).__init__()

        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )

        self.heuristic_layers = nn.Sequential(
            nn.Linear(heuristic_dim, hidden_dims[1] // 2),
            nn.ReLU()
        )

        self.combination_layer = nn.Sequential(
            nn.Linear(hidden_dims[1] + hidden_dims[1] // 2, hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], output_dim)
        )

    def forward(self, x: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): The input tensor.
            heuristic (torch.Tensor): The heuristic features.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Process input features
        features = self.feature_layers(x)

        # Process heuristic features
        heuristic_features = self.heuristic_layers(heuristic)

        # Combine features
        combined = torch.cat([features, heuristic_features], dim=1)

        # Final processing
        output = self.combination_layer(combined)

        return output


class HybridAgent(Agent):
    """
    An agent that combines heuristics with neural network learning.

    This agent uses both domain knowledge (heuristics) and learned patterns
    to make decisions.

    Attributes:
        player_id (int): The ID of the player controlled by this agent.
        model (HeuristicNetwork): The neural network model.
        optimizer (optim.Optimizer): The optimizer.
        device (torch.device): Device for computation (CPU/GPU).
        memory (List): Memory of experiences.
        batch_size (int): Batch size for training.
        gamma (float): Discount factor.
    """

    def __init__(self, player_id: int, input_dim: int, hidden_dims: List[int], action_dim: int,
                lr: float = 0.001, device: Optional[torch.device] = None,
                batch_size: int = 32, gamma: float = 0.99):
        """
        Initialize a hybrid agent.

        Args:
            player_id (int): The ID of the player controlled by this agent.
            input_dim (int): Input dimension for the neural network.
            hidden_dims (List[int]): Hidden dimensions for the neural network.
            action_dim (int): Output dimension for the neural network.
            lr (float, optional): Learning rate. Defaults to 0.001.
            device (torch.device, optional): Device for computation (CPU/GPU). Defaults to CPU.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            gamma (float, optional): Discount factor. Defaults to 0.99.
        """
        super().__init__(player_id)
        self.device = device if device is not None else torch.device("cpu")
        self.model = HeuristicNetwork(input_dim, hidden_dims, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.memory = []
        self.batch_size = batch_size
        self.gamma = gamma

        # For current episode
        self.current_state = None
        self.current_heuristic = None
        self.current_action = None
        self.current_reward = 0.0

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

        # Calculate heuristic features
        heuristic_features = self._calculate_heuristic_features(game_state)

        # Convert state to tensor
        state_tensor = self._state_to_tensor(game_state)
        heuristic_tensor = torch.tensor(heuristic_features, dtype=torch.float32).to(self.device).unsqueeze(0)

        # Store current state and heuristic
        self.current_state = game_state.to_feature_vector(self.player_id)
        self.current_heuristic = heuristic_features

        # Forward pass
        with torch.no_grad():
            action_logits = self.model(state_tensor, heuristic_tensor).squeeze(0)

            # Filter illegal actions
            legal_mask = torch.zeros(action_logits.size(0), dtype=torch.bool)
            for i in range(min(len(legal_actions), action_logits.size(0))):
                legal_mask[i] = True

            masked_logits = action_logits.clone()
            masked_logits[~legal_mask] = float('-inf')

            # Sample action
            probs = F.softmax(masked_logits, dim=0)
            action_idx = torch.multinomial(probs, 1).item()

            self.current_action = action_idx
            return legal_actions[action_idx]

    def _state_to_tensor(self, game_state: GameState) -> torch.Tensor:
        """
        Convert a game state to a tensor.

        Args:
            game_state (GameState): The game state.

        Returns:
            torch.Tensor: The tensor representation.
        """
        # Get feature vector
        features = game_state.to_feature_vector(self.player_id)

        # Convert to tensor
        tensor = torch.tensor(features, dtype=torch.float32).to(self.device).unsqueeze(0)

        return tensor

    def _calculate_heuristic_features(self, game_state: GameState) -> List[float]:
        """
        Calculate heuristic features for the game state.

        Args:
            game_state (GameState): The game state.

        Returns:
            List[float]: The heuristic features.
        """
        features = []

        # Feature 1: Hand strength
        hole_cards = game_state.hole_cards[self.player_id] if self.player_id < len(game_state.hole_cards) else []
        community_cards = game_state.community_cards

        if game_state.stage.value >= 2:  # Flop, Turn, River
            hand = hole_cards + community_cards
            hand_rank, _ = HandEvaluator.evaluate_hand(hand)
            hand_strength = min(1.0, hand_rank / 8.0)
        else:  # Preflop
            if len(hole_cards) == 2:
                # Estimate preflop hand strength
                rank1, rank2 = hole_cards[0].rank.value, hole_cards[1].rank.value
                suited = hole_cards[0].suit == hole_cards[1].suit

                # Pocket pairs are strong
                if rank1 == rank2:
                    hand_strength = 0.5 + (rank1 - 2) / 24.0
                else:
                    # Higher ranks and suited cards are better
                    rank_strength = (rank1 + rank2 - 4) / 26.0
                    suited_bonus = 0.1 if suited else 0.0
                    hand_strength = min(1.0, rank_strength + suited_bonus)
            else:
                hand_strength = 0.0

        features.append(hand_strength)

        # Feature 2: Pot odds
        pot_size = game_state.pot + sum(game_state.current_bets)
        amount_to_call = 0
        if max(game_state.current_bets) > game_state.current_bets[self.player_id]:
            amount_to_call = max(game_state.current_bets) - game_state.current_bets[self.player_id]

        pot_odds = 0.0
        if pot_size > 0 and amount_to_call > 0:
            pot_odds = amount_to_call / (pot_size + amount_to_call)

        features.append(pot_odds)

        # Feature 3: Relative position
        position = (self.player_id - game_state.button) % game_state.num_players
        position_value = position / (game_state.num_players - 1)

        features.append(position_value)

        # Feature 4: Stack to pot ratio
        stack = game_state.stacks[self.player_id] if self.player_id < len(game_state.stacks) else 0
        spr = 0.0
        if pot_size > 0 and stack > 0:
            spr = min(1.0, stack / pot_size)

        features.append(spr)

        # Feature 5: Number of active players
        active_players = sum(game_state.active) / game_state.num_players

        features.append(active_players)

        return features

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
        self.current_reward += reward

        # If terminal state, add to memory and train
        if game_state.is_terminal():
            self.memory.append((
                self.current_state,
                self.current_heuristic,
                self.current_action,
                self.current_reward,
                None,  # Next state
                None,  # Next heuristic
                True   # Done
            ))

            self._train_step()

            # Reset current episode
            self.current_state = None
            self.current_heuristic = None
            self.current_action = None
            self.current_reward = 0.0
        else:
            # Get next state and heuristic
            next_state = game_state.to_feature_vector(self.player_id)
            next_heuristic = self._calculate_heuristic_features(game_state)

            # Add to memory
            self.memory.append((
                self.current_state,
                self.current_heuristic,
                self.current_action,
                reward,
                next_state,
                next_heuristic,
                False
            ))

            # Update current state and heuristic
            self.current_state = next_state
            self.current_heuristic = next_heuristic
            self.current_action = None

    def _train_step(self) -> None:
        """Train the model on a batch of data."""
        # Skip if we don't have enough samples
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch
        batch_indices = random.sample(range(len(self.memory)), self.batch_size)
        batch = [self.memory[i] for i in batch_indices]

        # Prepare data
        states = torch.tensor([x[0] for x in batch], dtype=torch.float32).to(self.device)
        heuristics = torch.tensor([x[1] for x in batch], dtype=torch.float32).to(self.device)
        actions = torch.tensor([x[2] for x in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([x[3] for x in batch], dtype=torch.float32).to(self.device)

        # Get non-terminal next states
        non_terminal_mask = torch.tensor([not x[6] for x in batch], dtype=torch.bool)
        non_terminal_next_states = [x[4] for i, x in enumerate(batch) if not batch[i][6]]
        non_terminal_next_heuristics = [x[5] for i, x in enumerate(batch) if not batch[i][6]]

        # Compute Q-values for current states
        q_values = self.model(states, heuristics)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        next_q_values = torch.zeros_like(rewards)

        if len(non_terminal_next_states) > 0:
            next_states_tensor = torch.tensor(non_terminal_next_states, dtype=torch.float32).to(self.device)
            next_heuristics_tensor = torch.tensor(non_terminal_next_heuristics, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                next_q = self.model(next_states_tensor, next_heuristics_tensor)
                next_q_max = next_q.max(1)[0]

            next_q_values[non_terminal_mask] = next_q_max

        expected_q_values = rewards + self.gamma * next_q_values

        # Compute loss
        loss = F.smooth_l1_loss(q_values, expected_q_values)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Limit memory size to avoid excessive memory usage
        if len(self.memory) > 10000:
            self.memory = self.memory[-10000:]

    def save(self, path: str) -> None:
        """
        Save the agent's model to a file.

        Args:
            path (str): The file path.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path: str) -> None:
        """
        Load the agent's model from a file.

        Args:
            path (str): The file path.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])