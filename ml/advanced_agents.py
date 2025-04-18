"""
Реализация продвинутых агентов на основе машинного обучения для покерных симуляций.

Этот модуль предоставляет сложные реализации агентов, использующих нейронные сети
и методы обучения с подкреплением для принятия решений в покере.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional
import random
import logging

from pokersim.agents.base_agent import Agent
from pokersim.ml.agents import MLAgent
from pokersim.game.state import GameState, Action, ActionType, Stage
from pokersim.game.evaluator import HandEvaluator
from pokersim.ml.models import PokerMLP, PokerCNN, PokerActorCritic
from pokersim.algorithms.deep_cfr import DeepCFRSolver
from pokersim.algorithms.ppo import PPOSolver

# Настройка логгера
logger = logging.getLogger(__name__)

class DeepCFRAgent(Agent):
    """
    Агент, использующий Deep Counterfactual Regret Minimization для принятия решений.

    Этот агент использует нейронную сеть, обученную с помощью Deep CFR, для аппроксимации
    оптимальной стратегии в играх с неполной информацией, таких как покер.

    Атрибуты:
        player_id (int): Идентификатор игрока, управляемого агентом.
        deep_cfr (DeepCFRSolver): Алгоритм Deep CFR.
        device (torch.device): Устройство для вычислений (CPU/GPU).
        epsilon (float): Скорость исследования для стратегии epsilon-greedy.
    """

    def __init__(self, player_id: int, game_state_class: Any, num_players: int = 2,
                 device: Optional[torch.device] = None, epsilon: float = 0.05, framework: str = "auto"):
        super().__init__(player_id)
        self.deep_cfr = DeepCFRSolver(game_state_class=game_state_class, num_players=num_players, framework=framework)
        self.device = device if device is not None else torch.device("cpu")
        self.epsilon = epsilon

    def act(self, game_state: GameState) -> Action:
        legal_actions = game_state.get_legal_actions()
        if not legal_actions:
            logger.error(f"PPOAgent {self.player_id}: Нет доступных действий")
            raise ValueError("Нет доступных действий")

        # Уменьшаем epsilon для более детерминированных действий
        if random.random() < self.epsilon * 0.5:  # Уменьшаем вероятность случайного действия
            action = random.choice(legal_actions)
            logger.debug(f"PPOAgent {self.player_id}: Выбрано случайное действие {action}")
            return action

        action, _, _ = self.ppo.get_action(game_state, self.player_id)
        if action not in legal_actions:
            logger.error(f"PPOAgent {self.player_id}: Модель вернула недопустимое действие {action}, выбор случайного")
            action = random.choice(legal_actions)
        logger.debug(f"PPOAgent {self.player_id}: Выбрано модельное действие {action}")
        return action

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

    Этот агент использует нейронную сеть, обученную с помощью PPO, для выбора действий
    в играх Spin and Go, балансируя между исследованием и использованием.

    Атрибуты:
        player_id (int): Идентификатор игрока, управляемого агентом.
        ppo (PPOSolver): Экземпляр алгоритма PPO.
        device (torch.device): Устройство для вычислений (CPU/GPU).
        epsilon (float): Скорость исследования для стратегии epsilon-greedy.
    """

    def __init__(self, player_id: int, game_state_class: Any, num_players: int = 3,
                 device: Optional[torch.device] = None, epsilon: float = 0.1,
                 framework: str = "auto"):
        """
        Инициализация агента PPO.

        Аргументы:
            player_id (int): Идентификатор игрока, управляемого агентом.
            game_state_class: Класс для создания игровых состояний.
            num_players (int, optional): Количество игроков. По умолчанию 3.
            device (torch.device, optional): Устройство для вычислений. По умолчанию None.
            epsilon (float, optional): Скорость исследования. По умолчанию 0.1.
            framework (str, optional): Фреймворк ML ('pytorch' или 'tensorflow'). По умолчанию "auto".
        """
        super().__init__(player_id)
        self.ppo = PPOSolver(game_state_class=game_state_class, num_players=num_players, framework=framework)
        self.device = device if device is not None else torch.device("cpu")
        self.epsilon = epsilon

    def act(self, game_state: GameState) -> Action:
        """
        Выбор действия для текущего игрового состояния.

        Аргументы:
            game_state (GameState): Текущее игровое состояние.

        Возвращает:
            Action: Выбранное действие.
        """
        legal_actions = game_state.get_legal_actions()
        if not legal_actions:
            logger.error(f"PPOAgent {self.player_id}: Нет доступных действий")
            raise ValueError("Нет доступных действий")

        if random.random() < self.epsilon:
            action = random.choice(legal_actions)
            logger.debug(f"PPOAgent {self.player_id}: Выбрано случайное действие {action}")
            return action

        action, _, _ = self.ppo.get_action(game_state, self.player_id)
        if action not in legal_actions:
            logger.error(f"PPOAgent {self.player_id}: Модель вернула недопустимое действие {action}, выбор случайного")
            action = random.choice(legal_actions)
        logger.debug(f"PPOAgent {self.player_id}: Выбрано модельное действие {action}")
        return action

    def observe(self, game_state: GameState) -> None:
        """
        Наблюдение за игровым состоянием (без действий для PPO).

        Аргументы:
            game_state (GameState): Текущее игровое состояние.
        """
        pass

    def train(self, num_iterations: int = 100, num_trajectories: int = 1000) -> Dict[str, float]:
        """
        Обучение агента PPO.

        Аргументы:
            num_iterations (int, optional): Количество итераций. По умолчанию 100.
            num_trajectories (int, optional): Количество траекторий за итерацию. По умолчанию 1000.

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
            path (str): Путь к сохранённой модели.
        """
        self.ppo.load(path)


class ImitationLearningAgent(Agent):
    """
    Агент, использующий имитационное обучение для подражания экспертным стратегиям.

    Этот агент учится имитировать поведение экспертных агентов, наблюдая за их действиями.

    Атрибуты:
        player_id (int): Идентификатор игрока, управляемого агентом.
        model (nn.Module): Модель нейронной сети.
        optimizer (optim.Optimizer): Оптимизатор.
        device (torch.device): Устройство для вычислений (CPU/GPU).
        expert (Agent): Экспертный агент для подражания.
        memory (List): Память пар состояний и действий.
        batch_size (int): Размер пакета для обучения.
    """

    def __init__(self, player_id: int, input_dim: int, hidden_dims: List[int], action_dim: int,
                 lr: float = 0.001, device: Optional[torch.device] = None, expert: Optional[Agent] = None,
                 batch_size: int = 32):
        super().__init__(player_id)
        self.device = device if device is not None else torch.device("cpu")
        self.model = PokerMLP(input_dim, hidden_dims, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.expert = expert
        self.memory = []
        self.batch_size = batch_size

    def act(self, game_state: GameState) -> Action:
        legal_actions = game_state.get_legal_actions()
        if not legal_actions:
            logger.error(f"ImitationLearningAgent {self.player_id}: Нет доступных действий")
            raise ValueError("Нет доступных действий")

        if self.expert and game_state.current_player == self.player_id:
            expert_action = self.expert.act(game_state)
            self._add_to_memory(game_state, expert_action, legal_actions)

        state_tensor = self._state_to_tensor(game_state)

        with torch.no_grad():
            action_logits = self.model(state_tensor).squeeze(0)
            legal_mask = torch.zeros(action_logits.size(0), dtype=torch.bool)
            for i in range(min(len(legal_actions), action_logits.size(0))):
                legal_mask[i] = True

            masked_logits = action_logits.clone()
            masked_logits[~legal_mask] = float('-inf')
            probs = F.softmax(masked_logits, dim=0)
            action_idx = torch.multinomial(probs, 1).item()

            logger.debug(f"ImitationLearningAgent {self.player_id}: Выбрано действие {legal_actions[action_idx]}")
            return legal_actions[action_idx]

    def _state_to_tensor(self, game_state: GameState) -> torch.Tensor:
        features = game_state.to_feature_vector(self.player_id)
        tensor = torch.tensor(features, dtype=torch.float32).to(self.device).unsqueeze(0)
        return tensor

    def _add_to_memory(self, game_state: GameState, action: Action, legal_actions: List[Action]) -> None:
        features = game_state.to_feature_vector(self.player_id)
        action_idx = legal_actions.index(action)
        self.memory.append((features, action_idx, len(legal_actions)))

    def _train_step(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        batch_indices = random.sample(range(len(self.memory)), self.batch_size)
        batch = [self.memory[i] for i in batch_indices]

        states = torch.tensor([x[0] for x in batch], dtype=torch.float32).to(self.device)
        actions = torch.tensor([x[1] for x in batch], dtype=torch.long).to(self.device)
        action_dims = [x[2] for x in batch]

        logits = self.model(states)
        loss = 0
        for i, (logit, action, dim) in enumerate(zip(logits, actions, action_dims)):
            loss += F.cross_entropy(logit[:dim].unsqueeze(0), action.unsqueeze(0))
        loss /= self.batch_size

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory = self.memory[self.batch_size:]

    def train_episode(self) -> None:
        """Обучение модели после сбора демонстраций за эпизод."""
        if len(self.memory) >= self.batch_size:
            self._train_step()

    def save(self, path: str) -> None:
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class HeuristicNetwork(nn.Module):
    """
    Нейронная сеть, сочетающая эвристические знания с обученными представлениями.

    Атрибуты:
        input_dim (int): Размер входных данных.
        hidden_dims (List[int]): Размеры скрытых слоёв.
        output_dim (int): Размер выходных данных.
        heuristic_dim (int): Размер эвристических признаков.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, heuristic_dim: int = 5):
        super().__init__()
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
        features = self.feature_layers(x)
        heuristic_features = self.heuristic_layers(heuristic)
        combined = torch.cat([features, heuristic_features], dim=1)
        output = self.combination_layer(combined)
        return output


class HybridAgent(Agent):
    """
    Агент, сочетающий эвристики с обучением нейронной сети.

    Атрибуты:
        player_id (int): Идентификатор игрока, управляемого агентом.
        model (HeuristicNetwork): Модель нейронной сети.
        optimizer (optim.Optimizer): Оптимизатор.
        device (torch.device): Устройство для вычислений (CPU/GPU).
        memory (List): Память опытов.
        batch_size (int): Размер пакета для обучения.
        gamma (float): Фактор дисконтирования.
        epsilon (float): Скорость исследования для стратегии epsilon-greedy.
    """

    def __init__(self, player_id: int, input_dim: int, hidden_dims: List[int], action_dim: int,
                 lr: float = 0.001, device: Optional[torch.device] = None,
                 batch_size: int = 32, gamma: float = 0.99, epsilon: float = 0.1):
        super().__init__(player_id)
        self.device = device if device is not None else torch.device("cpu")
        self.model = HeuristicNetwork(input_dim, hidden_dims, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = []
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.current_state = None
        self.current_heuristic = None
        self.current_action = None
        self.current_reward = 0.0

    def act(self, game_state: GameState) -> Action:
        legal_actions = game_state.get_legal_actions()
        if not legal_actions:
            logger.error(f"HybridAgent {self.player_id}: Нет доступных действий")
            raise ValueError("Нет доступных действий")

        if random.random() < self.epsilon:
            self.current_action = random.choice(legal_actions)
            logger.debug(f"HybridAgent {self.player_id}: Выбрано случайное действие {self.current_action}")
            return self.current_action

        heuristic_features = self._calculate_heuristic_features(game_state)
        state_tensor = self._state_to_tensor(game_state)
        heuristic_tensor = torch.tensor(heuristic_features, dtype=torch.float32).to(self.device).unsqueeze(0)

        self.current_state = game_state.to_feature_vector(self.player_id)
        self.current_heuristic = heuristic_features

        with torch.no_grad():
            action_logits = self.model(state_tensor, heuristic_tensor).squeeze(0)
            legal_mask = torch.zeros(action_logits.size(0), dtype=torch.bool)
            for i in range(min(len(legal_actions), action_logits.size(0))):
                legal_mask[i] = True

            masked_logits = action_logits.clone()
            masked_logits[~legal_mask] = float('-inf')
            probs = F.softmax(masked_logits, dim=0)
            action_idx = torch.multinomial(probs, 1).item()

            self.current_action = legal_actions[action_idx]
            logger.debug(f"HybridAgent {self.player_id}: Выбрано модельное действие {self.current_action}")
            return self.current_action

    def _state_to_tensor(self, game_state: GameState) -> torch.Tensor:
        features = game_state.to_feature_vector(self.player_id)
        tensor = torch.tensor(features, dtype=torch.float32).to(self.device).unsqueeze(0)
        return tensor

    def _calculate_heuristic_features(self, game_state: GameState) -> List[float]:
        if self.player_id >= len(game_state.hole_cards) or self.player_id >= len(game_state.stacks):
            logger.warning(f"HybridAgent {self.player_id}: Недопустимый player_id для доступа к состоянию")
            return [0.0] * 5

        features = []
        hole_cards = game_state.hole_cards[self.player_id]
        community_cards = game_state.community_cards

        if game_state.stage.value >= 2:
            hand = hole_cards + community_cards
            hand_rank, _ = HandEvaluator.evaluate_hand(hand)
            hand_strength = min(1.0, hand_rank / 8.0)
        else:
            if len(hole_cards) == 2:
                rank1, rank2 = hole_cards[0].rank.value, hole_cards[1].rank.value
                suited = hole_cards[0].suit == hole_cards[1].suit
                if rank1 == rank2:
                    hand_strength = 0.5 + (rank1 - 2) / 24.0
                else:
                    rank_strength = (rank1 + rank2 - 4) / 26.0
                    suited_bonus = 0.1 if suited else 0.0
                    hand_strength = min(1.0, rank_strength + suited_bonus)
            else:
                hand_strength = 0.0
        features.append(hand_strength)

        pot_size = game_state.pot + sum(game_state.current_bets)
        amount_to_call = max(game_state.current_bets) - game_state.current_bets[self.player_id] if max(game_state.current_bets) > game_state.current_bets[self.player_id] else 0
        pot_odds = amount_to_call / (pot_size + amount_to_call) if pot_size > 0 and amount_to_call > 0 else 0.0
        features.append(pot_odds)

        position = (self.player_id - game_state.button) % game_state.num_players
        position_value = position / (game_state.num_players - 1)
        features.append(position_value)

        stack = game_state.stacks[self.player_id]
        spr = min(1.0, stack / pot_size) if pot_size > 0 and stack > 0 else 0.0
        features.append(spr)

        active_players = sum(game_state.active) / game_state.num_players
        features.append(active_players)

        return features

    def observe(self, game_state: GameState) -> None:
        if self.current_state is None or self.current_action is None:
            return

        reward = game_state.get_rewards()[self.player_id]
        self.current_reward += reward

        if game_state.is_terminal():
            self.memory.append((
                self.current_state,
                self.current_heuristic,
                self.current_action,
                self.current_reward,
                None,
                None,
                True
            ))
            self._train_step()
            self.current_state = None
            self.current_heuristic = None
            self.current_action = None
            self.current_reward = 0.0
        else:
            next_state = game_state.to_feature_vector(self.player_id)
            next_heuristic = self._calculate_heuristic_features(game_state)
            self.memory.append((
                self.current_state,
                self.current_heuristic,
                self.current_action,
                reward,
                next_state,
                next_heuristic,
                False
            ))
            self.current_state = next_state
            self.current_heuristic = next_heuristic
            self.current_action = None

    def _train_step(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        batch_indices = random.sample(range(len(self.memory)), self.batch_size)
        batch = [self.memory[i] for i in batch_indices]

        states = torch.tensor([x[0] for x in batch], dtype=torch.float32).to(self.device)
        heuristics = torch.tensor([x[1] for x in batch], dtype=torch.float32).to(self.device)
        actions = torch.tensor([x[2] for x in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([x[3] for x in batch], dtype=torch.float32).to(self.device)

        non_terminal_mask = torch.tensor([not x[6] for x in batch], dtype=torch.bool)
        non_terminal_next_states = [x[4] for i, x in enumerate(batch) if not batch[i][6]]
        non_terminal_next_heuristics = [x[5] for i, x in enumerate(batch) if not batch[i][6]]

        q_values = self.model(states, heuristics)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_values = torch.zeros_like(rewards)
        if len(non_terminal_next_states) > 0:
            next_states_tensor = torch.tensor(non_terminal_next_states, dtype=torch.float32).to(self.device)
            next_heuristics_tensor = torch.tensor(non_terminal_next_heuristics, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                next_q = self.model(next_states_tensor, next_heuristics_tensor)
                next_q_max = next_q.max(1)[0]
            next_q_values[non_terminal_mask] = next_q_max

        expected_q_values = rewards + self.gamma * next_q_values
        loss = F.smooth_l1_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if len(self.memory) > 10000:
            self.memory = self.memory[-10000:]