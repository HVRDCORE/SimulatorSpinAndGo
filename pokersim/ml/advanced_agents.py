"""
Implementation of advanced machine learning-based agents for poker simulations.

This module provides sophisticated agent implementations that leverage neural networks
and reinforcement learning techniques to make poker decisions.
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

# Configure logger
logger = logging.getLogger(__name__)

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
        if len(action_probs) != len(legal_actions):
            logger.warning(f"Action probabilities mismatch: {len(action_probs)} vs {len(legal_actions)}")
            action_probs = np.ones(len(legal_actions)) / len(legal_actions)
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
    An agent that uses Proximal Policy Optimization for decision making.

    This agent employs a neural network trained with PPO to select actions
    in Spin and Go poker games, balancing exploration and exploitation.

    Attributes:
        player_id (int): The ID of the player controlled by this agent.
        ppo (PPOSolver): The PPO algorithm instance.
        device (torch.device): Device for computation (CPU/GPU).
        epsilon (float): Exploration rate for epsilon-greedy strategy.
    """

    def __init__(self, player_id: int, game_state_class: Any, num_players: int = 3,
                 device: Optional[torch.device] = None, epsilon: float = 0.1,
                 framework: str = "auto"):
        """
        Initialize the PPO agent.

        Args:
            player_id (int): The ID of the player controlled by this agent.
            game_state_class: Class for creating game states.
            num_players (int, optional): Number of players. Defaults to 3.
            device (torch.device, optional): Device for computation. Defaults to None.
            epsilon (float, optional): Exploration rate. Defaults to 0.1.
            framework (str, optional): ML framework ('pytorch' or 'tensorflow'). Defaults to "auto".
        """
        super().__init__(player_id)
        self.ppo = PPOSolver(game_state_class=game_state_class, num_players=num_players, framework=framework)
        self.device = device if device is not None else torch.device("cpu")
        self.epsilon = epsilon

    def act(self, game_state: GameState) -> Action:
        """
        Choose an action for the current game state.

        Args:
            game_state (GameState): The current game state.

        Returns:
            Action: The chosen action.
        """
        legal_actions = game_state.get_legal_actions()
        if not legal_actions:
            logger.error(f"PPOAgent {self.player_id}: No legal actions available")
            raise ValueError("No legal actions available")

        if random.random() < self.epsilon:
            action = random.choice(legal_actions)
            logger.debug(f"PPOAgent {self.player_id}: Selected random action {action}")
            return action

        action, _, _ = self.ppo.get_action(game_state, self.player_id)
        if action not in legal_actions:
            logger.error(f"PPOAgent {self.player_id}: Model returned illegal action {action}, choosing random")
            action = random.choice(legal_actions)
        logger.debug(f"PPOAgent {self.player_id}: Selected model action {action}")
        return action

    def observe(self, game_state: GameState) -> None:
        """
        Observe the game state (no-op for PPO).

        Args:
            game_state (GameState): The current game state.
        """
        pass

    def train(self, num_iterations: int = 100, num_trajectories: int = 1000) -> Dict[str, float]:
        """
        Train the PPO agent.

        Args:
            num_iterations (int, optional): Number of iterations. Defaults to 100.
            num_trajectories (int, optional): Number of trajectories per iteration. Defaults to 1000.

        Returns:
            Dict[str, float]: Training metrics.
        """
        metrics = self.ppo.train(num_iterations=num_iterations, num_trajectories=num_trajectories)
        return metrics

    def save(self, path: str) -> None:
        """
        Save the PPO model.

        Args:
            path (str): Path to save the model.
        """
        self.ppo.save(path)

    def load(self, path: str) -> None:
        """
        Load the PPO model.

        Args:
            path (str): Path to the saved model.
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
            logger.error(f"ImitationLearningAgent {self.player_id}: No legal actions available")
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

            logger.debug(f"ImitationLearningAgent {self.player_id}: Selected action {legal_actions[action_idx]}")
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
        """Train the model after collecting demonstrations for an episode."""
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
    A neural network that combines heuristic knowledge with learned representations.

    Attributes:
        input_dim (int): The input dimension.
        hidden_dims (List[int]): The hidden layer dimensions.
        output_dim (int): The output dimension.
        heuristic_dim (int): The dimension of heuristic features.
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
    An agent that combines heuristics with neural network learning.

    Attributes:
        player_id (int): The ID of the player controlled by this agent.
        model (HeuristicNetwork): The neural network model.
        optimizer (optim.Optimizer): The optimizer.
        device (torch.device): Device for computation (CPU/GPU).
        memory (List): Memory of experiences.
        batch_size (int): Batch size for training.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate for epsilon-greedy strategy.
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
            logger.error(f"HybridAgent {self.player_id}: No legal actions available")
            raise ValueError("No legal actions available")

        if random.random() < self.epsilon:
            self.current_action = random.choice(legal_actions)
            logger.debug(f"HybridAgent {self.player_id}: Selected random action {self.current_action}")
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
            logger.debug(f"HybridAgent {self.player_id}: Selected model action {self.current_action}")
            return self.current_action

    def _state_to_tensor(self, game_state: GameState) -> torch.Tensor:
        features = game_state.to_feature_vector(self.player_id)
        tensor = torch.tensor(features, dtype=torch.float32).to(self.device).unsqueeze(0)
        return tensor

    def _calculate_heuristic_features(self, game_state: GameState) -> List[float]:
        if self.player_id >= len(game_state.hole_cards) or self.player_id >= len(game_state.stacks):
            logger.warning(f"HybridAgent {self.player_id}: Invalid player_id for state access")
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