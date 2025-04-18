"""
Implementation of Neural Fictitious Self-Play (NFSP) algorithm for poker.

This module provides an implementation of the NFSP algorithm for learning
approximate Nash equilibria in zero-sum games through self-play and deep
reinforcement learning. NFSP combines fictitious play with neural networks
to handle large state spaces.
"""

import random
import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Any, Optional
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    raise ImportError("PyTorch is required for NFSP implementation")

from pokersim.game.state import GameState
from pokersim.game.spingo import SpinGoGame
from pokersim.ml.models import create_policy_network, create_dueling_q_network


class ReservoirBuffer:
    """
    Reservoir buffer for storing experience from a best-response policy.
    
    This buffer maintains a random sample of experiences using reservoir sampling.
    
    Attributes:
        capacity (int): Maximum size of the buffer.
        buffer (List): List of experiences.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize a reservoir buffer.
        
        Args:
            capacity (int): Maximum size of the buffer.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer using reservoir sampling.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether the episode is done.
        """
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            # Reservoir sampling: with probability capacity/position, replace a random element
            if random.random() < self.capacity / float(self.position + 1):
                index = random.randint(0, self.capacity - 1)
                self.buffer[index] = experience
        
        self.position += 1
    
    def sample(self, batch_size: int) -> List:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size (int): Size of the batch to sample.
            
        Returns:
            List: A batch of experiences.
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return batch
    
    def __len__(self) -> int:
        """
        Get the current size of the buffer.
        
        Returns:
            int: Number of experiences in the buffer.
        """
        return len(self.buffer)


class AveragePolicyBuffer:
    """
    Buffer for storing experiences to train an average policy.
    
    Attributes:
        capacity (int): Maximum size of the buffer.
        buffer (deque): Experience buffer.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize an average policy buffer.
        
        Args:
            capacity (int): Maximum size of the buffer.
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action_probs):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state.
            action_probs: Action probabilities from best-response policy.
        """
        self.buffer.append((state, action_probs))
    
    def sample(self, batch_size: int) -> List:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size (int): Size of the batch to sample.
            
        Returns:
            List: A batch of experiences.
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        """
        Get the current size of the buffer.
        
        Returns:
            int: Number of experiences in the buffer.
        """
        return len(self.buffer)


class NFSP:
    """
    Neural Fictitious Self-Play algorithm.
    
    This class implements the NFSP algorithm for learning approximate Nash
    equilibria through deep reinforcement learning and self-play.
    
    Attributes:
        q_network (nn.Module): Q-network for the best-response policy.
        target_q_network (nn.Module): Target Q-network for stable learning.
        avg_policy_network (nn.Module): Network for the average policy.
        q_optimizer (optim.Optimizer): Optimizer for the Q-network.
        policy_optimizer (optim.Optimizer): Optimizer for the average policy.
        br_buffer (ReservoirBuffer): Buffer for best-response experiences.
        avg_buffer (AveragePolicyBuffer): Buffer for average policy experiences.
        anticipatory_param (float): Probability of using the best-response policy.
        batch_size (int): Batch size for training.
        gamma (float): Discount factor.
        tau (float): Target network update rate.
    """
    
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256,
                br_buffer_size: int = 100000, avg_buffer_size: int = 100000,
                batch_size: int = 128, gamma: float = 0.99, tau: float = 0.001,
                anticipatory_param: float = 0.1, q_lr: float = 0.001, policy_lr: float = 0.001):
        """
        Initialize NFSP algorithm.
        
        Args:
            input_dim (int): Dimension of state representation.
            action_dim (int): Number of possible actions.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
            br_buffer_size (int, optional): Size of best-response buffer. Defaults to 100000.
            avg_buffer_size (int, optional): Size of average policy buffer. Defaults to 100000.
            batch_size (int, optional): Batch size for training. Defaults to 128.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            tau (float, optional): Target network update rate. Defaults to 0.001.
            anticipatory_param (float, optional): Anticipatory parameter. Defaults to 0.1.
            q_lr (float, optional): Learning rate for Q-network. Defaults to 0.001.
            policy_lr (float, optional): Learning rate for policy network. Defaults to 0.001.
        """
        # Create networks
        self.q_network = create_dueling_q_network(input_dim, [hidden_dim, hidden_dim], action_dim)
        self.target_q_network = create_dueling_q_network(input_dim, [hidden_dim, hidden_dim], action_dim)
        self.avg_policy_network = create_policy_network(input_dim, [hidden_dim, hidden_dim], action_dim)
        
        # Copy parameters from Q-network to target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Create optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.avg_policy_network.parameters(), lr=policy_lr)
        
        # Create buffers
        self.br_buffer = ReservoirBuffer(br_buffer_size)
        self.avg_buffer = AveragePolicyBuffer(avg_buffer_size)
        
        # Set parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.anticipatory_param = anticipatory_param
        self.action_dim = action_dim
        
        # Device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_q_network.to(self.device)
        self.avg_policy_network.to(self.device)
        
        # Metrics
        self.q_losses = []
        self.policy_losses = []
        self.episodes = 0
    
    def act(self, state: np.ndarray, legal_actions: List[Any], eval_mode: bool = False) -> Any:
        """
        Choose an action using the current policy.
        
        In training mode, this uses both the best-response and average policy
        according to the anticipatory parameter. In evaluation mode, it uses
        only the average policy.
        
        Args:
            state (np.ndarray): State representation.
            legal_actions (List[Any]): List of legal actions.
            eval_mode (bool, optional): Whether to use evaluation mode. Defaults to False.
        
        Returns:
            Any: The chosen action.
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if eval_mode or random.random() > self.anticipatory_param:
            # Use average policy
            with torch.no_grad():
                action_probs = F.softmax(self.avg_policy_network(state_tensor), dim=1)
                action_probs = action_probs.squeeze(0).cpu().numpy()
            
            # Mask illegal actions
            action_mask = np.zeros(self.action_dim)
            for i, action in enumerate(legal_actions):
                action_type = self._action_to_idx(action)
                action_mask[action_type] = 1
            
            masked_probs = action_probs * action_mask
            if np.sum(masked_probs) > 0:
                masked_probs = masked_probs / np.sum(masked_probs)
            else:
                # Fallback to uniform distribution over legal actions
                masked_probs = action_mask / np.sum(action_mask)
            
            # Sample action
            action_idx = np.random.choice(len(masked_probs), p=masked_probs)
            
            # Find the corresponding action
            for action in legal_actions:
                if self._action_to_idx(action) == action_idx:
                    chosen_action = action
                    break
            else:
                # Fallback if no matching action type
                chosen_action = legal_actions[0]
            
            # Store for average policy training (if not in eval mode)
            if not eval_mode:
                # Get best-response action probabilities for supervised learning
                with torch.no_grad():
                    q_values = self.q_network(state_tensor).squeeze(0).cpu().numpy()
                
                # Mask illegal actions
                q_values = q_values * action_mask - 1e9 * (1 - action_mask)
                
                # Convert to probabilities (greedy, not epsilon-greedy)
                best_action = np.argmax(q_values)
                br_probs = np.zeros_like(q_values)
                br_probs[best_action] = 1.0
                
                # Store in average policy buffer
                self.avg_buffer.push(state, br_probs)
        else:
            # Use best-response policy (epsilon-greedy)
            with torch.no_grad():
                q_values = self.q_network(state_tensor).squeeze(0).cpu().numpy()
            
            # Mask illegal actions
            action_mask = np.zeros(self.action_dim)
            for i, action in enumerate(legal_actions):
                action_type = self._action_to_idx(action)
                action_mask[action_type] = 1
            
            q_values = q_values * action_mask - 1e9 * (1 - action_mask)
            
            # Epsilon-greedy action selection
            epsilon = 0.1  # Fixed epsilon for simplicity
            if random.random() < epsilon:
                action_idx = random.randrange(len(legal_actions))
                chosen_action = legal_actions[action_idx]
            else:
                action_idx = np.argmax(q_values)
                # Find the corresponding action
                for action in legal_actions:
                    if self._action_to_idx(action) == action_idx:
                        chosen_action = action
                        break
                else:
                    # Fallback if no matching action type
                    chosen_action = legal_actions[0]
        
        return chosen_action
    
    def _action_to_idx(self, action: Any) -> int:
        """
        Convert an action to an index.
        
        Args:
            action (Any): The action.
        
        Returns:
            int: Index representation of the action.
        """
        # This is a simplified representation - in practice, you would have a more
        # sophisticated mapping based on your action space
        if hasattr(action, "action_type"):
            return action.action_type.value - 1  # Assuming 1-indexed enum
        else:
            # Fallback for simpler action types
            return hash(str(action)) % self.action_dim
    
    def store_transition(self, state: np.ndarray, action: Any, 
                        reward: float, next_state: np.ndarray, done: bool):
        """
        Store a transition in the best-response buffer.
        
        Args:
            state (np.ndarray): Current state.
            action (Any): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool): Whether the episode is done.
        """
        action_idx = self._action_to_idx(action)
        self.br_buffer.push(state, action_idx, reward, next_state, done)
    
    def train(self):
        """Train the Q-network and average policy network."""
        # Only train if we have enough samples
        if len(self.br_buffer) < self.batch_size or len(self.avg_buffer) < self.batch_size:
            return
        
        # Train Q-network (best-response)
        q_loss = self._train_q_network()
        self.q_losses.append(q_loss)
        
        # Train average policy network
        policy_loss = self._train_avg_policy()
        self.policy_losses.append(policy_loss)
        
        # Update target network
        self._update_target_network()
    
    def _train_q_network(self) -> float:
        """
        Train the Q-network using experiences from the best-response buffer.
        
        Returns:
            float: The loss value.
        """
        # Sample a batch from the best-response buffer
        batch = self.br_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones, dtype=np.float32)).unsqueeze(1).to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        
        return loss.item()
    
    def _train_avg_policy(self) -> float:
        """
        Train the average policy network using experiences from the average policy buffer.
        
        Returns:
            float: The loss value.
        """
        # Sample a batch from the average policy buffer
        batch = self.avg_buffer.sample(self.batch_size)
        states, action_probs = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        action_probs = torch.FloatTensor(np.array(action_probs)).to(self.device)
        
        # Forward pass
        policy_logits = self.avg_policy_network(states)
        
        # Cross-entropy loss
        loss = F.cross_entropy(policy_logits, action_probs)
        
        # Optimize
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        
        return loss.item()
    
    def _update_target_network(self):
        """Update the target network parameters."""
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
    
    def train_episode(self, env):
        """
        Train for one episode.
        
        Args:
            env: Environment to train in.
        
        Returns:
            float: Total episode reward.
        """
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Choose and perform action
            legal_actions = env.get_legal_actions()
            action = self.act(state, legal_actions)
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            self.store_transition(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            total_reward += reward
            
            # Train
            self.train()
        
        self.episodes += 1
        return total_reward
    
    def evaluate(self, env, num_episodes: int = 10) -> float:
        """
        Evaluate the agent.
        
        Args:
            env: Environment to evaluate in.
            num_episodes (int, optional): Number of episodes. Defaults to 10.
        
        Returns:
            float: Average reward.
        """
        total_reward = 0
        
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Choose action
                legal_actions = env.get_legal_actions()
                action = self.act(state, legal_actions, eval_mode=True)
                
                # Take action
                next_state, reward, done, _ = env.step(action)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
            
            total_reward += episode_reward
        
        return total_reward / num_episodes


class NFSPPlayer:
    """
    Player that uses NFSP to make decisions.
    
    Attributes:
        player_id (int): Player ID.
        nfsp (NFSP): NFSP algorithm instance.
        eval_mode (bool): Whether to use evaluation mode.
    """
    
    def __init__(self, player_id: int, nfsp: NFSP, eval_mode: bool = True):
        """
        Initialize NFSP player.
        
        Args:
            player_id (int): Player ID.
            nfsp (NFSP): NFSP algorithm instance.
            eval_mode (bool, optional): Whether to use evaluation mode. Defaults to True.
        """
        self.player_id = player_id
        self.nfsp = nfsp
        self.eval_mode = eval_mode
    
    def act(self, game_state: GameState) -> Any:
        """
        Choose an action using NFSP.
        
        Args:
            game_state (GameState): Current game state.
        
        Returns:
            Any: The chosen action.
        """
        # Get state representation
        state = self._get_state_representation(game_state)
        
        # Get legal actions
        legal_actions = game_state.get_legal_actions()
        
        # Choose action
        return self.nfsp.act(state, legal_actions, eval_mode=self.eval_mode)
    
    def _get_state_representation(self, game_state: GameState) -> np.ndarray:
        """
        Get a vector representation of the game state.
        
        Args:
            game_state (GameState): Current game state.
        
        Returns:
            np.ndarray: Vector representation of the state.
        """
        # This is a simplified representation - in practice, you would have a more
        # sophisticated state representation
        return np.zeros(128)  # Placeholder state representation