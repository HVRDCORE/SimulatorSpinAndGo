"""
Integration with TensorFlow for poker simulations.

This module provides classes and functions for integrating TensorFlow with
the poker simulation framework, enabling the development of agents using
TensorFlow's neural network capabilities.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

from pokersim.game.state import GameState, Action
from pokersim.agents.base_agent import Agent


class PokerMLP_TF(keras.Model):
    """
    A multi-layer perceptron for poker decision-making implemented in TensorFlow.
    
    Attributes:
        input_dim (int): The input dimension.
        hidden_dims (List[int]): The hidden layer dimensions.
        output_dim (int): The output dimension.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        """
        Initialize a poker MLP.
        
        Args:
            input_dim (int): The input dimension.
            hidden_dims (List[int]): The hidden layer dimensions.
            output_dim (int): The output dimension.
        """
        super(PokerMLP_TF, self).__init__()
        
        self.input_layer = keras.layers.InputLayer(input_shape=(input_dim,))
        
        self.hidden_layers = []
        for dim in hidden_dims:
            self.hidden_layers.append(keras.layers.Dense(
                dim, activation='relu',
                kernel_initializer='he_normal'
            ))
        
        self.output_layer = keras.layers.Dense(
            output_dim,
            kernel_initializer='glorot_uniform'
        )
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass.
        
        Args:
            inputs (tf.Tensor): The input tensor.
            training (bool, optional): Whether in training mode. Defaults to False.
            
        Returns:
            tf.Tensor: The output tensor.
        """
        x = self.input_layer(inputs)
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        return self.output_layer(x)


class PokerCNN_TF(keras.Model):
    """
    A convolutional neural network for poker decision-making implemented in TensorFlow.
    
    This model uses convolutional layers to extract features from card
    representations, followed by fully-connected layers for decision-making.
    
    Attributes:
        input_dim (int): The input dimension.
        output_dim (int): The output dimension.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize a poker CNN.
        
        Args:
            input_dim (int): The input dimension.
            output_dim (int): The output dimension.
        """
        super(PokerCNN_TF, self).__init__()
        
        # Reshape input for convolutional layers
        self.input_dim = input_dim
        
        # Convolutional layers for card features
        self.conv1 = keras.layers.Conv1D(32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv2 = keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.pool = keras.layers.MaxPooling1D(pool_size=2, strides=2)
        
        # Flatten for fully-connected layers
        self.flatten = keras.layers.Flatten()
        
        # Fully-connected layers
        fc_input_dim = 64 * (input_dim // 4)  # After 2 pooling layers
        self.fc1 = keras.layers.Dense(128, activation='relu')
        self.fc2 = keras.layers.Dense(64, activation='relu')
        self.fc3 = keras.layers.Dense(output_dim)
        
        # Dropout for regularization
        self.dropout = keras.layers.Dropout(0.2)
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass.
        
        Args:
            inputs (tf.Tensor): The input tensor.
            training (bool, optional): Whether in training mode. Defaults to False.
            
        Returns:
            tf.Tensor: The output tensor.
        """
        # Reshape for convolutional layers
        x = tf.reshape(inputs, [-1, self.input_dim, 1])
        
        # Convolutional layers
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        
        # Flatten for fully-connected layers
        x = self.flatten(x)
        
        # Fully-connected layers with dropout
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        x = self.fc3(x)
        
        return x


class PokerActorCritic_TF(keras.Model):
    """
    An actor-critic model for poker decision-making implemented in TensorFlow.
    
    This model outputs both policy (actor) and value (critic) predictions.
    
    Attributes:
        input_dim (int): The input dimension.
        action_dim (int): The number of possible actions.
    """
    
    def __init__(self, input_dim: int, action_dim: int):
        """
        Initialize a poker actor-critic model.
        
        Args:
            input_dim (int): The input dimension.
            action_dim (int): The number of possible actions.
        """
        super(PokerActorCritic_TF, self).__init__()
        
        # Input layer
        self.input_layer = keras.layers.InputLayer(input_shape=(input_dim,))
        
        # Shared layers
        self.shared_1 = keras.layers.Dense(128, activation='relu')
        self.shared_2 = keras.layers.Dense(64, activation='relu')
        
        # Actor (policy) head
        self.actor_1 = keras.layers.Dense(32, activation='relu')
        self.actor_2 = keras.layers.Dense(action_dim)
        
        # Critic (value) head
        self.critic_1 = keras.layers.Dense(32, activation='relu')
        self.critic_2 = keras.layers.Dense(1)
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass.
        
        Args:
            inputs (tf.Tensor): The input tensor.
            training (bool, optional): Whether in training mode. Defaults to False.
            
        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The policy and value predictions.
        """
        x = self.input_layer(inputs)
        
        # Shared representation
        shared = self.shared_1(x)
        shared = self.shared_2(shared)
        
        # Actor (policy) head
        policy = self.actor_1(shared)
        policy = self.actor_2(policy)
        
        # Critic (value) head
        value = self.critic_1(shared)
        value = self.critic_2(value)
        
        return policy, value


class TFReplayBuffer:
    """
    Replay buffer for storing experiences during training with TensorFlow.
    
    Attributes:
        states (List[np.ndarray]): The states.
        actions (List[int]): The actions.
        rewards (List[float]): The rewards.
        next_states (List[np.ndarray]): The next states.
        dones (List[bool]): Whether the episodes are done.
        capacity (int): The capacity of the buffer.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize a replay buffer.
        
        Args:
            capacity (int, optional): The capacity. Defaults to 10000.
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.capacity = capacity
    
    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            state (np.ndarray): The state.
            action (int): The action.
            reward (float): The reward.
            next_state (np.ndarray): The next state.
            done (bool): Whether the episode is done.
        """
        # If buffer is full, remove oldest experience
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
        
        # Add new experience
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def sample(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Sample experiences from the buffer.
        
        Args:
            batch_size (int): The batch size.
            
        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: The sampled experiences.
        """
        # Sample indices
        indices = np.random.choice(len(self.states), min(batch_size, len(self.states)), replace=False)
        
        # Get sampled experiences
        states = np.array([self.states[i] for i in indices])
        actions = np.array([self.actions[i] for i in indices])
        rewards = np.array([self.rewards[i] for i in indices])
        next_states = np.array([self.next_states[i] for i in indices])
        dones = np.array([self.dones[i] for i in indices])
        
        # Convert to TensorFlow tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Get the current size of the buffer."""
        return len(self.states)


class TFAgent(Agent):
    """
    Base class for TensorFlow-based poker agents.
    
    Attributes:
        player_id (int): The ID of the player controlled by this agent.
        model: The TensorFlow model.
        optimizer: The optimizer.
    """
    
    def __init__(self, player_id: int, model: keras.Model, optimizer: keras.optimizers.Optimizer):
        """
        Initialize a TensorFlow agent.
        
        Args:
            player_id (int): The ID of the player controlled by this agent.
            model (keras.Model): The TensorFlow model.
            optimizer (keras.optimizers.Optimizer): The optimizer.
        """
        super().__init__(player_id)
        self.model = model
        self.optimizer = optimizer
    
    def state_to_tensor(self, game_state: GameState) -> tf.Tensor:
        """
        Convert a game state to a TensorFlow tensor.
        
        Args:
            game_state (GameState): The game state.
            
        Returns:
            tf.Tensor: The tensor representation of the game state.
        """
        # Get feature vector
        features = game_state.to_feature_vector(self.player_id)
        features = np.resize(features, 128)
        
        # Convert to tensor
        tensor = tf.convert_to_tensor(features, dtype=tf.float32)
        tensor = tf.expand_dims(tensor, axis=0)
        
        return tf.expand_dims(tensor, axis=0)

        policy_logits = tf.clip_by_value(policy_logits, -10, 10)
    
    def tensor_to_action(self, tensor: tf.Tensor, game_state: GameState) -> Action:
        """
        Convert a TensorFlow tensor to an action.
        
        Args:
            tensor (tf.Tensor): The tensor.
            game_state (GameState): The game state.
            
        Returns:
            Action: The action.
        """
        # Get legal actions
        legal_actions = game_state.get_legal_actions()
        
        # If no legal actions, return None
        if not legal_actions:
            return None
        
        # Get action probabilities
        probs = tf.nn.softmax(tensor).numpy()[0]
        
        # Choose action with highest probability
        action_idx = np.argmax(probs)
        
        # Map to legal action
        return legal_actions[min(action_idx, len(legal_actions) - 1)]
    
    def act(self, game_state: GameState) -> Action:
        """
        Choose an action based on the current game state.
        
        Args:
            game_state (GameState): The current game state.
            
        Returns:
            Action: The chosen action.
        """
        # Implement in subclasses
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path (str): The file path.
        """
        self.model.save_weights(path)
    
    def load(self, path: str) -> None:
        """
        Load the model from a file.
        
        Args:
            path (str): The file path.
        """
        self.model.load_weights(path)


class DQNAgentTF(TFAgent):
    """
    A Deep Q-Network agent implemented in TensorFlow.
    
    Attributes:
        player_id (int): The ID of the player controlled by this agent.
        model: The Q-network model.
        target_model: The target Q-network model.
        optimizer: The optimizer.
        buffer: The replay buffer.
        epsilon (float): The exploration rate.
        gamma (float): The discount factor.
        batch_size (int): The batch size for training.
        target_update_freq (int): The frequency of target network updates.
        step_counter (int): Counter for target network updates.
    """
    
    def __init__(self, player_id: int, model: keras.Model, optimizer: keras.optimizers.Optimizer,
                epsilon: float = 0.1, gamma: float = 0.99, batch_size: int = 64, 
                target_update_freq: int = 100):
        """
        Initialize a DQN agent.
        
        Args:
            player_id (int): The ID of the player controlled by this agent.
            model (keras.Model): The Q-network model.
            optimizer (keras.optimizers.Optimizer): The optimizer.
            epsilon (float, optional): The exploration rate. Defaults to 0.1.
            gamma (float, optional): The discount factor. Defaults to 0.99.
            batch_size (int, optional): The batch size for training. Defaults to 64.
            target_update_freq (int, optional): The frequency of target network updates. Defaults to 100.
        """
        super().__init__(player_id, model, optimizer)
        
        # Create a clone of the model for the target network
        self.target_model = keras.models.clone_model(model)
        self.target_model.set_weights(model.get_weights())
        
        self.buffer = TFReplayBuffer()
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_counter = 0
        
        # For storing the current state and action
        self.current_state = None
        self.current_action = None
    
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
        
        # Store current state
        self.current_state = game_state.to_feature_vector(self.player_id)
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            action = random.choice(legal_actions)
            self.current_action = legal_actions.index(action)
            return action
        
        # Convert state to tensor
        state_tensor = self.state_to_tensor(game_state)
        
        # Get Q-values from model
        q_values = self.model(state_tensor, training=False)
        
        # Filter out illegal actions
        legal_q_values = q_values.numpy()[0][:len(legal_actions)]
        
        # Choose action with highest Q-value
        action_idx = np.argmax(legal_q_values)
        
        self.current_action = action_idx
        return legal_actions[action_idx]
    
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
        
        # Get next state
        next_state = game_state.to_feature_vector(self.player_id)
        
        # Check if the episode is done
        done = game_state.is_terminal()
        
        # Add experience to buffer
        self.buffer.add(self.current_state, self.current_action, reward, next_state, done)
        
        # Train the model
        self.train()
        
        # Update current state and action
        self.current_state = next_state
        self.current_action = None
    
    def train(self) -> None:
        """Train the model on a batch of experiences."""
        # Skip if the buffer doesn't have enough experiences
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Compute target Q-values
        next_q_values = self.target_model(next_states, training=False)
        max_next_q = tf.reduce_max(next_q_values, axis=1)
        target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Compute loss and update model
        with tf.GradientTape() as tape:
            # Current Q-values
            current_q = self.model(states, training=True)
            
            # Gather Q-values for the actions taken
            action_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), actions], axis=1)
            current_q_selected = tf.gather_nd(current_q, action_indices)
            
            # Compute loss
            loss = keras.losses.MSE(target_q, current_q_selected)
        
        # Compute gradients and update weights
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Update target network periodically
        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())


class ActorCriticAgentTF(TFAgent):
    """
    An actor-critic agent implemented in TensorFlow.
    
    Attributes:
        player_id (int): The ID of the player controlled by this agent.
        model: The actor-critic model.
        optimizer: The optimizer.
        gamma (float): The discount factor.
        states: List of states in the current trajectory.
        actions: List of actions in the current trajectory.
        rewards: List of rewards in the current trajectory.
        dones: List of done flags in the current trajectory.
    """
    
    def __init__(self, player_id: int, model: PokerActorCritic_TF, 
                optimizer: keras.optimizers.Optimizer, gamma: float = 0.99):
        """
        Initialize an actor-critic agent.
        
        Args:
            player_id (int): The ID of the player controlled by this agent.
            model (PokerActorCritic_TF): The actor-critic model.
            optimizer (keras.optimizers.Optimizer): The optimizer.
            gamma (float, optional): The discount factor. Defaults to 0.99.
        """
        super().__init__(player_id, model, optimizer)
        self.gamma = gamma
        
        # For storing trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
    
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
        
        # Convert state to tensor
        state_tensor = self.state_to_tensor(game_state)
        
        # Get policy and value from model
        policy, _ = self.model(state_tensor, training=False)
        
        # Filter out illegal actions
        legal_policy = policy.numpy()[0][:len(legal_actions)]
        
        # Apply softmax to get probabilities
        probs = tf.nn.softmax(legal_policy).numpy()
        
        # Sample action according to probabilities
        action_idx = np.random.choice(len(legal_actions), p=probs)
        
        # Store state and action
        self.states.append(game_state.to_feature_vector(self.player_id))
        self.actions.append(action_idx)
        
        return legal_actions[action_idx]
    
    def observe(self, game_state: GameState) -> None:
        """
        Update the agent's internal state based on the current game state.
        
        Args:
            game_state (GameState): The current game state.
        """
        # Skip if we don't have any states or actions
        if not self.states or not self.actions:
            return
        
        # Get reward
        reward = game_state.get_rewards()[self.player_id]
        
        # Check if the episode is done
        done = game_state.is_terminal()
        
        # Store reward and done flag
        self.rewards.append(reward)
        self.dones.append(done)
        
        # If the episode is done, train on the trajectory
        if done:
            self.train()
    
    def train(self) -> None:
        """Train the model on the collected trajectory."""
        # Skip if we don't have enough data
        if not self.states or not self.actions or not self.rewards:
            return
        
        # Convert to tensors
        states = tf.convert_to_tensor(np.array(self.states), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.array(self.actions), dtype=tf.int32)
        rewards = tf.convert_to_tensor(np.array(self.rewards), dtype=tf.float32)
        dones = tf.convert_to_tensor(np.array(self.dones), dtype=tf.float32)
        
        # Compute returns
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        
        returns = tf.convert_to_tensor(np.array(returns), dtype=tf.float32)
        
        # Normalize returns
        returns = (returns - tf.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-8)
        
        # Train the model
        with tf.GradientTape() as tape:
            # Forward pass
            policy_logits, values = self.model(states, training=True)
            
            # Gather policy logits for the actions taken
            action_indices = tf.stack([tf.range(len(actions), dtype=tf.int32), actions], axis=1)
            policy_logits_taken = tf.gather_nd(policy_logits, action_indices)
            
            # Compute advantages
            advantages = returns - tf.squeeze(values)
            
            # Compute actor (policy) loss using policy gradients
            policy_loss = -tf.reduce_mean(tf.math.log(tf.nn.softmax(policy_logits_taken) + 1e-8) * advantages)
            
            # Compute critic (value) loss
            value_loss = tf.reduce_mean(tf.square(advantages))
            
            # Compute entropy loss for exploration
            entropy = tf.reduce_mean(tf.reduce_sum(
                tf.nn.softmax(policy_logits, axis=1) * 
                tf.math.log(tf.nn.softmax(policy_logits, axis=1) + 1e-8),
                axis=1
            ))
            
            # Compute total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        # Compute gradients and update weights
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Clear trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
