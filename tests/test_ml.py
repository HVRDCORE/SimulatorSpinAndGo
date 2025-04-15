"""
Unit tests for the ml module of the pokersim package.

This module contains tests for machine learning components, models, and ML-based agents.
"""

import pytest
import random
import numpy as np
import torch
from typing import List, Dict, Tuple

from pokersim.game.state import GameState, Action, ActionType, Stage
from pokersim.ml.models import PokerMLP, PokerCNN, PokerActorCritic
from pokersim.ml.torch_integration import TorchAgent, ExperienceBuffer, ActorCriticTrainer
from pokersim.ml.advanced_agents import DeepCFRAgent, PPOAgent, ImitationLearningAgent, HybridAgent


class TestModels:
    """Tests for neural network models."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if CUDA not available")
    def test_poker_mlp(self):
        """Test PokerMLP model."""
        input_dim = 100
        hidden_dims = [64, 32]
        output_dim = 5
        
        model = PokerMLP(input_dim, hidden_dims, output_dim)
        
        # Check model structure
        assert isinstance(model, torch.nn.Module)
        assert isinstance(model.layers, torch.nn.Sequential)
        
        # Test forward pass
        x = torch.randn(10, input_dim)
        output = model(x)
        
        assert output.shape == (10, output_dim)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if CUDA not available")
    def test_poker_cnn(self):
        """Test PokerCNN model."""
        input_dim = 364  # Needs to be divisible by 4 after 2 pooling layers
        output_dim = 5
        
        model = PokerCNN(input_dim, output_dim)
        
        # Test forward pass
        x = torch.randn(10, input_dim)
        output = model(x)
        
        assert output.shape == (10, output_dim)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if CUDA not available")
    def test_poker_actor_critic(self):
        """Test PokerActorCritic model."""
        input_dim = 100
        action_dim = 5
        
        model = PokerActorCritic(input_dim, action_dim)
        
        # Test forward pass
        x = torch.randn(10, input_dim)
        policy, value = model(x)
        
        assert policy.shape == (10, action_dim)
        assert value.shape == (10, 1)


class TestTorchIntegration:
    """Tests for PyTorch integration."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if CUDA not available")
    def test_experience_buffer(self):
        """Test ExperienceBuffer."""
        buffer = ExperienceBuffer(capacity=5)
        
        # Add experiences
        for i in range(3):
            buffer.add(
                state=np.array([i, i+1]),
                action=i,
                reward=float(i),
                next_state=np.array([i+1, i+2]),
                done=(i == 2)
            )
        
        # Check buffer size
        assert len(buffer) == 3
        
        # Sample from buffer
        states, actions, rewards, next_states, dones = buffer.sample(2)
        
        assert states.shape == (2, 2)
        assert actions.shape == (2,)
        assert rewards.shape == (2,)
        assert next_states.shape == (2, 2)
        assert dones.shape == (2,)
        
        # Check overflow behavior
        for i in range(5):
            buffer.add(
                state=np.array([i+10, i+11]),
                action=i+10,
                reward=float(i+10),
                next_state=np.array([i+11, i+12]),
                done=(i == 4)
            )
        
        # Buffer should be at capacity
        assert len(buffer) == 5
        
        # First items should be dropped
        states, actions, rewards, next_states, dones = buffer.sample(5)
        assert np.min(actions) >= 10  # First actions (0, 1, 2) should be dropped
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if CUDA not available")
    def test_actor_critic_trainer(self):
        """Test ActorCriticTrainer."""
        # Create model and trainer
        input_dim = 10
        action_dim = 3
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = PokerActorCritic(input_dim, action_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        trainer = ActorCriticTrainer(model, optimizer, device, gamma=0.9)
        
        # Create batch data
        batch_size = 4
        states = torch.randn(batch_size, input_dim, device=device)
        actions = torch.randint(0, action_dim, (batch_size,), device=device)
        rewards = torch.rand(batch_size, device=device)
        next_states = torch.randn(batch_size, input_dim, device=device)
        dones = torch.zeros(batch_size, device=device)
        dones[-1] = 1  # Last one is terminal
        
        # Train step
        metrics = trainer.train_step(states, actions, rewards, next_states, dones)
        
        # Check metrics
        assert 'loss' in metrics
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert 'entropy' in metrics


class TestAdvancedAgents:
    """Tests for advanced ML agents."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if CUDA not available")
    def test_deep_cfr_agent(self):
        """Test DeepCFRAgent."""
        input_dim = 2*52 + 5*52 + 1 + 2*3 + 2  # Simplified feature vector size for a 2-player game
        action_dim = 5
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        agent = DeepCFRAgent(0, input_dim, action_dim, device, epsilon=0.1)
        
        # Create game state
        game_state = GameState(num_players=2)
        
        # Test action selection
        action = agent.act(game_state)
        assert action in game_state.get_legal_actions()
        
        # Test training (minimal)
        metrics = agent.train(game_state, num_trajectories=2, batch_size=2, epochs=1)
        assert 'advantage_loss' in metrics
        assert 'strategy_loss' in metrics
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if CUDA not available")
    def test_ppo_agent(self):
        """Test PPOAgent."""
        input_dim = 2*52 + 5*52 + 1 + 2*3 + 2  # Simplified feature vector size for a 2-player game
        action_dim = 5
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        agent = PPOAgent(0, input_dim, action_dim, lr=0.001, device=device, epsilon=0.1)
        
        # Create game state
        game_state = GameState(num_players=2)
        
        # Test action selection
        action = agent.act(game_state)
        assert action in game_state.get_legal_actions()
        
        # Test observation
        agent.observe(game_state)
        
        # Terminal state should trigger training
        terminal_state = type('MockGameState', (), {
            'is_terminal': lambda: True,
            'get_rewards': lambda: [1.0, 0.0],
            'to_feature_vector': lambda player_id: np.zeros(input_dim)
        })
        
        # This won't actually train, just check that it doesn't crash
        agent.observe(terminal_state)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if CUDA not available")
    def test_imitation_learning_agent(self):
        """Test ImitationLearningAgent."""
        input_dim = 2*52 + 5*52 + 1 + 2*3 + 2  # Simplified feature vector size for a 2-player game
        hidden_dims = [64, 32]
        action_dim = 5
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create expert (random for testing)
        from pokersim.agents.random_agent import RandomAgent
        expert = RandomAgent(0)
        
        agent = ImitationLearningAgent(0, input_dim, hidden_dims, action_dim, 
                                       lr=0.001, device=device, expert=expert, batch_size=2)
        
        # Create game state
        game_state = GameState(num_players=2)
        
        # Test action selection (should also collect demonstration)
        action = agent.act(game_state)
        assert action in game_state.get_legal_actions()
        
        # Check that memory was updated
        assert len(agent.memory) > 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if CUDA not available")
    def test_hybrid_agent(self):
        """Test HybridAgent."""
        input_dim = 2*52 + 5*52 + 1 + 2*3 + 2  # Simplified feature vector size for a 2-player game
        hidden_dims = [64, 32, 16]
        action_dim = 5
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        agent = HybridAgent(0, input_dim, hidden_dims, action_dim, 
                           lr=0.001, device=device, batch_size=2)
        
        # Create game state
        game_state = GameState(num_players=2)
        
        # Test heuristic feature calculation
        features = agent._calculate_heuristic_features(game_state)
        assert len(features) == 5  # Should have 5 heuristic features
        
        # Test action selection
        action = agent.act(game_state)
        assert action in game_state.get_legal_actions()
        
        # Test observation
        agent.observe(game_state)
        
        # Terminal state should add to memory and train
        terminal_state = type('MockGameState', (), {
            'is_terminal': lambda: True,
            'get_rewards': lambda: [1.0, 0.0],
            'to_feature_vector': lambda player_id: np.zeros(input_dim),
            'hole_cards': [[Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES)]],
            'community_cards': [],
            'pot': 10,
            'current_bets': [5, 5],
            'button': 0,
            'num_players': 2,
            'active': [True, True],
            'stacks': [95, 95]
        })
        
        # This won't actually train (not enough samples), just check it doesn't crash
        agent.observe(terminal_state)
