"""
Unit tests for the algorithms module of the pokersim package.

This module contains tests for reinforcement learning algorithms like Deep CFR and PPO.
"""

import pytest
import random
import numpy as np
import torch
from typing import List, Dict, Tuple

from pokersim.game.state import GameState, Action, ActionType, Stage
from pokersim.game.card import Card, Suit, Rank
from pokersim.algorithms.deep_cfr import DeepCFR
from pokersim.algorithms.ppo import PPO
from pokersim.ml.models import PokerActorCritic


class TestDeepCFR:
    """Tests for the Deep CFR algorithm."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if CUDA not available")
    def test_initialization(self):
        """Test initialization of Deep CFR."""
        input_dim = 100
        action_dim = 5
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        deep_cfr = DeepCFR(input_dim, action_dim, device)
        
        # Check that networks and optimizers are initialized
        assert deep_cfr.advantage_net is not None
        assert deep_cfr.strategy_net is not None
        assert deep_cfr.advantage_optimizer is not None
        assert deep_cfr.strategy_optimizer is not None
        assert deep_cfr.advantage_memory == []
        assert deep_cfr.strategy_memory == []
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if CUDA not available")
    def test_regret_matching(self):
        """Test regret matching function."""
        deep_cfr = DeepCFR(100, 5)
        
        # Test with positive regrets
        regrets = [3.0, 1.0, 2.0]
        strategy = deep_cfr._regret_matching(regrets)
        
        assert len(strategy) == len(regrets)
        assert sum(strategy) == pytest.approx(1.0)
        assert strategy[0] > strategy[1]  # Higher regret should get higher probability
        
        # Test with negative regrets
        regrets = [-1.0, -2.0, -3.0]
        strategy = deep_cfr._regret_matching(regrets)
        
        assert len(strategy) == len(regrets)
        assert sum(strategy) == pytest.approx(1.0)
        assert strategy[0] == strategy[1] == strategy[2]  # All negative, should be uniform
        
        # Test with mixed regrets
        regrets = [2.0, -1.0, 1.0]
        strategy = deep_cfr._regret_matching(regrets)
        
        assert len(strategy) == len(regrets)
        assert sum(strategy) == pytest.approx(1.0)
        assert strategy[0] > strategy[2] > 0  # Ordering should be preserved for positive values
        assert strategy[1] == 0  # Negative regret should be zero
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if CUDA not available")
    def test_compute_strategy(self):
        """Test strategy computation."""
        input_dim = 2*52 + 5*52 + 1 + 2*3 + 2  # Simplified feature vector size for a 2-player game
        action_dim = 5
        
        deep_cfr = DeepCFR(input_dim, action_dim)
        
        # Create a game state
        game_state = GameState(num_players=2)
        
        # Compute strategy
        strategy = deep_cfr.compute_strategy(game_state, 0)
        
        # Check that strategy is a valid probability distribution
        assert isinstance(strategy, dict)
        assert all(0 <= p <= 1 for p in strategy.values())
        assert sum(strategy.values()) == pytest.approx(1.0)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if CUDA not available")
    def test_train(self):
        """Test training on memory samples."""
        input_dim = 10  # Small dimension for testing
        action_dim = 3
        
        deep_cfr = DeepCFR(input_dim, action_dim)
        
        # Add some fake data to memory
        for _ in range(10):
            state = np.random.random(input_dim)
            cf_values = {0: 1.0, 1: 0.5, 2: 0.0}
            strategy = {0: 0.2, 1: 0.3, 2: 0.5}
            
            deep_cfr.advantage_memory.append((state, cf_values))
            deep_cfr.strategy_memory.append((state, strategy))
        
        # Train
        metrics = deep_cfr.train(batch_size=5, epochs=2)
        
        # Check that metrics are computed
        assert 'advantage_loss' in metrics
        assert 'strategy_loss' in metrics
        
        # Memory should be cleared after training
        assert len(deep_cfr.advantage_memory) == 0
        assert len(deep_cfr.strategy_memory) == 0


class TestPPO:
    """Tests for the PPO algorithm."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if CUDA not available")
    def test_initialization(self):
        """Test initialization of PPO."""
        input_dim = 100
        action_dim = 5
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = PokerActorCritic(input_dim, action_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        ppo = PPO(model, optimizer, device)
        
        # Check that attributes are initialized
        assert ppo.model is model
        assert ppo.optimizer is optimizer
        assert ppo.device == device
        assert ppo.gamma == 0.99  # Default value
        assert ppo.clip_param == 0.2  # Default value
        assert ppo.states == []
        assert ppo.actions == []
        assert ppo.rewards == []
        assert ppo.values == []
        assert ppo.log_probs == []
        assert ppo.dones == []
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if CUDA not available")
    def test_select_action(self):
        """Test action selection."""
        input_dim = 2*52 + 5*52 + 1 + 2*3 + 2  # Simplified feature vector size for a 2-player game
        action_dim = 5
        
        model = PokerActorCritic(input_dim, action_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        ppo = PPO(model, optimizer, device)
        
        # Create a game state
        game_state = GameState(num_players=2)
        
        # Select action
        action = ppo.select_action(game_state, 0)
        
        # Check that action is valid
        assert action in game_state.get_legal_actions()
        
        # Check that trajectory data is stored
        assert len(ppo.states) == 1
        assert len(ppo.actions) == 1
        assert len(ppo.values) == 1
        assert len(ppo.log_probs) == 1
        assert len(ppo.dones) == 1
        assert not ppo.dones[0]  # Should be False initially
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if CUDA not available")
    def test_update_trajectory(self):
        """Test updating trajectory with reward and done flag."""
        input_dim = 10
        action_dim = 3
        
        model = PokerActorCritic(input_dim, action_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        ppo = PPO(model, optimizer)
        
        # Initialize with some fake data
        ppo.states = [np.zeros(input_dim)]
        ppo.actions = [0]
        ppo.values = [0.5]
        ppo.log_probs = [np.log(0.3)]
        ppo.dones = [False]
        
        # Update trajectory
        ppo.update_trajectory(1.0, True)
        
        # Check that reward is added and done flag is updated
        assert len(ppo.rewards) == 1
        assert ppo.rewards[0] == 1.0
        assert ppo.dones[0]  # Should be True now
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if CUDA not available")
    def test_compute_returns(self):
        """Test computing returns from rewards."""
        model = PokerActorCritic(10, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        ppo = PPO(model, optimizer, gamma=0.9)
        
        # Set up rewards and dones
        ppo.rewards = [1.0, 0.0, 2.0]
        ppo.dones = [False, False, True]
        
        # Compute returns with next_value = 0.5
        returns = ppo.compute_returns(0.5)
        
        # Check shape
        assert returns.shape == torch.Size([3])
        
        # Check values (manual calculation with gamma=0.9)
        # R3 = r3 + gamma * v * (1-done3) = 2.0 + 0.9 * 0.5 * (1-1) = 2.0
        # R2 = r2 + gamma * R3 * (1-done2) = 0.0 + 0.9 * 2.0 * (1-0) = 1.8
        # R1 = r1 + gamma * R2 * (1-done1) = 1.0 + 0.9 * 1.8 * (1-0) = 2.62
        expected = torch.tensor([2.62, 1.8, 2.0], dtype=torch.float32)
        assert torch.allclose(returns, expected, rtol=1e-2)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if CUDA not available")
    def test_clear_memory(self):
        """Test clearing memory after training."""
        model = PokerActorCritic(10, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        ppo = PPO(model, optimizer)
        
        # Add some fake data
        ppo.states = [np.zeros(10)]
        ppo.actions = [0]
        ppo.rewards = [1.0]
        ppo.values = [0.5]
        ppo.log_probs = [np.log(0.3)]
        ppo.dones = [False]
        
        # Clear memory
        ppo.clear_memory()
        
        # Check that all lists are empty
        assert len(ppo.states) == 0
        assert len(ppo.actions) == 0
        assert len(ppo.rewards) == 0
        assert len(ppo.values) == 0
        assert len(ppo.log_probs) == 0
        assert len(ppo.dones) == 0
