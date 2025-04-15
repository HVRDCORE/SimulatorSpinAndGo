"""
Unit tests for the agents module of the pokersim package.

This module contains tests for various poker playing agents.
"""

import pytest
import random
import numpy as np
import torch
from typing import List, Dict, Tuple

from pokersim.game.state import GameState, Action, ActionType, Stage
from pokersim.agents.base_agent import Agent
from pokersim.agents.random_agent import RandomAgent
from pokersim.agents.call_agent import CallAgent
from pokersim.agents.rule_based_agent import RuleBased1Agent, RuleBased2Agent
from pokersim.ml.agents import MLAgent, RandomMLAgent, TorchMLAgent
from pokersim.ml.models import PokerMLP
from pokersim.ml.advanced_agents import PPOAgent, DeepCFRAgent, HybridAgent


class TestBaseAgent:
    """Tests for the base Agent class."""
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        # Should not be able to instantiate abstract base class
        with pytest.raises(TypeError):
            Agent(0)
        
        # Create a minimal concrete implementation
        class ConcreteAgent(Agent):
            def act(self, game_state):
                return game_state.get_legal_actions()[0]
        
        # Should be able to instantiate concrete implementation
        agent = ConcreteAgent(0)
        assert agent.player_id == 0
        
        # Default methods should be implemented
        agent.observe(None)  # No-op by default
        agent.reset()  # No-op by default
        agent.end_hand(None)  # No-op by default


class TestRandomAgent:
    """Tests for the RandomAgent class."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = RandomAgent(1)
        assert agent.player_id == 1
    
    def test_act(self):
        """Test that the agent selects random actions."""
        agent = RandomAgent(0)
        game_state = GameState(num_players=2)
        
        # Set seed for reproducibility
        random.seed(42)
        
        # Act multiple times
        actions = []
        for _ in range(10):
            action = agent.act(game_state)
            actions.append(action)
            
            # Action should be in legal actions
            assert action in game_state.get_legal_actions()
        
        # Should have some variety in actions due to randomness
        action_types = set(action.action_type for action in actions)
        assert len(action_types) > 1
        
        # Should raise an error if no legal actions
        game_state_mock = type('MockGameState', (), {'get_legal_actions': lambda: []})
        with pytest.raises(ValueError):
            agent.act(game_state_mock)


class TestCallAgent:
    """Tests for the CallAgent class."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = CallAgent(1)
        assert agent.player_id == 1
    
    def test_act_prefer_check(self):
        """Test that the agent prefers to check when possible."""
        agent = CallAgent(0)
        
        # Create a mock game state where check is available
        check_action = Action(ActionType.CHECK)
        call_action = Action(ActionType.CALL)
        legal_actions = [Action(ActionType.FOLD), check_action, call_action]
        
        game_state_mock = type('MockGameState', (), {'get_legal_actions': lambda: legal_actions})
        
        action = agent.act(game_state_mock)
        assert action.action_type == ActionType.CHECK
    
    def test_act_prefer_call(self):
        """Test that the agent prefers to call when check is not available."""
        agent = CallAgent(0)
        
        # Create a mock game state where call is available but check is not
        call_action = Action(ActionType.CALL)
        legal_actions = [Action(ActionType.FOLD), call_action, Action(ActionType.RAISE, 4)]
        
        game_state_mock = type('MockGameState', (), {'get_legal_actions': lambda: legal_actions})
        
        action = agent.act(game_state_mock)
        assert action.action_type == ActionType.CALL
    
    def test_act_fallback(self):
        """Test that the agent falls back to first action when neither check nor call is available."""
        agent = CallAgent(0)
        
        # Create a mock game state where only fold and raise are available
        fold_action = Action(ActionType.FOLD)
        legal_actions = [fold_action, Action(ActionType.RAISE, 4)]
        
        game_state_mock = type('MockGameState', (), {'get_legal_actions': lambda: legal_actions})
        
        action = agent.act(game_state_mock)
        assert action.action_type == ActionType.FOLD


class TestRuleBased1Agent:
    """Tests for the RuleBased1Agent class."""
    
    def test_initialization(self):
        """Test agent initialization with default and custom aggression."""
        agent1 = RuleBased1Agent(1)
        assert agent1.player_id == 1
        assert agent1.aggression == 0.5  # Default
        
        agent2 = RuleBased1Agent(2, aggression=0.8)
        assert agent2.aggression == 0.8
        
        # Should clamp aggression to [0, 1]
        agent3 = RuleBased1Agent(3, aggression=1.5)
        assert agent3.aggression == 1.0
        
        agent4 = RuleBased1Agent(4, aggression=-0.5)
        assert agent4.aggression == 0.0
    
    def test_evaluate_preflop(self):
        """Test preflop hand evaluation."""
        agent = RuleBased1Agent(0)
        
        # Pocket aces should be strong
        pocket_aces = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS)
        ]
        
        aces_strength = agent._evaluate_preflop(pocket_aces)
        assert aces_strength > 0.9  # Should be very high
        
        # Low unsuited cards should be weak
        weak_hand = [
            Card(Rank.TWO, Suit.SPADES),
            Card(Rank.SEVEN, Suit.HEARTS)
        ]
        
        weak_strength = agent._evaluate_preflop(weak_hand)
        assert weak_strength < 0.3  # Should be quite low
        
        # Suited connectors should be medium
        suited_connectors = [
            Card(Rank.TEN, Suit.CLUBS),
            Card(Rank.NINE, Suit.CLUBS)
        ]
        
        connectors_strength = agent._evaluate_preflop(suited_connectors)
        assert 0.3 < connectors_strength < 0.7  # Should be medium
    
    def test_act_strong_hand(self):
        """Test behavior with a strong hand."""
        # Use a fixed seed for reproducibility
        random.seed(42)
        
        agent = RuleBased1Agent(0, aggression=0.7)
        game_state = GameState(num_players=2)
        
        # Mock a strong hand by overriding _evaluate_hand_strength
        real_evaluate = agent._evaluate_hand_strength
        agent._evaluate_hand_strength = lambda x: 0.9  # Very strong hand
        
        # Act multiple times
        actions = []
        for _ in range(10):
            action = agent.act(game_state)
            actions.append(action)
        
        # Restore original method
        agent._evaluate_hand_strength = real_evaluate
        
        # With a strong hand and high aggression, should often raise
        raise_count = sum(1 for a in actions if a.action_type in {ActionType.RAISE, ActionType.BET})
        assert raise_count >= 5  # Should raise at least half the time
    
    def test_act_weak_hand(self):
        """Test behavior with a weak hand."""
        # Use a fixed seed for reproducibility
        random.seed(42)
        
        agent = RuleBased1Agent(0, aggression=0.3)
        game_state = GameState(num_players=2)
        
        # Mock a weak hand by overriding _evaluate_hand_strength
        real_evaluate = agent._evaluate_hand_strength
        agent._evaluate_hand_strength = lambda x: 0.1  # Very weak hand
        
        # Act multiple times
        actions = []
        for _ in range(10):
            action = agent.act(game_state)
            actions.append(action)
        
        # Restore original method
        agent._evaluate_hand_strength = real_evaluate
        
        # With a weak hand, should often fold or check
        fold_check_count = sum(1 for a in actions if a.action_type in {ActionType.FOLD, ActionType.CHECK})
        assert fold_check_count >= 7  # Should fold/check most of the time


class TestRuleBased2Agent:
    """Tests for the RuleBased2Agent class."""
    
    def test_initialization(self):
        """Test agent initialization with default and custom parameters."""
        agent1 = RuleBased2Agent(1)
        assert agent1.player_id == 1
        assert agent1.aggression == 0.5  # Default
        assert agent1.bluff_frequency == 0.1  # Default
        
        agent2 = RuleBased2Agent(2, aggression=0.8, bluff_frequency=0.2)
        assert agent2.aggression == 0.8
        assert agent2.bluff_frequency == 0.2
        
        # Should clamp values to [0, 1]
        agent3 = RuleBased2Agent(3, aggression=1.5, bluff_frequency=1.5)
        assert agent3.aggression == 1.0
        assert agent3.bluff_frequency == 1.0
    
    def test_calculate_pot_odds(self):
        """Test pot odds calculation."""
        agent = RuleBased2Agent(0)
        
        # Create a mock game state with a specific pot and bet structure
        mock_game_state = type('MockGameState', (), {
            'pot': 10,
            'current_bets': [0, 5, 0],
            'player_id': 0
        })
        
        # Pot odds for player 0 should be 5 / (10 + 5 + 5) = 5/20 = 0.25
        pot_odds = agent._calculate_pot_odds(mock_game_state)
        assert 0.24 <= pot_odds <= 0.26
        
        # If no need to call, pot odds should be 0
        mock_game_state.current_bets = [5, 5, 0]
        pot_odds = agent._calculate_pot_odds(mock_game_state)
        assert pot_odds == 0.0
    
    def test_evaluate_position(self):
        """Test position evaluation."""
        agent = RuleBased2Agent(0)
        
        # Button position should be best
        mock_game_state = type('MockGameState', (), {
            'button': 0,
            'num_players': 3,
            'active': [True, True, True]
        })
        
        position_value = agent._evaluate_position(mock_game_state)
        assert position_value == 0.0  # Button is position 0
        
        # Last to act is best
        agent.player_id = 2
        position_value = agent._evaluate_position(mock_game_state)
        assert position_value == 1.0  # Last position
        
        # Middle position
        agent.player_id = 1
        position_value = agent._evaluate_position(mock_game_state)
        assert position_value == 0.5  # Middle of 3 players


class TestMLAgent:
    """Tests for the MLAgent class."""
    
    def test_random_ml_agent(self):
        """Test the RandomMLAgent implementation."""
        agent = RandomMLAgent(0)
        
        game_state = GameState(num_players=2)
        
        # Set seed for reproducibility
        random.seed(42)
        
        # Act multiple times
        actions = []
        for _ in range(10):
            action = agent.act(game_state)
            actions.append(action)
            
            # Action should be in legal actions
            assert action in game_state.get_legal_actions()
        
        # Should have some variety in actions due to randomness
        action_types = set(action.action_type for action in actions)
        assert len(action_types) > 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTorchAgent:
    """Tests for the TorchMLAgent class (requires PyTorch)."""
    
    def test_state_to_tensor(self):
        """Test converting game state to tensor."""
        # Skip if no CUDA
        if not torch.cuda.is_available():
            return
        
        # Create a simple model
        input_dim = 100  # Simplified
        hidden_dims = [64, 32]
        action_dim = 5
        
        model = PokerMLP(input_dim, hidden_dims, action_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        agent = TorchMLAgent(0, model, optimizer, device, epsilon=0.1)
        
        # Create a mock game state with a feature_vector method
        mock_feature_vector = np.random.random(input_dim).astype(np.float32)
        mock_game_state = type('MockGameState', (), {
            'to_feature_vector': lambda player_id: mock_feature_vector
        })
        
        # Convert to tensor
        tensor = agent.state_to_tensor(mock_game_state, 0)
        
        # Check tensor properties
        assert tensor.shape == torch.Size([input_dim])
        assert tensor.dtype == torch.float32
        assert tensor.device.type == device.type
