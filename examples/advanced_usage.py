"""
Advanced usage examples for the PokerSim framework.

This script demonstrates advanced features of the PokerSim framework including:
- Advanced opponent modeling
- Multiple game variants
- Performance optimization with Numba
- Custom hand evaluation
- Monte Carlo simulations
- Integration with PyTorch and TensorFlow
"""

import sys
import os
import argparse
import random
import time
import numpy as np
import torch
import tensorflow as tf
from typing import List, Dict, Tuple, Any, Optional

# Add the parent directory to sys.path to allow imports from the pokersim package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pokersim.game.state import GameState, Action, ActionType, Stage
from pokersim.game.card import Card, Suit, Rank
from pokersim.game.deck import Deck
from pokersim.game.evaluator import HandEvaluator
from pokersim.agents.base_agent import Agent
from pokersim.agents.random_agent import RandomAgent
from pokersim.agents.call_agent import CallAgent
from pokersim.agents.rule_based_agent import RuleBased1Agent, RuleBased2Agent
from pokersim.ml.advanced_agents import (
    DeepCFRAgent, PPOAgent, ImitationLearningAgent, HybridAgent
)
from pokersim.ml.models import PokerMLP, PokerCNN, PokerActorCritic
from pokersim.utils.numba_optimizations import (
    fast_evaluate_hand_strength, fast_monte_carlo_simulation, 
    optimize_bet_sizing, optimize_action_sequence
)
from pokersim.utils.logging import GameLogger, TrainingLogger


def example_opponent_modeling():
    """Example demonstrating opponent modeling techniques."""
    print("\n=== Example: Opponent Modeling ===")
    
    # Create a game with 6 players
    game_state = GameState(num_players=6, small_blind=1, big_blind=2)
    
    # Create a hybrid agent that uses both heuristics and neural networks
    input_dim = 2*52 + 5*52 + 1 + 6*3 + 6  # Feature vector size for a 6-player game
    hidden_dims = [128, 64, 32]
    action_dim = 5
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a hybrid agent (combines heuristics with neural networks)
    hybrid_agent = HybridAgent(
        player_id=0,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        action_dim=action_dim,
        device=device
    )
    
    # Create opponents with different strategies
    opponents = [
        RandomAgent(1),                            # Purely random
        CallAgent(2),                              # Always calls/checks
        RuleBased1Agent(3, aggression=0.3),        # Conservative
        RuleBased1Agent(4, aggression=0.8),        # Aggressive
        RuleBased2Agent(5, aggression=0.6, bluff_frequency=0.2)  # Balanced
    ]
    
    # Play a few hands to demonstrate opponent modeling
    for hand in range(3):
        print(f"\nPlaying hand {hand+1}...")
        
        # Reset game state
        game_state = GameState(num_players=6, small_blind=1, big_blind=2)
        
        # Play until the game is over
        while not game_state.is_terminal():
            current_player = game_state.current_player
            
            if current_player == -1:
                continue
            
            # Get the current player's agent
            if current_player == 0:
                agent = hybrid_agent
            else:
                agent = opponents[current_player-1]
            
            # Get the agent's action
            action = agent.act(game_state)
            
            # Print the action
            print(f"  Player {current_player} takes action: {action}")
            
            # Apply the action to the game state
            new_state = game_state.apply_action(action)
            
            # If hybrid agent, observe the new state to update models
            if current_player == 0:
                hybrid_agent.observe(new_state)
            
            # Update the game state
            game_state = new_state
        
        # Hand is over
        payouts = game_state.get_payouts()
        winners = [i for i, p in enumerate(payouts) if p > 0]
        print(f"  Hand {hand+1} winners: {winners}, payouts: {payouts}")
        
        # Final observation for the hybrid agent
        hybrid_agent.observe(game_state)
    
    print("\nOpponent modeling example complete")


def example_monte_carlo_simulation():
    """Example demonstrating Monte Carlo simulations for hand strength evaluation."""
    print("\n=== Example: Monte Carlo Simulation ===")
    
    # Create a deck and deal hole cards
    deck = Deck()
    hole_cards = deck.deal(2)
    
    print(f"Your hole cards: {hole_cards[0]} {hole_cards[1]}")
    
    # Deal the flop
    flop = deck.deal(3)
    print(f"Flop: {flop[0]} {flop[1]} {flop[2]}")
    
    # Convert cards to integer representations for fast simulation
    hole_values = [card.to_int() for card in hole_cards]
    flop_values = [card.to_int() for card in flop]
    
    # Run Monte Carlo simulations with different numbers of opponents
    print("\nEstimating win probabilities:")
    
    for num_opponents in [1, 2, 3, 5, 8]:
        # Time the simulation
        start_time = time.time()
        
        # Run 10,000 simulations
        win_prob = fast_monte_carlo_simulation(
            hole_values=hole_values,
            community_values=flop_values,
            num_opponents=num_opponents,
            num_simulations=10000
        )
        
        elapsed = time.time() - start_time
        
        print(f"  Against {num_opponents} opponents: {win_prob:.4f} win probability (calculated in {elapsed:.3f} seconds)")
    
    # Calculate optimal bet sizing based on win probability
    pot_size = 10  # Example pot size
    stack_size = 100  # Example stack size
    
    # Calculate win probability against 2 opponents
    win_prob = fast_monte_carlo_simulation(
        hole_values=hole_values,
        community_values=flop_values,
        num_opponents=2,
        num_simulations=10000
    )
    
    # Calculate optimal bet sizes with different risk factors
    print("\nOptimal bet sizing:")
    
    for risk_factor in [0.5, 1.0, 2.0]:
        bet_size = optimize_bet_sizing(
            pot_size=pot_size,
            stack_size=stack_size,
            win_probability=win_prob,
            risk_factor=risk_factor
        )
        
        print(f"  Risk factor {risk_factor}: Optimal bet = {bet_size} (pot = {pot_size}, win prob = {win_prob:.4f})")
    
    print("\nMonte Carlo simulation example complete")


def example_advanced_agents():
    """Example demonstrating the use of advanced agents."""
    print("\n=== Example: Advanced Agents ===")
    
    # Create a game with 4 players
    game_state = GameState(num_players=4, small_blind=1, big_blind=2)
    
    # Feature vector size for a 4-player game
    input_dim = 2*52 + 5*52 + 1 + 4*3 + 4
    action_dim = 5
    
    # Create a simple model for imitation learning
    hidden_dims = [64, 32]
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create an expert agent to imitate
    expert = RuleBased2Agent(0, aggression=0.7, bluff_frequency=0.1)
    
    # Create an imitation learning agent
    imitation_agent = ImitationLearningAgent(
        player_id=0,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        action_dim=action_dim,
        expert=expert,
        device=device
    )
    
    # Create other agents for the game
    ppo_agent = PPOAgent(
        player_id=1,
        input_dim=input_dim,
        action_dim=action_dim,
        device=device
    )
    
    deep_cfr_agent = DeepCFRAgent(
        player_id=2,
        input_dim=input_dim,
        action_dim=action_dim,
        device=device
    )
    
    # Create a random agent for player 3
    random_agent = RandomAgent(3)
    
    # Create a list of all agents
    agents = [imitation_agent, ppo_agent, deep_cfr_agent, random_agent]
    
    # Create a game logger
    game_logger = GameLogger()
    
    # Play a few hands
    for hand in range(2):
        print(f"\nPlaying hand {hand+1}...")
        
        # Reset game state
        game_state = GameState(num_players=4, small_blind=1, big_blind=2)
        
        # Log initial game state
        game_logger.log_game_state(game_state, f"Starting hand {hand+1}")
        
        # Play until the game is over
        while not game_state.is_terminal():
            current_player = game_state.current_player
            
            if current_player == -1:
                continue
            
            # Get the current player's agent
            agent = agents[current_player]
            
            # Get the agent's action
            action = agent.act(game_state)
            
            # Log the action
            game_logger.log_action(current_player, action)
            
            # Apply the action to the game state
            new_state = game_state.apply_action(action)
            
            # Observe the transition (for learning agents)
            if isinstance(agent, (PPOAgent, ImitationLearningAgent)):
                agent.observe(new_state)
            
            # Update the game state
            game_state = new_state
        
        # Hand is over
        payouts = game_state.get_payouts()
        winners = [i for i, p in enumerate(payouts) if p > 0]
        print(f"  Hand {hand+1} winners: {winners}, payouts: {payouts}")
        
        # Log game outcome
        game_logger.log_game_outcome(game_state)
        
        # Final observation for learning agents
        for agent in agents:
            if isinstance(agent, (PPOAgent, ImitationLearningAgent)):
                agent.observe(game_state)
    
    print("\nAdvanced agents example complete")


def example_optimization_techniques():
    """Example demonstrating optimization techniques."""
    print("\n=== Example: Optimization Techniques ===")
    
    # Create a deck and deal some cards
    deck = Deck()
    hole_cards = deck.deal(2)
    flop = deck.deal(3)
    turn = deck.deal(1)
    river = deck.deal(1)
    
    print(f"Hole cards: {hole_cards[0]} {hole_cards[1]}")
    print(f"Community cards: {flop[0]} {flop[1]} {flop[2]} {turn[0]} {river[0]}")
    
    # Combine all cards
    all_cards = hole_cards + flop + turn + river
    
    # Standard hand evaluation (Python)
    start_time = time.time()
    rank, description = HandEvaluator.evaluate_hand(all_cards)
    std_time = time.time() - start_time
    
    print(f"\nStandard evaluation: {description}")
    print(f"Time: {std_time:.6f} seconds")
    
    # Convert cards to integer values for fast evaluation
    card_values = [card.to_int() for card in all_cards]
    hole_values = [card.to_int() for card in hole_cards]
    community_values = [card.to_int() for card in flop + turn + river]
    
    # Fast hand strength evaluation (Numba)
    start_time = time.time()
    strength = fast_evaluate_hand_strength(
        hand_value=(hole_values[0] * 100 + hole_values[1]),
        community_values=community_values
    )
    fast_time = time.time() - start_time
    
    print(f"\nFast evaluation (hand strength): {strength:.4f}")
    print(f"Time: {fast_time:.6f} seconds")
    print(f"Speedup: {std_time / fast_time:.2f}x")
    
    # Optimize action sequence
    stack_sizes = [100, 90, 80, 70]
    pot_size = 30
    num_players = 4
    position = 0
    
    start_time = time.time()
    action_recommendation = optimize_action_sequence(
        hole_values=hole_values,
        community_values=community_values,
        stack_sizes=stack_sizes,
        pot_size=pot_size,
        num_players=num_players,
        position=position
    )
    
    action_types = ["fold", "check/call", "bet/raise"]
    action_str = action_types[action_recommendation]
    
    print(f"\nAction recommendation: {action_str}")
    print(f"Time: {time.time() - start_time:.6f} seconds")
    
    print("\nOptimization techniques example complete")


def example_ml_frameworks():
    """Example demonstrating integration with ML frameworks."""
    print("\n=== Example: ML Framework Integration ===")
    
    # Feature vector size for a 2-player game
    input_dim = 2*52 + 5*52 + 1 + 2*3 + 2
    action_dim = 5
    hidden_dims = [128, 64]
    
    print("\nPyTorch Models:")
    
    # Check if CUDA is available for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create PyTorch MLP model
    torch_mlp = PokerMLP(input_dim, hidden_dims, action_dim)
    print(f"  PokerMLP params: {sum(p.numel() for p in torch_mlp.parameters())}")
    
    # Create PyTorch CNN model
    torch_cnn = PokerCNN(input_dim, action_dim)
    print(f"  PokerCNN params: {sum(p.numel() for p in torch_cnn.parameters())}")
    
    # Create PyTorch Actor-Critic model
    torch_ac = PokerActorCritic(input_dim, action_dim)
    print(f"  PokerActorCritic params: {sum(p.numel() for p in torch_ac.parameters())}")
    
    print("\nTensorFlow Models:")
    
    # Check if TensorFlow is available
    try:
        # Import TensorFlow models
        from pokersim.ml.tf_integration import PokerMLP_TF, PokerCNN_TF, PokerActorCritic_TF
        
        # Create TensorFlow MLP model
        tf_mlp = PokerMLP_TF(input_dim, hidden_dims, action_dim)
        tf_mlp.build((None, input_dim))
        print(f"  PokerMLP_TF params: {tf_mlp.count_params()}")
        
        # Create TensorFlow CNN model
        tf_cnn = PokerCNN_TF(input_dim, action_dim)
        tf_cnn.build((None, input_dim))
        print(f"  PokerCNN_TF params: {tf_cnn.count_params()}")
        
        # Create TensorFlow Actor-Critic model
        tf_ac = PokerActorCritic_TF(input_dim, action_dim)
        tf_ac.build((None, input_dim))
        print(f"  PokerActorCritic_TF params: {tf_ac.count_params()}")
    except (ImportError, ModuleNotFoundError):
        print("  TensorFlow not available or not installed")
    
    print("\nML Framework Integration example complete")


def main():
    """Main function to run the advanced usage examples."""
    parser = argparse.ArgumentParser(description="PokerSim Advanced Usage Examples")
    parser.add_argument("--all", action="store_true", help="Run all examples")
    parser.add_argument("--opponent-modeling", action="store_true", help="Run opponent modeling example")
    parser.add_argument("--monte-carlo", action="store_true", help="Run Monte Carlo simulation example")
    parser.add_argument("--advanced-agents", action="store_true", help="Run advanced agents example")
    parser.add_argument("--optimization", action="store_true", help="Run optimization techniques example")
    parser.add_argument("--ml-frameworks", action="store_true", help="Run ML framework integration example")
    
    args = parser.parse_args()
    
    # If no specific examples are selected, run all
    if not any([args.opponent_modeling, args.monte_carlo, args.advanced_agents, 
               args.optimization, args.ml_frameworks, args.all]):
        args.all = True
    
    print("PokerSim Advanced Usage Examples")
    print("===============================")
    
    # Run selected examples
    if args.all or args.opponent_modeling:
        example_opponent_modeling()
    
    if args.all or args.monte_carlo:
        example_monte_carlo_simulation()
    
    if args.all or args.advanced_agents:
        example_advanced_agents()
    
    if args.all or args.optimization:
        example_optimization_techniques()
    
    if args.all or args.ml_frameworks:
        example_ml_frameworks()
    
    print("\nAll examples completed!")
    print("For more information, check the documentation and other example scripts.")


if __name__ == "__main__":
    main()
