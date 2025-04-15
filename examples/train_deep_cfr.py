"""
Example script demonstrating training an agent with Deep CFR.

This script trains a poker agent using Deep Counterfactual Regret Minimization
and evaluates its performance against baseline agents.
"""

import sys
import os
import random
import time
import argparse
import numpy as np
import torch
from typing import List, Dict, Tuple

# Add the parent directory to sys.path to allow imports from the pokersim package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pokersim.game.state import GameState, Action, ActionType, Stage
from pokersim.game.evaluator import HandEvaluator
from pokersim.agents.random_agent import RandomAgent
from pokersim.agents.call_agent import CallAgent
from pokersim.agents.rule_based_agent import RuleBased1Agent, RuleBased2Agent
from pokersim.algorithms.deep_cfr import DeepCFR
from pokersim.ml.advanced_agents import DeepCFRAgent


def evaluate_agent(agent, opponents, num_games: int = 100, num_players: int = 2, verbose: bool = False) -> Dict:
    """
    Evaluate an agent's performance against opponents.
    
    Args:
        agent: The agent to evaluate.
        opponents: List of opponent agents.
        num_games (int, optional): Number of games to play. Defaults to 100.
        num_players (int, optional): Number of players per game. Defaults to 2.
        verbose (bool, optional): Whether to print detailed results. Defaults to False.
        
    Returns:
        Dict: Evaluation metrics including win rate and profit.
    """
    total_profit = 0
    wins = 0
    num_hands_played = 0
    
    for game in range(num_games):
        if verbose and game % 10 == 0:
            print(f"Evaluating game {game}/{num_games}...")
        
        # Initialize game state
        game_state = GameState(num_players=num_players, small_blind=1, big_blind=2)
        
        # Randomly assign positions
        player_positions = list(range(num_players))
        random.shuffle(player_positions)
        
        agent_position = player_positions[0]
        opponent_positions = player_positions[1:]
        
        # Set player_id for the agent and opponents
        agent.player_id = agent_position
        for i, opp in enumerate(opponents):
            if i < len(opponent_positions):
                opp.player_id = opponent_positions[i]
        
        # Play the hand
        while not game_state.is_terminal():
            current_player = game_state.current_player
            
            # Get action from the current player
            if current_player == agent_position:
                action = agent.act(game_state)
            else:
                # Find the opponent with this position
                for opp in opponents:
                    if opp.player_id == current_player:
                        action = opp.act(game_state)
                        break
                else:
                    # If no opponent found, use random action
                    temp_agent = RandomAgent(current_player)
                    action = temp_agent.act(game_state)
            
            # Apply action
            game_state = game_state.apply_action(action)
        
        # Record results
        num_hands_played += 1
        profit = game_state.get_payouts()[agent_position]
        total_profit += profit
        
        if profit > 0:
            wins += 1
        
        # Reset agent states
        agent.reset()
        for opp in opponents:
            opp.reset()
    
    # Calculate metrics
    win_rate = wins / num_hands_played if num_hands_played > 0 else 0
    avg_profit = total_profit / num_hands_played if num_hands_played > 0 else 0
    
    if verbose:
        print(f"Evaluation complete: Win rate: {win_rate:.2f}, Avg profit: {avg_profit:.2f}")
    
    return {
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'total_profit': total_profit,
        'hands_played': num_hands_played
    }


def train_deep_cfr(args):
    """
    Train an agent using Deep CFR algorithm.
    
    Args:
        args: Command line arguments.
    """
    print("Starting Deep CFR training...")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Set up training parameters
    input_dim = 2*52 + 5*52 + 1 + args.num_players*3 + args.num_players  # Based on feature vector size
    action_dim = 5  # FOLD, CHECK, CALL, BET, RAISE
    
    # Create the agent
    agent = DeepCFRAgent(player_id=0, input_dim=input_dim, action_dim=action_dim, device=device)
    
    # Create opponents for training
    opponents = [
        RandomAgent(1), 
        CallAgent(2),
        RuleBased1Agent(3, aggression=0.5),
    ]
    
    # Limit opponents to number of players - 1
    opponents = opponents[:args.num_players - 1]
    
    # Create opponents for evaluation (different set)
    eval_opponents = [
        RuleBased2Agent(1, aggression=0.6, bluff_frequency=0.1),
        RuleBased1Agent(2, aggression=0.4)
    ]
    
    # Limit eval opponents to number of players - 1
    eval_opponents = eval_opponents[:args.num_players - 1]
    
    # Training loop
    best_win_rate = 0.0
    training_metrics = {
        'advantage_loss': [],
        'strategy_loss': [],
        'win_rate': [],
        'avg_profit': []
    }
    
    print(f"Starting training for {args.iterations} iterations...")
    
    for iteration in range(args.iterations):
        print(f"\nIteration {iteration + 1}/{args.iterations}")
        
        # Initialize game state for collecting trajectories
        game_state = GameState(num_players=args.num_players)
        
        # Collect trajectories and train
        print("Collecting trajectories and training...")
        metrics = agent.train(game_state, 
                              num_trajectories=args.trajectories_per_iter, 
                              batch_size=args.batch_size, 
                              epochs=args.epochs)
        
        print(f"Training metrics - Advantage loss: {metrics['advantage_loss']:.4f}, "
              f"Strategy loss: {metrics['strategy_loss']:.4f}")
        
        # Record training metrics
        training_metrics['advantage_loss'].append(metrics['advantage_loss'])
        training_metrics['strategy_loss'].append(metrics['strategy_loss'])
        
        # Evaluate periodically
        if (iteration + 1) % args.eval_interval == 0:
            print("Evaluating agent...")
            eval_metrics = evaluate_agent(agent, eval_opponents, num_games=args.eval_games, 
                                         num_players=args.num_players, verbose=True)
            
            training_metrics['win_rate'].append(eval_metrics['win_rate'])
            training_metrics['avg_profit'].append(eval_metrics['avg_profit'])
            
            print(f"Evaluation results - Win rate: {eval_metrics['win_rate']:.2f}, "
                  f"Avg profit: {eval_metrics['avg_profit']:.2f}")
            
            # Save best model
            if eval_metrics['win_rate'] > best_win_rate:
                best_win_rate = eval_metrics['win_rate']
                if args.save_model:
                    save_path = args.model_path
                    print(f"Saving best model to {save_path}")
                    agent.save(save_path)
    
    print("\nTraining complete!")
    
    # Final evaluation
    print("\nPerforming final evaluation...")
    final_metrics = evaluate_agent(agent, eval_opponents, num_games=args.eval_games*2, 
                                  num_players=args.num_players, verbose=True)
    
    print("\nFinal evaluation results:")
    print(f"Win rate: {final_metrics['win_rate']:.4f}")
    print(f"Average profit: {final_metrics['avg_profit']:.4f}")
    print(f"Total games played: {final_metrics['hands_played']}")
    
    return agent, training_metrics, final_metrics


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(description="Train a poker agent using Deep CFR")
    
    # Training parameters
    parser.add_argument("--iterations", type=int, default=20, 
                        help="Number of training iterations")
    parser.add_argument("--trajectories-per-iter", type=int, default=100, 
                        help="Number of trajectories to collect per iteration")
    parser.add_argument("--batch-size", type=int, default=32, 
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, 
                        help="Number of epochs per training iteration")
    
    # Game parameters
    parser.add_argument("--num-players", type=int, default=3, 
                        help="Number of players in the game")
    
    # Evaluation parameters
    parser.add_argument("--eval-interval", type=int, default=5, 
                        help="Evaluate agent every N iterations")
    parser.add_argument("--eval-games", type=int, default=50, 
                        help="Number of games to play during evaluation")
    
    # Model parameters
    parser.add_argument("--save-model", action="store_true", 
                        help="Save the best model during training")
    parser.add_argument("--model-path", type=str, default="models/deep_cfr_agent.pt", 
                        help="Path to save the model")
    parser.add_argument("--cpu", action="store_true", 
                        help="Force using CPU even if CUDA is available")
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    if args.save_model:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Train the agent
    agent, training_metrics, final_metrics = train_deep_cfr(args)
    
    print("\nDeep CFR training complete!")
    print("For more examples and details, check the README.md and docs/ directory.")


if __name__ == "__main__":
    main()
