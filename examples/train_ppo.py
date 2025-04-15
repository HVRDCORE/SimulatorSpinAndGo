"""
Example script demonstrating training an agent with PPO.

This script trains a poker agent using Proximal Policy Optimization
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
from pokersim.ml.models import PokerActorCritic
from pokersim.algorithms.ppo import PPO
from pokersim.ml.advanced_agents import PPOAgent


def play_episode(agent, opponents, num_players: int = 2, verbose: bool = False) -> Dict:
    """
    Play a single poker episode with the agent and opponents.
    
    Args:
        agent: The agent to play.
        opponents: List of opponent agents.
        num_players (int, optional): Number of players. Defaults to 2.
        verbose (bool, optional): Whether to print details. Defaults to False.
        
    Returns:
        Dict: Episode results including rewards and terminal state.
    """
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
            agent.observe(game_state)
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
    
    # Final observation for the agent
    agent.observe(game_state)
    
    # Record results
    reward = game_state.get_rewards()[agent_position]
    
    if verbose:
        print(f"Episode complete: Agent reward: {reward}")
    
    return {
        'reward': reward,
        'terminal_state': game_state
    }


def evaluate_agent(agent, opponents, num_episodes: int = 100, num_players: int = 2, verbose: bool = False) -> Dict:
    """
    Evaluate an agent's performance against opponents.
    
    Args:
        agent: The agent to evaluate.
        opponents: List of opponent agents.
        num_episodes (int, optional): Number of episodes to play. Defaults to 100.
        num_players (int, optional): Number of players per game. Defaults to 2.
        verbose (bool, optional): Whether to print detailed results. Defaults to False.
        
    Returns:
        Dict: Evaluation metrics including average reward and win rate.
    """
    total_reward = 0
    wins = 0
    
    for episode in range(num_episodes):
        if verbose and episode % 10 == 0:
            print(f"Evaluating episode {episode}/{num_episodes}...")
        
        # Play episode
        result = play_episode(agent, opponents, num_players, verbose=False)
        
        total_reward += result['reward']
        if result['reward'] > 0:
            wins += 1
    
    # Calculate metrics
    avg_reward = total_reward / num_episodes
    win_rate = wins / num_episodes
    
    if verbose:
        print(f"Evaluation complete: Win rate: {win_rate:.2f}, Avg reward: {avg_reward:.2f}")
    
    return {
        'avg_reward': avg_reward,
        'win_rate': win_rate,
        'total_reward': total_reward,
        'episodes_played': num_episodes
    }


def train_ppo(args):
    """
    Train an agent using PPO algorithm.
    
    Args:
        args: Command line arguments.
    """
    print("Starting PPO training...")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Set up training parameters
    input_dim = 2*52 + 5*52 + 1 + args.num_players*3 + args.num_players  # Based on feature vector size
    action_dim = 5  # FOLD, CHECK, CALL, BET, RAISE
    
    # Create the agent
    agent = PPOAgent(
        player_id=0, 
        input_dim=input_dim, 
        action_dim=action_dim,
        lr=args.learning_rate,
        device=device, 
        epsilon=args.epsilon
    )
    
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
        'episode_rewards': [],
        'win_rates': [],
        'avg_rewards': []
    }
    
    print(f"Starting training for {args.episodes} episodes...")
    
    for episode in range(args.episodes):
        if episode % 10 == 0:
            print(f"\nEpisode {episode + 1}/{args.episodes}")
        
        # Play episode
        result = play_episode(agent, opponents, num_players=args.num_players, verbose=False)
        
        # Record episode reward
        training_metrics['episode_rewards'].append(result['reward'])
        
        # Print progress periodically
        if (episode + 1) % 10 == 0:
            avg_reward = sum(training_metrics['episode_rewards'][-10:]) / 10
            print(f"Last 10 episodes average reward: {avg_reward:.2f}")
        
        # Evaluate periodically
        if (episode + 1) % args.eval_interval == 0:
            print("Evaluating agent...")
            eval_metrics = evaluate_agent(agent, eval_opponents, num_episodes=args.eval_episodes, 
                                         num_players=args.num_players, verbose=True)
            
            training_metrics['win_rates'].append(eval_metrics['win_rate'])
            training_metrics['avg_rewards'].append(eval_metrics['avg_reward'])
            
            print(f"Evaluation results - Win rate: {eval_metrics['win_rate']:.2f}, "
                  f"Avg reward: {eval_metrics['avg_reward']:.2f}")
            
            # Save best model
            if eval_metrics['win_rate'] > best_win_rate:
                best_win_rate = eval_metrics['win_rate']
                if args.save_model:
                    save_path = args.model_path
                    print(f"Saving best model to {save_path}")
                    agent.save(save_path)
        
        # Reduce exploration over time
        if episode > args.episodes // 2:
            agent.epsilon = max(0.01, agent.epsilon * 0.999)
    
    print("\nTraining complete!")
    
    # Final evaluation
    print("\nPerforming final evaluation...")
    final_metrics = evaluate_agent(agent, eval_opponents, num_episodes=args.eval_episodes*2, 
                                  num_players=args.num_players, verbose=True)
    
    print("\nFinal evaluation results:")
    print(f"Win rate: {final_metrics['win_rate']:.4f}")
    print(f"Average reward: {final_metrics['avg_reward']:.4f}")
    print(f"Total episodes played: {final_metrics['episodes_played']}")
    
    return agent, training_metrics, final_metrics


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(description="Train a poker agent using PPO")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=1000, 
                        help="Number of training episodes")
    parser.add_argument("--learning-rate", type=float, default=0.001, 
                        help="Learning rate")
    parser.add_argument("--epsilon", type=float, default=0.2, 
                        help="Initial exploration rate")
    
    # Game parameters
    parser.add_argument("--num-players", type=int, default=3, 
                        help="Number of players in the game")
    
    # Evaluation parameters
    parser.add_argument("--eval-interval", type=int, default=100, 
                        help="Evaluate agent every N episodes")
    parser.add_argument("--eval-episodes", type=int, default=50, 
                        help="Number of episodes to play during evaluation")
    
    # Model parameters
    parser.add_argument("--save-model", action="store_true", 
                        help="Save the best model during training")
    parser.add_argument("--model-path", type=str, default="models/ppo_agent.pt", 
                        help="Path to save the model")
    parser.add_argument("--cpu", action="store_true", 
                        help="Force using CPU even if CUDA is available")
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    if args.save_model:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Train the agent
    agent, training_metrics, final_metrics = train_ppo(args)
    
    print("\nPPO training complete!")
    print("For more examples and details, check the README.md and docs/ directory.")


if __name__ == "__main__":
    main()
