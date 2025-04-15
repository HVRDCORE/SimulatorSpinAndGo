"""
Example script demonstrating training of AI agents in Spin and Go format.

This script shows how to train advanced poker agents using reinforcement learning
in the Spin and Go tournament format. It demonstrates the integration of the
tournament structure with learning algorithms like Deep CFR and PPO.
"""

import random
import numpy as np
import time
import argparse
import torch
from typing import List, Dict, Any, Tuple

from pokersim.game.spingo import SpinGoGame
from pokersim.agents.base_agent import Agent
from pokersim.agents.random_agent import RandomAgent
from pokersim.agents.rule_based_agent import RuleBased1Agent
from pokersim.agents.advanced_opponent_agent import AdvancedOpponentAgent, AdvancedOpponentProfile
from pokersim.ml.models import create_value_network, create_policy_network
from pokersim.algorithms.deep_cfr import DeepCFR
from pokersim.algorithms.ppo import PPO


def print_separator(char: str = "-", length: int = 80) -> None:
    """Print a separator line."""
    print(char * length)


def simulate_tournaments(num_tournaments: int = 100, 
                        buy_in: int = 10, 
                        num_players: int = 3,
                        agent_types: List[str] = None,
                        verbose: bool = True) -> Dict:
    """
    Simulate multiple Spin and Go tournaments.
    
    Args:
        num_tournaments (int, optional): Number of tournaments to simulate. Defaults to 100.
        buy_in (int, optional): Buy-in amount. Defaults to 10.
        num_players (int, optional): Number of players per tournament. Defaults to 3.
        agent_types (List[str], optional): Types of agents to use. Defaults to None (rule-based).
        verbose (bool, optional): Whether to print details. Defaults to True.
        
    Returns:
        Dict: Simulation results.
    """
    if agent_types is None:
        agent_types = ["rule_based"] * num_players
    elif not isinstance(agent_types, list):
        agent_types = ["rule_based"] * num_players
    
    # Statistics to track
    stats = {
        "tournaments_played": 0,
        "multiplier_distribution": {},
        "player_wins": [0] * num_players,
        "player_roi": [0.0] * num_players,
        "avg_tournament_length": 0,
        "total_hands_played": 0
    }
    
    total_hands = 0
    
    for t in range(num_tournaments):
        if verbose and t % 10 == 0:
            print(f"Tournament {t+1}/{num_tournaments}")
            
        # Create a new tournament
        tournament = SpinGoGame(buy_in=buy_in, num_players=num_players)
        
        # Track multiplier distribution
        mult = tournament.multiplier
        stats["multiplier_distribution"][mult] = stats["multiplier_distribution"].get(mult, 0) + 1
        
        # Create agents
        agents = []
        for i in range(num_players):
            agent_type = agent_types[i] if i < len(agent_types) else "rule_based"
            
            if agent_type == "random":
                agents.append(RandomAgent(i))
            elif agent_type == "advanced":
                profile = AdvancedOpponentProfile(
                    aggression=random.uniform(0.3, 0.8),
                    bluff_frequency=random.uniform(0.1, 0.4),
                    call_threshold=random.uniform(0.3, 0.6),
                    raise_threshold=random.uniform(0.5, 0.8)
                )
                agents.append(AdvancedOpponentAgent(i, profile=profile))
            else:  # rule_based
                agents.append(RuleBased1Agent(i))
        
        # Simulate the tournament
        tournament_hands = 0
        
        while not tournament.is_tournament_over():
            # Start a new hand
            game_state = tournament.start_new_hand()
            tournament_hands += 1
            
            # Play the hand
            while not game_state.is_terminal():
                current_player = game_state.current_player
                if current_player < len(agents):
                    action = agents[current_player].act(game_state)
                    game_state = game_state.apply_action(action)
            
            # Update player stacks after the hand
            tournament.update_stacks_after_hand()
        
        # Record results
        winner = tournament.get_winner()
        if winner is not None:  # Make sure there is a winner
            prize = tournament.get_prize(winner)
            stats["player_wins"][winner] += 1
            stats["player_roi"][winner] += (prize / buy_in) - 1
        
        stats["tournaments_played"] += 1
        stats["total_hands_played"] += tournament_hands
        
    # Calculate averages
    for i in range(num_players):
        stats["player_roi"][i] /= stats["tournaments_played"]
    
    stats["avg_tournament_length"] = stats["total_hands_played"] / stats["tournaments_played"]
    
    return stats


def train_deep_cfr_agent(num_iterations: int = 1000, 
                         evaluation_freq: int = 100,
                         num_eval_tournaments: int = 50,
                         memory_size: int = 10000,
                         batch_size: int = 128) -> Dict:
    """
    Train an agent using Deep CFR algorithm in Spin and Go format.
    
    Args:
        num_iterations (int, optional): Number of training iterations. Defaults to 1000.
        evaluation_freq (int, optional): Frequency of evaluation. Defaults to 100.
        num_eval_tournaments (int, optional): Number of evaluation tournaments. Defaults to 50.
        memory_size (int, optional): Size of the replay memory. Defaults to 10000.
        batch_size (int, optional): Batch size for training. Defaults to 128.
        
    Returns:
        Dict: Training results.
    """
    print_separator()
    print("Training Deep CFR Agent for Spin and Go")
    print_separator()
    
    # Create a value network for the agent
    value_network = create_value_network(
        input_size=128,  # State representation size
        hidden_layers=[256, 256],
        output_size=1
    )
    
    # Initialize Deep CFR algorithm
    input_dim = 128  # Dimensionality of state representation
    action_dim = 5   # Number of action types in poker (fold, check, call, bet, raise)
    
    deep_cfr = DeepCFR(
        value_network=value_network,
        input_dim=input_dim,
        action_dim=action_dim,
        game_type='spingo',
        num_players=3,
        memory_size=memory_size,
        batch_size=batch_size
    )
    
    # Training loop
    results = {
        "training_iterations": num_iterations,
        "evaluation_scores": [],
        "win_rates": []
    }
    
    for iteration in range(num_iterations):
        # Train for one iteration
        deep_cfr.train_iteration()
        
        # Evaluate periodically
        if (iteration + 1) % evaluation_freq == 0 or iteration == num_iterations - 1:
            print(f"Evaluating after iteration {iteration+1}/{num_iterations}")
            
            # Create agent list with our trained agent at position 0
            agents = ["deep_cfr", "rule_based", "rule_based"]
            
            # Run evaluation tournaments
            eval_stats = simulate_tournaments(
                num_tournaments=num_eval_tournaments,
                agent_types=agents,
                verbose=False
            )
            
            win_rate = eval_stats["player_wins"][0] / eval_stats["tournaments_played"]
            roi = eval_stats["player_roi"][0]
            
            print(f"Win Rate: {win_rate:.3f}, ROI: {roi:.3f}")
            
            results["evaluation_scores"].append((iteration + 1, roi))
            results["win_rates"].append((iteration + 1, win_rate))
    
    print_separator()
    print("Training complete!")
    print(f"Final Win Rate: {results['win_rates'][-1][1]:.3f}")
    print(f"Final ROI: {results['evaluation_scores'][-1][1]:.3f}")
    print_separator()
    
    return results


def train_ppo_agent(num_episodes: int = 10000,
                   evaluation_freq: int = 500,
                   num_eval_tournaments: int = 50) -> Dict:
    """
    Train an agent using PPO algorithm in Spin and Go format.
    
    Args:
        num_episodes (int, optional): Number of training episodes. Defaults to 10000.
        evaluation_freq (int, optional): Frequency of evaluation. Defaults to 500.
        num_eval_tournaments (int, optional): Number of evaluation tournaments. Defaults to 50.
        
    Returns:
        Dict: Training results.
    """
    print_separator()
    print("Training PPO Agent for Spin and Go")
    print_separator()
    
    # Create policy network for the agent
    policy_network = create_policy_network(
        input_size=128,  # State representation size
        hidden_layers=[256, 256],
        action_space_size=5  # Fold, Check, Call, Bet, Raise
    )
    
    # Create optimizer for the policy network
    optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.0001)
    
    # Initialize PPO algorithm
    ppo = PPO(
        model=policy_network,
        optimizer=optimizer,
        game_type='spingo',
        num_players=3,
        learning_rate=0.0001,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01
    )
    
    # Training loop
    results = {
        "training_episodes": num_episodes,
        "evaluation_scores": [],
        "win_rates": []
    }
    
    for episode in range(num_episodes):
        # Train for one episode
        ppo.train_episode()
        
        # Evaluate periodically
        if (episode + 1) % evaluation_freq == 0 or episode == num_episodes - 1:
            print(f"Evaluating after episode {episode+1}/{num_episodes}")
            
            # Create agent list with our trained agent at position 0
            agents = ["ppo", "rule_based", "rule_based"]
            
            # Run evaluation tournaments
            eval_stats = simulate_tournaments(
                num_tournaments=num_eval_tournaments,
                agent_types=agents,
                verbose=False
            )
            
            win_rate = eval_stats["player_wins"][0] / eval_stats["tournaments_played"]
            roi = eval_stats["player_roi"][0]
            
            print(f"Win Rate: {win_rate:.3f}, ROI: {roi:.3f}")
            
            results["evaluation_scores"].append((episode + 1, roi))
            results["win_rates"].append((episode + 1, win_rate))
    
    print_separator()
    print("Training complete!")
    print(f"Final Win Rate: {results['win_rates'][-1][1]:.3f}")
    print(f"Final ROI: {results['evaluation_scores'][-1][1]:.3f}")
    print_separator()
    
    return results


def main():
    """Main function for Spin and Go training demonstration."""
    parser = argparse.ArgumentParser(description="Spin and Go Training Example")
    parser.add_argument("--mode", type=str, default="simulate", 
                        choices=["simulate", "deep_cfr", "ppo"],
                        help="Mode of operation")
    parser.add_argument("--num_tournaments", type=int, default=10,
                        help="Number of tournaments to simulate")
    parser.add_argument("--buy_in", type=int, default=10,
                        help="Buy-in amount")
    parser.add_argument("--num_players", type=int, default=3,
                        help="Number of players")
    parser.add_argument("--num_iterations", type=int, default=100,
                        help="Number of training iterations (for deep_cfr)")
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="Number of training episodes (for ppo)")
    args = parser.parse_args()
    
    if args.mode == "simulate":
        print_separator()
        print(f"Simulating {args.num_tournaments} Spin and Go tournaments")
        print_separator()
        
        # Use a mix of agent types
        agent_types = ["rule_based", "advanced", "random"]
        
        stats = simulate_tournaments(
            num_tournaments=args.num_tournaments,
            buy_in=args.buy_in,
            num_players=args.num_players,
            agent_types=agent_types,
            verbose=True
        )
        
        print_separator()
        print("Simulation Results:")
        print(f"Tournaments played: {stats['tournaments_played']}")
        print(f"Average tournament length: {stats['avg_tournament_length']:.2f} hands")
        print(f"Total hands played: {stats['total_hands_played']}")
        print("\nMultiplier distribution:")
        for mult, count in sorted(stats["multiplier_distribution"].items()):
            print(f"  {mult}x: {count} ({count/args.num_tournaments*100:.1f}%)")
        print("\nPlayer performance:")
        for i in range(args.num_players):
            win_rate = stats["player_wins"][i] / stats["tournaments_played"]
            roi = stats["player_roi"][i]
            agent_type = agent_types[i] if i < len(agent_types) else "rule_based"
            print(f"  Player {i} ({agent_type}): Win rate {win_rate:.3f}, ROI {roi:.3f}")
    
    elif args.mode == "deep_cfr":
        train_deep_cfr_agent(
            num_iterations=args.num_iterations,
            evaluation_freq=max(1, args.num_iterations // 10),
            num_eval_tournaments=20
        )
    
    elif args.mode == "ppo":
        train_ppo_agent(
            num_episodes=args.num_episodes,
            evaluation_freq=max(1, args.num_episodes // 10),
            num_eval_tournaments=20
        )


if __name__ == "__main__":
    main()