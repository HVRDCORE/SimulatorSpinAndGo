"""
Example script demonstrating saving and loading trained models.

This script shows how to save and load trained models and agents, which is useful
for continuing training from a previously saved state or for evaluating trained agents.
"""

import os
import argparse
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from pokersim.ml.models import create_value_network, create_policy_network
from pokersim.ml.model_io import save_model, load_model, save_agent, load_agent
from pokersim.algorithms.deep_cfr import DeepCFR
from pokersim.algorithms.ppo import PPO
from pokersim.game.spingo import SpinGoGame
from examples.spin_and_go_training import simulate_tournaments, print_separator


def train_and_save_deep_cfr(num_iterations: int = 20, 
                          save_dir: str = "./saved_models") -> None:
    """
    Train a Deep CFR agent and save it to disk.
    
    Args:
        num_iterations (int, optional): Number of training iterations. Defaults to 20.
        save_dir (str, optional): Directory to save models in. Defaults to "./saved_models".
    """
    print_separator()
    print("Training and Saving Deep CFR Agent")
    print_separator()
    
    # Create value network
    input_dim = 128
    action_dim = 5
    value_network = create_value_network(
        input_size=input_dim,
        hidden_layers=[256, 256],
        output_size=action_dim
    )
    
    # Create DeepCFR agent
    deep_cfr = DeepCFR(
        value_network=value_network,
        input_dim=input_dim,
        action_dim=action_dim,
        game_type='spingo',
        num_players=3,
        memory_size=1000,
        batch_size=32
    )
    
    # Train for a few iterations
    for i in range(num_iterations):
        print(f"Training iteration {i+1}/{num_iterations}")
        deep_cfr.train_iteration()
    
    # Save the agent
    os.makedirs(save_dir, exist_ok=True)
    agent_path = os.path.join(save_dir, "deep_cfr_agent")
    save_agent(deep_cfr, agent_path, "deep_cfr", extras={
        "training_iterations": num_iterations,
        "description": "DeepCFR agent trained for Spin and Go"
    })
    
    print(f"Agent saved to {agent_path}")
    
    # Also save just the model separately
    model_path = os.path.join(save_dir, "deep_cfr_value_network.pt")
    save_model(value_network, model_path, {
        "training_iterations": num_iterations,
        "input_dim": input_dim,
        "action_dim": action_dim,
        "description": "Value network for DeepCFR in Spin and Go"
    })
    
    print(f"Model saved to {model_path}")


def train_and_save_ppo(num_episodes: int = 20, 
                     save_dir: str = "./saved_models") -> None:
    """
    Train a PPO agent and save it to disk.
    
    Args:
        num_episodes (int, optional): Number of training episodes. Defaults to 20.
        save_dir (str, optional): Directory to save models in. Defaults to "./saved_models".
    """
    print_separator()
    print("Training and Saving PPO Agent")
    print_separator()
    
    # Create policy network
    input_dim = 128
    action_dim = 5
    policy_network = create_policy_network(
        input_size=input_dim,
        hidden_layers=[256, 256],
        action_space_size=action_dim
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.0001)
    
    # Create PPO agent
    ppo = PPO(
        model=policy_network,
        optimizer=optimizer,
        game_type='spingo',
        num_players=3,
        learning_rate=0.0001,
        clip_ratio=0.2
    )
    
    # Train for a few episodes
    for i in range(num_episodes):
        print(f"Training episode {i+1}/{num_episodes}")
        ppo.train_episode()
    
    # Save the agent
    os.makedirs(save_dir, exist_ok=True)
    agent_path = os.path.join(save_dir, "ppo_agent")
    save_agent(ppo, agent_path, "ppo", extras={
        "training_episodes": num_episodes,
        "description": "PPO agent trained for Spin and Go"
    })
    
    print(f"Agent saved to {agent_path}")
    
    # Also save just the model separately
    model_path = os.path.join(save_dir, "ppo_policy_network.pt")
    save_model(policy_network, model_path, {
        "training_episodes": num_episodes,
        "input_dim": input_dim,
        "action_dim": action_dim,
        "description": "Policy network for PPO in Spin and Go"
    })
    
    print(f"Model saved to {model_path}")


def load_and_evaluate_agent(agent_path: str, agent_type: str, num_tournaments: int = 10):
    """
    Load a saved agent and evaluate it.
    
    Args:
        agent_path (str): Path to the saved agent.
        agent_type (str): Type of agent ('deep_cfr' or 'ppo').
        num_tournaments (int, optional): Number of tournaments to simulate. 
                                        Defaults to 10.
    """
    print_separator()
    print(f"Loading and Evaluating {agent_type.upper()} Agent")
    print_separator()
    
    # Load the agent
    agent = load_agent(agent_path, agent_type)
    print(f"Agent loaded from {agent_path}")
    
    # Get the training progress
    if agent_type == 'deep_cfr':
        progress = f"Training iterations: {agent.iteration}"
    else:  # ppo
        progress = f"Training episodes: {agent.episode}"
    
    print(f"Agent progress: {progress}")
    
    # Create agent list with our trained agent at position 0
    agents = [agent_type, "rule_based", "rule_based"]
    
    # Run evaluation tournaments
    print(f"Evaluating agent over {num_tournaments} tournaments...")
    eval_stats = simulate_tournaments(
        num_tournaments=num_tournaments,
        agent_types=agents,
        verbose=False
    )
    
    # Print results
    win_rate = eval_stats["player_wins"][0] / eval_stats["tournaments_played"]
    roi = eval_stats["player_roi"][0]
    
    print(f"Win Rate: {win_rate:.3f}, ROI: {roi:.3f}")
    print(f"Average tournament length: {eval_stats['avg_tournament_length']:.2f} hands")
    print(f"Total hands played: {eval_stats['total_hands_played']}")


def main():
    """Main function for model saving/loading example."""
    parser = argparse.ArgumentParser(description="Model Saving and Loading Example")
    parser.add_argument("--mode", type=str, default="train_and_save", 
                        choices=["train_and_save", "load_and_evaluate"],
                        help="Mode of operation")
    parser.add_argument("--algorithm", type=str, default="deep_cfr", 
                        choices=["deep_cfr", "ppo"],
                        help="Algorithm to use")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Number of training iterations/episodes")
    parser.add_argument("--save_dir", type=str, default="./saved_models",
                        help="Directory to save models in")
    parser.add_argument("--agent_path", type=str, default=None,
                        help="Path to saved agent for loading")
    parser.add_argument("--num_tournaments", type=int, default=10,
                        help="Number of tournaments for evaluation")
    args = parser.parse_args()
    
    if args.mode == "train_and_save":
        if args.algorithm == "deep_cfr":
            train_and_save_deep_cfr(
                num_iterations=args.iterations,
                save_dir=args.save_dir
            )
        else:  # ppo
            train_and_save_ppo(
                num_episodes=args.iterations,
                save_dir=args.save_dir
            )
    
    elif args.mode == "load_and_evaluate":
        if args.agent_path is None:
            # Use default path if none provided
            args.agent_path = os.path.join(args.save_dir, f"{args.algorithm}_agent")
        
        load_and_evaluate_agent(
            agent_path=args.agent_path,
            agent_type=args.algorithm,
            num_tournaments=args.num_tournaments
        )


if __name__ == "__main__":
    main()