"""
Example script for training and evaluating MCTS for poker.

This script demonstrates how to use Monte Carlo Tree Search (MCTS) for poker
decision making and shows its performance against various opponents.
"""

import sys
import os
import time
import argparse
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional, Callable

# Add the parent directory to sys.path to allow imports from the pokersim package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pokersim.game.state import GameState, Action, ActionType
from pokersim.game.spingo import SpinGoGame
from pokersim.algorithms.mcts import MCTSSolver
from pokersim.agents.random_agent import RandomAgent
from pokersim.agents.call_agent import CallAgent
from pokersim.agents.rule_based_agent import RuleBased1Agent, RuleBased2Agent
from pokersim.logging.game_logger import get_logger
from pokersim.logging.data_exporter import get_exporter
from pokersim.config.config_manager import get_config


def setup_environment(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set up the environment for MCTS training and evaluation.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        Dict[str, Any]: The environment components.
    """
    # Get configuration
    config = get_config()
    
    # Update config based on command-line arguments
    if args.get('iterations'):
        config.set("mcts.iterations", args['iterations'])
    if args.get('time_limit'):
        config.set("mcts.time_limit", args['time_limit'])
    if args.get('exploration'):
        config.set("mcts.exploration_weight", args['exploration'])
    
    # Set up logging
    logger = get_logger()
    data_exporter = get_exporter()
    
    # Print configuration
    print(f"MCTS configuration:")
    print(f"  - Max iterations: {config.get('mcts.iterations', 1000)}")
    print(f"  - Time limit: {config.get('mcts.time_limit', 5.0)} seconds")
    print(f"  - Exploration weight: {config.get('mcts.exploration_weight', 1.0)}")
    print(f"  - Game variant: {args.get('game_variant', 'holdem')}")
    print(f"  - Number of players: {args.get('num_players', 2)}")
    
    # Return the environment
    return {
        "config": config,
        "logger": logger,
        "data_exporter": data_exporter,
        "game_variant": args.get('game_variant', 'holdem'),
        "num_players": args.get('num_players', 2)
    }


def create_game_state(game_variant: str = 'holdem', num_players: int = 2) -> Any:
    """
    Create a game state for the specified variant.
    
    Args:
        game_variant (str, optional): Game variant. Defaults to 'holdem'.
        num_players (int, optional): Number of players. Defaults to 2.
        
    Returns:
        Any: Game state.
    """
    if game_variant == 'spin_and_go':
        return SpinGoGame(num_players=num_players)
    else:
        return GameState(num_players=num_players)


def create_opponent_agent(agent_type: str, player_id: int) -> Any:
    """
    Create an opponent agent of the specified type.
    
    Args:
        agent_type (str): Agent type.
        player_id (int): Player ID.
        
    Returns:
        Any: Agent instance.
    """
    if agent_type == 'random':
        return RandomAgent(player_id)
    elif agent_type == 'call':
        return CallAgent(player_id)
    elif agent_type == 'rule_based_1':
        return RuleBased1Agent(player_id)
    elif agent_type == 'rule_based_2':
        return RuleBased2Agent(player_id)
    else:
        # Default to random agent
        return RandomAgent(player_id)


def evaluate_mcts_against_agents(
    mcts_solver: MCTSSolver,
    game_variant: str,
    num_players: int,
    opponent_types: List[str],
    num_games: int = 100,
    mcts_player_id: int = 0
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate MCTS against different opponent agents.
    
    Args:
        mcts_solver (MCTSSolver): MCTS solver.
        game_variant (str): Game variant.
        num_players (int): Number of players.
        opponent_types (List[str]): List of opponent agent types.
        num_games (int, optional): Number of games. Defaults to 100.
        mcts_player_id (int, optional): MCTS player ID. Defaults to 0.
        
    Returns:
        Dict[str, Dict[str, float]]: Evaluation results.
    """
    results = {}
    
    for opponent_type in opponent_types:
        print(f"Evaluating MCTS against {opponent_type} opponent...")
        
        results[opponent_type] = {'wins': 0, 'ties': 0, 'losses': 0, 'avg_reward': 0}
        total_reward = 0.0
        
        for game_idx in range(num_games):
            # Create game state
            state = create_game_state(game_variant, num_players)
            
            # Create opponent agent
            opponent_id = 1 if mcts_player_id == 0 else 0
            opponent = create_opponent_agent(opponent_type, opponent_id)
            
            # Play game
            while not state.is_terminal():
                current_player = state.current_player
                
                if current_player == mcts_player_id:
                    # MCTS player's turn
                    action, _ = mcts_solver.search(state, mcts_player_id)
                else:
                    # Opponent's turn
                    action = opponent.act(state)
                
                # Apply action
                state = state.apply_action(action)
            
            # Get rewards
            rewards = state.get_rewards()
            mcts_reward = rewards[mcts_player_id]
            opponent_reward = rewards[opponent_id]
            
            # Update statistics
            total_reward += mcts_reward
            
            if mcts_reward > opponent_reward:
                results[opponent_type]['wins'] += 1
            elif mcts_reward < opponent_reward:
                results[opponent_type]['losses'] += 1
            else:
                results[opponent_type]['ties'] += 1
            
            # Display progress
            if (game_idx + 1) % 10 == 0 or game_idx == num_games - 1:
                print(f"  Progress: {game_idx + 1}/{num_games} games")
        
        # Calculate final statistics
        results[opponent_type]['win_rate'] = results[opponent_type]['wins'] / num_games
        results[opponent_type]['avg_reward'] = total_reward / num_games
        
        # Display results
        print(f"  Results against {opponent_type}:")
        print(f"    Wins: {results[opponent_type]['wins']}")
        print(f"    Ties: {results[opponent_type]['ties']}")
        print(f"    Losses: {results[opponent_type]['losses']}")
        print(f"    Win rate: {results[opponent_type]['win_rate']:.2%}")
        print(f"    Average reward: {results[opponent_type]['avg_reward']:.3f}")
    
    return results


def analyze_mcts_performance(
    mcts_solver: MCTSSolver,
    game_variant: str,
    num_players: int,
    num_games: int = 10,
    mcts_player_id: int = 0
) -> Dict[str, Any]:
    """
    Analyze MCTS performance on specific game scenarios.
    
    Args:
        mcts_solver (MCTSSolver): MCTS solver.
        game_variant (str): Game variant.
        num_players (int): Number of players.
        num_games (int, optional): Number of games. Defaults to 10.
        mcts_player_id (int, optional): MCTS player ID. Defaults to 0.
        
    Returns:
        Dict[str, Any]: Performance metrics.
    """
    metrics = {
        'avg_search_time': 0.0,
        'avg_iterations': 0,
        'avg_nodes': 0,
        'avg_depth': 0,
        'avg_info_sets': 0,
        'stage_actions': {},
    }
    
    total_search_time = 0.0
    total_iterations = 0
    total_nodes = 0
    total_depth = 0
    total_info_sets = 0
    
    # Track actions by stage
    stage_action_counts = {}
    
    for game_idx in range(num_games):
        print(f"Analyzing game {game_idx + 1}/{num_games}...")
        
        # Create game state
        state = create_game_state(game_variant, num_players)
        
        # Create opponent agent (using rule-based agent for better play)
        opponent_id = 1 if mcts_player_id == 0 else 0
        opponent = create_opponent_agent('rule_based_2', opponent_id)
        
        # Play game and collect metrics
        num_actions = 0
        
        while not state.is_terminal():
            current_player = state.current_player
            
            if current_player == mcts_player_id:
                # MCTS player's turn
                stage = str(state.stage)
                
                # Record time
                start_time = time.time()
                
                # Perform search
                action, root = mcts_solver.search(state, mcts_player_id)
                
                # Record search time
                search_time = time.time() - start_time
                total_search_time += search_time
                
                # Update metrics
                total_iterations += mcts_solver.max_iterations
                total_nodes += mcts_solver.total_nodes
                total_depth += mcts_solver.max_depth
                total_info_sets += len(mcts_solver.info_set_nodes)
                
                # Record action by stage
                if stage not in stage_action_counts:
                    stage_action_counts[stage] = {}
                
                action_str = str(action.action_type)
                if action_str not in stage_action_counts[stage]:
                    stage_action_counts[stage][action_str] = 0
                stage_action_counts[stage][action_str] += 1
                
                # Count actions
                num_actions += 1
                
                # Apply action
                state = state.apply_action(action)
            else:
                # Opponent's turn
                action = opponent.act(state)
                state = state.apply_action(action)
        
        # Get rewards
        rewards = state.get_rewards()
        mcts_reward = rewards[mcts_player_id]
        
        print(f"  Game {game_idx + 1} complete, MCTS reward: {mcts_reward:.3f}")
    
    # Calculate averages
    metrics['avg_search_time'] = total_search_time / (num_games * num_actions) if num_actions > 0 else 0
    metrics['avg_iterations'] = total_iterations / (num_games * num_actions) if num_actions > 0 else 0
    metrics['avg_nodes'] = total_nodes / (num_games * num_actions) if num_actions > 0 else 0
    metrics['avg_depth'] = total_depth / (num_games * num_actions) if num_actions > 0 else 0
    metrics['avg_info_sets'] = total_info_sets / (num_games * num_actions) if num_actions > 0 else 0
    metrics['stage_actions'] = stage_action_counts
    
    # Display results
    print("\nMCTS Performance Metrics:")
    print(f"  Average search time: {metrics['avg_search_time']:.4f} seconds")
    print(f"  Average iterations: {metrics['avg_iterations']:.2f}")
    print(f"  Average nodes: {metrics['avg_nodes']:.2f}")
    print(f"  Average depth: {metrics['avg_depth']:.2f}")
    print(f"  Average info sets: {metrics['avg_info_sets']:.2f}")
    
    print("\nAction distribution by stage:")
    for stage, actions in sorted(stage_action_counts.items()):
        print(f"  {stage}:")
        total = sum(actions.values())
        for action, count in sorted(actions.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total * 100 if total > 0 else 0
            print(f"    {action}: {count} ({percentage:.1f}%)")
    
    return metrics


def compare_exploration_settings(
    game_variant: str,
    num_players: int,
    exploration_values: List[float],
    num_games: int = 50,
    mcts_player_id: int = 0
) -> Dict[float, Dict[str, float]]:
    """
    Compare MCTS performance with different exploration settings.
    
    Args:
        game_variant (str): Game variant.
        num_players (int): Number of players.
        exploration_values (List[float]): List of exploration weights to test.
        num_games (int, optional): Number of games. Defaults to 50.
        mcts_player_id (int, optional): MCTS player ID. Defaults to 0.
        
    Returns:
        Dict[float, Dict[str, float]]: Performance metrics for each exploration weight.
    """
    results = {}
    
    for exploration in exploration_values:
        print(f"\nTesting exploration weight {exploration}...")
        
        # Create MCTS solver with specified exploration weight
        mcts_solver = MCTSSolver(
            max_iterations=1000,
            max_time=1.0,
            exploration_weight=exploration,
            random_rollout_depth=10,
            info_set_grouping=True
        )
        
        # Evaluate against rule-based opponent
        eval_results = evaluate_mcts_against_agents(
            mcts_solver,
            game_variant,
            num_players,
            ['rule_based_1'],
            num_games,
            mcts_player_id
        )
        
        # Store results
        results[exploration] = {
            'win_rate': eval_results['rule_based_1']['win_rate'],
            'avg_reward': eval_results['rule_based_1']['avg_reward']
        }
    
    # Display comparison
    print("\nExploration weight comparison:")
    print("  Exploration\tWin Rate\tAvg Reward")
    for exploration, metrics in sorted(results.items()):
        print(f"  {exploration:.2f}\t\t{metrics['win_rate']:.2%}\t\t{metrics['avg_reward']:.3f}")
    
    # Find the best exploration weight
    best_exploration = max(results.keys(), key=lambda x: results[x]['win_rate'])
    print(f"\nBest exploration weight: {best_exploration} (Win rate: {results[best_exploration]['win_rate']:.2%})")
    
    return results


def main():
    """Main function to train and evaluate MCTS agents."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate MCTS for poker")
    parser.add_argument("--iterations", type=int, default=1000, help="Maximum MCTS iterations")
    parser.add_argument("--time-limit", type=float, default=5.0, help="Maximum search time in seconds")
    parser.add_argument("--exploration", type=float, default=1.0, help="Exploration weight")
    parser.add_argument("--game-variant", choices=["holdem", "spin_and_go"], default="holdem", help="Game variant")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    parser.add_argument("--eval-games", type=int, default=100, help="Number of evaluation games")
    parser.add_argument("--analyze", action="store_true", help="Analyze MCTS performance")
    parser.add_argument("--compare-exploration", action="store_true", help="Compare exploration settings")
    parser.add_argument("--opponent", choices=["random", "call", "rule_based_1", "rule_based_2"], 
                      default="random", help="Opponent type for evaluation")
    
    args = vars(parser.parse_args())
    
    # Set up environment
    env = setup_environment(args)
    
    # Create MCTS solver
    mcts_solver = MCTSSolver(
        max_iterations=args['iterations'],
        max_time=args['time_limit'],
        exploration_weight=args['exploration'],
        random_rollout_depth=10,
        info_set_grouping=True
    )
    
    # Analyze MCTS performance
    if args['analyze']:
        analyze_mcts_performance(
            mcts_solver,
            args['game_variant'],
            args['num_players'],
            num_games=10,
            mcts_player_id=0
        )
    
    # Compare exploration settings
    elif args['compare_exploration']:
        compare_exploration_settings(
            args['game_variant'],
            args['num_players'],
            [0.5, 1.0, 1.5, 2.0, 2.5],
            num_games=50,
            mcts_player_id=0
        )
    
    # Evaluate against specified opponent
    else:
        evaluate_mcts_against_agents(
            mcts_solver,
            args['game_variant'],
            args['num_players'],
            [args['opponent']],
            num_games=args['eval_games'],
            mcts_player_id=0
        )
    
    print("\nMCTS training and evaluation complete.")


if __name__ == "__main__":
    main()