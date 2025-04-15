"""
Example script demonstrating the usage of advanced algorithms.

This script shows how to use more sophisticated algorithms like MCTS and CFR
for poker decision making, training, and evaluation.
"""

import random
import numpy as np
import time
import argparse
from typing import List, Dict, Any, Tuple

from pokersim.game.spingo import SpinGoGame
from pokersim.agents.base_agent import Agent
from pokersim.agents.rule_based_agent import RuleBased1Agent
from pokersim.algorithms.mcts import MCTS, ISMCTSPlayer
from pokersim.algorithms.cfr import CFR, CFRPlayer
from examples.spin_and_go_training import print_separator, simulate_tournaments


def mcts_demo(num_players: int = 3, num_simulations: int = 100):
    """
    Demonstrate Monte Carlo Tree Search algorithm on a Spin and Go game.
    
    Args:
        num_players (int, optional): Number of players. Defaults to 3.
        num_simulations (int, optional): Number of MCTS simulations. Defaults to 100.
    """
    print_separator()
    print("Monte Carlo Tree Search (MCTS) Demonstration")
    print_separator()
    
    # Create a new game
    game = SpinGoGame(num_players=num_players)
    game_state = game.start_new_hand()
    
    # Create agents
    agents = []
    
    # MCTS agent for player 0
    mcts_agent = ISMCTSPlayer(player_id=0, num_simulations=num_simulations)
    agents.append(mcts_agent)
    
    # Rule-based agents for other players
    for i in range(1, num_players):
        agents.append(RuleBased1Agent(i))
    
    # Play a single hand
    print(f"Starting a Spin and Go hand with {num_players} players")
    print(f"Player 0 is using MCTS with {num_simulations} simulations per decision")
    print(f"Players 1 to {num_players-1} are using rule-based strategies")
    
    hand_num = 1
    tournament_over = False
    
    while not tournament_over:
        print(f"\nHand #{hand_num}")
        
        # Play the hand
        while not game_state.is_terminal():
            current_player = game_state.current_player
            
            print(f"Player {current_player}'s turn")
            
            # Get legal actions
            legal_actions = game_state.get_legal_actions()
            print(f"Legal actions: {legal_actions}")
            
            # Use the appropriate agent to choose an action
            start_time = time.time()
            action = agents[current_player].act(game_state)
            end_time = time.time()
            
            decision_time = end_time - start_time
            print(f"Player {current_player} chooses {action} (took {decision_time:.2f}s)")
            
            # Apply the action
            game_state = game_state.apply_action(action)
        
        # Hand is over
        print("\nHand complete!")
        print(f"Community cards: {game_state.community_cards}")
        
        for i in range(num_players):
            if i in game_state.hole_cards:
                print(f"Player {i} hole cards: {game_state.hole_cards[i]}")
        
        # Check if tournament is over
        game.update_stacks_after_hand()
        
        if game.is_tournament_over():
            tournament_over = True
            winner = game.get_winner()
            prize = game.get_prize(winner)
            print(f"\nTournament over! Player {winner} wins ${prize}!")
        else:
            # Start a new hand
            print("\nStarting a new hand...")
            game_state = game.start_new_hand()
            hand_num += 1


def train_cfr_agent(num_iterations: int = 1000) -> CFR:
    """
    Train a CFR agent for Spin and Go.
    
    Args:
        num_iterations (int, optional): Number of training iterations. Defaults to 1000.
    
    Returns:
        CFR: Trained CFR agent.
    """
    print_separator()
    print("Training CFR Agent for Spin and Go")
    print_separator()
    
    # Create and train CFR agent
    cfr = CFR(game_type='spingo', num_players=3)
    
    print(f"Training for {num_iterations} iterations...")
    start_time = time.time()
    
    # Run CFR training
    cfr.train(num_iterations=num_iterations, pruning=True)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Training complete in {training_time:.2f} seconds")
    print(f"Learned {len(cfr.info_sets)} information sets")
    
    return cfr


def evaluate_algorithms(num_tournaments: int = 20):
    """
    Compare different algorithms on Spin and Go tournaments.
    
    Args:
        num_tournaments (int, optional): Number of tournaments. Defaults to 20.
    """
    print_separator()
    print("Algorithm Comparison on Spin and Go")
    print_separator()
    
    # Define agent combinations to evaluate
    agent_combinations = [
        ["rule_based", "rule_based", "rule_based"],  # Baseline
        ["mcts", "rule_based", "rule_based"],        # MCTS vs rule-based
        ["cfr", "rule_based", "rule_based"],         # CFR vs rule-based
        ["mcts", "cfr", "rule_based"]                # MCTS vs CFR vs rule-based
    ]
    
    results = {}
    
    # Train a CFR agent (with fewer iterations for the example)
    cfr = train_cfr_agent(num_iterations=100)
    
    # Register the CFR agent with the simulator
    # (This would need to be implemented in the simulate_tournaments function)
    
    # Evaluate each combination
    for agents in agent_combinations:
        combo_name = " vs ".join(agents)
        print(f"\nEvaluating: {combo_name}")
        
        stats = simulate_tournaments(
            num_tournaments=num_tournaments,
            agent_types=agents,
            verbose=False
        )
        
        # Print summary
        print("Results:")
        for i in range(len(agents)):
            win_rate = stats["player_wins"][i] / stats["tournaments_played"]
            roi = stats["player_roi"][i]
            print(f"  Player {i} ({agents[i]}): Win rate {win_rate:.3f}, ROI {roi:.3f}")
        
        # Store results
        results[combo_name] = {
            "win_rates": [stats["player_wins"][i] / stats["tournaments_played"] for i in range(len(agents))],
            "roi": stats["player_roi"],
            "avg_tournament_length": stats["avg_tournament_length"]
        }
    
    print("\nOverall comparison:")
    for combo_name, res in results.items():
        print(f"{combo_name}:")
        for i, agent_type in enumerate(combo_name.split(" vs ")):
            print(f"  {agent_type}: Win rate {res['win_rates'][i]:.3f}, ROI {res['roi'][i]:.3f}")


def main():
    """Main function for the advanced algorithms example."""
    parser = argparse.ArgumentParser(description="Advanced Algorithms Example")
    parser.add_argument("--mode", type=str, default="mcts", 
                      choices=["mcts", "cfr", "compare"],
                      help="Mode of operation")
    parser.add_argument("--num_players", type=int, default=3,
                      help="Number of players")
    parser.add_argument("--num_simulations", type=int, default=100,
                      help="Number of MCTS simulations")
    parser.add_argument("--num_iterations", type=int, default=1000,
                      help="Number of CFR training iterations")
    parser.add_argument("--num_tournaments", type=int, default=20,
                      help="Number of tournaments for evaluation")
    args = parser.parse_args()
    
    if args.mode == "mcts":
        mcts_demo(
            num_players=args.num_players,
            num_simulations=args.num_simulations
        )
    
    elif args.mode == "cfr":
        train_cfr_agent(
            num_iterations=args.num_iterations
        )
    
    elif args.mode == "compare":
        evaluate_algorithms(
            num_tournaments=args.num_tournaments
        )


if __name__ == "__main__":
    main()