"""
Example script demonstrating basic poker game simulation.

This script simulates a poker game with different types of agents and
visualizes the game state and actions taken by each agent.
"""

import sys
import os
import random
import time
from typing import List, Dict

# Add the parent directory to sys.path to allow imports from the pokersim package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pokersim.game.state import GameState, Action, ActionType, Stage
from pokersim.game.card import Card, Suit, Rank
from pokersim.game.deck import Deck
from pokersim.game.evaluator import HandEvaluator
from pokersim.agents.random_agent import RandomAgent
from pokersim.agents.call_agent import CallAgent
from pokersim.agents.rule_based_agent import RuleBased1Agent, RuleBased2Agent


def print_separator(char: str = "-", length: int = 80) -> None:
    """Print a separator line."""
    print(char * length)


def print_game_state(game_state: GameState, show_all_cards: bool = False) -> None:
    """
    Print the current game state in a readable format.
    
    Args:
        game_state (GameState): The game state to print.
        show_all_cards (bool, optional): Whether to show all player cards. Defaults to False.
    """
    print_separator()
    print(f"Stage: {game_state.stage}")
    print(f"Pot: {game_state.pot}")
    
    # Print community cards
    if game_state.community_cards:
        print(f"Community cards: {' '.join(str(card) for card in game_state.community_cards)}")
    else:
        print("Community cards: None")
    
    # Print player info
    print_separator("-", 40)
    print("Players:")
    for i in range(game_state.num_players):
        position = ""
        if i == game_state.button:
            position += "BTN "
        elif i == (game_state.button + 1) % game_state.num_players:
            position += "SB  "
        elif i == (game_state.button + 2) % game_state.num_players:
            position += "BB  "
        else:
            position += f"P{i}  "
        
        status = "Active" if game_state.active[i] else "Folded"
        
        # Print cards
        cards = ""
        if show_all_cards or i == 0:  # Show our cards or all cards if specified
            if game_state.hole_cards[i]:
                cards = f"Cards: {' '.join(str(card) for card in game_state.hole_cards[i])}"
            else:
                cards = "Cards: None"
        else:
            if game_state.active[i]:
                cards = "Cards: [Hidden]"
            else:
                cards = "Cards: Folded"
        
        # Print stack and bet
        stack_info = f"Stack: ${game_state.stacks[i]}"
        bet_info = f"Bet: ${game_state.current_bets[i]}"
        
        # Highlight current player
        current = "→ " if i == game_state.current_player else "  "
        
        print(f"{current}{position} {status} | {stack_info} | {bet_info} | {cards}")
    
    # Print legal actions if any
    if game_state.current_player >= 0 and game_state.current_player < game_state.num_players:
        print_separator("-", 40)
        print(f"Legal actions for Player {game_state.current_player}:")
        for action in game_state.get_legal_actions():
            print(f"  {action}")


def simulate_game(num_players: int = 6, initial_stacks: List[int] = None, 
                verbose: bool = True, delay: float = 0.5) -> Dict:
    """
    Simulate a complete poker hand.
    
    Args:
        num_players (int, optional): Number of players. Defaults to 6.
        initial_stacks (List[int], optional): Initial stacks. Defaults to None (100 for each player).
        verbose (bool, optional): Whether to print game state. Defaults to True.
        delay (float, optional): Delay between actions in seconds. Defaults to 0.5.
        
    Returns:
        Dict: Game results including winners and payouts.
    """
    if initial_stacks is None:
        initial_stacks = [100] * num_players
    
    # Set up the game
    game_state = GameState(num_players=num_players, initial_stacks=initial_stacks)
    
    # Create agents - mix of different types for variety
    agents = []
    agents.append(RuleBased2Agent(0, aggression=0.7, bluff_frequency=0.1))  # Player (human in UI)
    agents.append(RandomAgent(1))  # Random player
    agents.append(CallAgent(2))    # Calling station
    agents.append(RuleBased1Agent(3, aggression=0.3))  # Tight player
    agents.append(RuleBased1Agent(4, aggression=0.8))  # Aggressive player
    agents.append(RuleBased2Agent(5, aggression=0.6, bluff_frequency=0.2))  # Balanced player
    
    # Ensure we have enough agents
    if len(agents) < num_players:
        for i in range(len(agents), num_players):
            agents.append(RandomAgent(i))
    
    # Print initial state
    if verbose:
        print("\n=== NEW HAND ===")
        print_game_state(game_state)
    
    # Game loop
    while not game_state.is_terminal():
        current_player = game_state.current_player
        
        # Get action from the current player's agent
        if current_player >= 0 and current_player < len(agents):
            action = agents[current_player].act(game_state)
            
            if verbose:
                print(f"\nPlayer {current_player} takes action: {action}")
                time.sleep(delay)
            
            # Apply the action
            game_state = game_state.apply_action(action)
            
            if verbose:
                print_game_state(game_state)
        else:
            if verbose:
                print("Error: Invalid current player")
            break
    
    # Game ended, show results
    if verbose:
        print("\n=== HAND COMPLETE ===")
        print_game_state(game_state, show_all_cards=True)
        
        print_separator()
        print("Results:")
        for i, payout in enumerate(game_state.get_payouts()):
            if payout > 0:
                print(f"Player {i} wins ${payout}")
        
        print_separator()
        print("Final hand evaluation:")
        for i, cards in enumerate(game_state.hole_cards):
            if game_state.active[i] and cards:
                hand = cards + game_state.community_cards
                rank, desc = HandEvaluator.evaluate_hand(hand)
                print(f"Player {i}: {desc}")
    
    # Return game results
    results = {
        'payouts': game_state.get_payouts(),
        'winners': [i for i, payout in enumerate(game_state.get_payouts()) if payout > 0],
        'final_state': game_state
    }
    
    return results


def main():
    """Main function to demonstrate poker simulation."""
    print("Welcome to PokerSim Basic Game Example")
    print("======================================")
    print("This script simulates a full poker hand with various types of agents.")
    print("You'll see the game state after each action, including cards, bets, and legal actions.")
    print("The player agents include a mix of strategies: random, calling station, and rule-based.")
    print()
    
    # Prompt user for simulation speed
    try:
        delay = float(input("Enter delay between actions (seconds, default 0.5): ") or 0.5)
        num_players = int(input("Enter number of players (2-9, default 6): ") or 6)
        num_players = max(2, min(9, num_players))
    except ValueError:
        print("Invalid input, using default values.")
        delay = 0.5
        num_players = 6
    
    # Run the simulation
    results = simulate_game(num_players=num_players, delay=delay)
    
    # Display summary
    print("\nGame Summary:")
    print(f"Total players: {num_players}")
    print(f"Winners: {', '.join([f'Player {i}' for i in results['winners']])}")
    print(f"Payouts: {', '.join([f'Player {i}: ${p}' for i, p in enumerate(results['payouts']) if p > 0])}")
    
    print("\nThank you for using PokerSim!")
    print("For more examples and details, check the README.md and docs/ directory.")


if __name__ == "__main__":
    main()
