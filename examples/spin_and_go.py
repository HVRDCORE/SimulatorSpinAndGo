"""
Example script demonstrating the Spin and Go game mode.

This script simulates a complete Spin and Go tournament with advanced opponent agents.
It shows the tournament flow, player eliminations, and final results.
"""

import sys
import os
import random
import time
import argparse
from typing import List, Dict, Tuple, Any

# Add the parent directory to sys.path to allow imports from the pokersim package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pokersim.game.state import GameState, Action, ActionType, Stage
from pokersim.game.spingo import SpinGoGame
from pokersim.agents.advanced_opponent_agent import (
    AdvancedOpponentAgent, LOOSE_AGGRESSIVE_PROFILE, TIGHT_PASSIVE_PROFILE,
    TIGHT_AGGRESSIVE_PROFILE, LOOSE_PASSIVE_PROFILE, BALANCED_PROFILE,
    MANIAC_PROFILE, ROCK_PROFILE, ADAPTIVE_PROFILE
)
from pokersim.agents.base_agent import Agent


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
    print(f"Stage: {game_state.stage}")
    print(f"Pot: {game_state.pot}")
    print(f"Community cards: {' '.join(str(card) for card in game_state.community_cards) if game_state.community_cards else 'None'}")
    print("----------------------------------------")
    print("Players:")
    
    for i in range(game_state.num_players):
        active_str = "Active" if game_state.active[i] else "Folded"
        current_marker = "→ " if i == game_state.current_player else "  "
        
        # Show cards based on parameters
        if show_all_cards:
            cards_str = f"Cards: {' '.join(str(card) for card in game_state.hole_cards[i])}"
        else:
            if not game_state.active[i]:
                cards_str = "Cards: Folded"
            elif i == 0:  # Always show player 0's cards (user perspective)
                cards_str = f"Cards: {' '.join(str(card) for card in game_state.hole_cards[i])}"
            else:
                cards_str = "Cards: [Hidden]"
        
        print(f"{current_marker}P{i}   {active_str} | Stack: ${game_state.stacks[i]} | Bet: ${game_state.current_bets[i]} | {cards_str}")
    
    # Print legal actions if it's player 0's turn
    if game_state.current_player == 0:
        print("----------------------------------------")
        print("Legal actions:")
        for action in game_state.get_legal_actions():
            print(f"  {action}")


def print_tournament_state(tournament: SpinGoGame) -> None:
    """
    Print the current tournament state.
    
    Args:
        tournament (SpinGoGame): The tournament to print.
    """
    print_separator()
    print(f"TOURNAMENT STATUS")
    print(f"Buy-in: ${tournament.buy_in}  |  Multiplier: {tournament.multiplier}x  |  Prize Pool: ${tournament.prize_pool}")
    print(f"Hands played: {tournament.hands_played}  |  Current blinds: {tournament.current_blinds[0]}/{tournament.current_blinds[1]}")
    print("Remaining players:")
    
    for player_id in tournament.remaining_players:
        print(f"  Player {player_id}: ${tournament.player_stacks[player_id]}")
    
    if tournament.eliminated_players:
        print("Eliminated players:")
        for player_id in tournament.eliminated_players:
            print(f"  Player {player_id}")
    
    print_separator()


def simulate_spin_and_go(buy_in: int = 10, num_players: int = 3, verbose: bool = True, delay: float = 0.5) -> Dict:
    """
    Simulate a complete Spin and Go tournament.
    
    Args:
        buy_in (int, optional): Buy-in amount. Defaults to 10.
        num_players (int, optional): Number of players. Defaults to 3.
        verbose (bool, optional): Whether to print details. Defaults to True.
        delay (float, optional): Delay between actions in seconds. Defaults to 0.5.
        
    Returns:
        Dict: Tournament results including winner and prize.
    """
    # Create a Spin and Go tournament
    tournament = SpinGoGame(buy_in=buy_in, num_players=num_players)
    
    # Print initial tournament info
    if verbose:
        print("\n=== NEW SPIN & GO TOURNAMENT ===")
        print(f"Buy-in: ${buy_in}  |  Players: {num_players}")
        print(f"Prize pool multiplier: {tournament.multiplier}x")
        print(f"Total prize pool: ${tournament.prize_pool}")
        print_separator()
    
    # Create agents for each player (with different profiles)
    profiles = [
        BALANCED_PROFILE,  # Player 0 (usually the human player in a real app)
        LOOSE_AGGRESSIVE_PROFILE,
        TIGHT_PASSIVE_PROFILE
    ]
    
    # Add more profiles if needed
    if num_players > 3:
        additional_profiles = [
            TIGHT_AGGRESSIVE_PROFILE,
            LOOSE_PASSIVE_PROFILE,
            MANIAC_PROFILE,
            ROCK_PROFILE,
            ADAPTIVE_PROFILE
        ]
        
        profiles.extend(additional_profiles[:num_players - 3])
    
    agents = [AdvancedOpponentAgent(i, profile) for i, profile in enumerate(profiles[:num_players])]
    
    # Play the tournament until we have a winner
    hand_count = 0
    while not tournament.is_tournament_over():
        hand_count += 1
        
        if verbose:
            print(f"\n=== HAND #{hand_count} ===")
            print_tournament_state(tournament)
        
        # Start a new hand
        game_state = tournament.start_new_hand()
        
        # Play the hand until completion
        while not game_state.is_terminal():
            # Print game state
            if verbose:
                print_separator()
                print_game_state(game_state, show_all_cards=False)
                time.sleep(delay)
            
            # Get the current player's agent
            current_player = game_state.current_player
            
            if current_player == -1:
                continue
                
            agent = agents[current_player]
            
            # Get the agent's action
            action = agent.act(game_state)
            
            # Apply action
            game_state = game_state.apply_action(action)
            
            # Print the action
            if verbose:
                print(f"Player {current_player} takes action: {action}")
        
        # Hand is complete, update tournament state
        tournament.update_stacks_after_hand()
        
        # Show the final hand state
        if verbose:
            print_separator()
            print_game_state(game_state, show_all_cards=True)
            
            # Show results
            print("\n=== HAND COMPLETE ===")
            print_separator()
            print("Results:")
            print_separator()
            for player_id, payout in enumerate(game_state.get_payouts()):
                if payout > 0:
                    print(f"Player {player_id} won ${payout}")
            
            # Notify players of eliminations
            eliminated_this_hand = [p for p in range(num_players) 
                                   if p not in tournament.eliminated_players and 
                                   p not in tournament.remaining_players]
            for player_id in eliminated_this_hand:
                print(f"Player {player_id} has been eliminated!")
            
            time.sleep(delay)
        
        # Allow agents to observe the outcome
        for agent in agents:
            agent.observe(game_state)
    
    # Tournament is over
    winner = tournament.get_winner()
    prize = 0
    
    if winner is not None:
        prize = tournament.get_prize(winner)
    
    if verbose:
        print("\n=== TOURNAMENT COMPLETE ===")
        print_separator()
        if winner is not None:
            print(f"Winner: Player {winner}")
            print(f"Prize: ${prize}")
        else:
            print("No winner determined")
        print_separator()
    
    return {
        'winner': winner,
        'prize': prize,
        'buy_in': buy_in,
        'multiplier': tournament.multiplier,
        'hands_played': hand_count
    }


def main():
    """Main function to demonstrate Spin and Go simulation."""
    parser = argparse.ArgumentParser(description='Simulate a Spin and Go tournament.')
    parser.add_argument('--buy_in', type=int, default=10, help='Buy-in amount')
    parser.add_argument('--players', type=int, default=3, help='Number of players')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between actions in seconds')
    parser.add_argument('--quiet', action='store_true', help='Run in quiet mode (no output)')
    
    args = parser.parse_args()
    
    # Simulate tournament
    results = simulate_spin_and_go(
        buy_in=args.buy_in,
        num_players=args.players,
        verbose=not args.quiet,
        delay=args.delay
    )
    
    # Print final summary
    print("\nTournament Summary:")
    print(f"Buy-in: ${results['buy_in']}")
    print(f"Multiplier: {results['multiplier']}x")
    print(f"Prize: ${results['prize']}")
    if results['winner'] is not None:
        print(f"Winner: Player {results['winner']}")
    else:
        print("No winner determined")
    print(f"Hands played: {results['hands_played']}")
    
    print("\nThank you for using PokerSim!")


if __name__ == "__main__":
    main()