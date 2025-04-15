"""
Numba optimizations for poker simulations.

This module provides Numba-accelerated functions for performance-critical
operations in poker simulations, such as hand evaluation, equity calculation,
and Monte Carlo simulations.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional, Union
import time
import logging

try:
    import numba
    from numba import jit, njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Create dummy decorators for when Numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    
    def prange(*args, **kwargs):
        return range(*args, **kwargs)


# Constants for card representations
RANKS = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])  # 2-14 (A=14)
SUITS = np.array([0, 1, 2, 3])  # 0=spades, 1=hearts, 2=diamonds, 3=clubs


@njit
def card_to_int(rank: int, suit: int) -> int:
    """
    Convert a card's rank and suit to an integer representation.
    
    Args:
        rank (int): Card rank (2-14, where Ace=14).
        suit (int): Card suit (0-3).
    
    Returns:
        int: Integer representation of the card.
    """
    return rank * 4 + suit


@njit
def int_to_card(card_int: int) -> Tuple[int, int]:
    """
    Convert an integer representation back to rank and suit.
    
    Args:
        card_int (int): Integer representation of the card.
    
    Returns:
        Tuple[int, int]: Rank and suit.
    """
    rank = card_int // 4
    suit = card_int % 4
    return rank, suit


@njit
def evaluate_hand_fast(cards: np.ndarray) -> Tuple[int, int]:
    """
    Fast hand evaluation using Numba.
    
    Args:
        cards (np.ndarray): Array of card integers.
    
    Returns:
        Tuple[int, int]: Hand rank and value.
    """
    if len(cards) < 5:
        return 0, 0
    
    # Extract ranks and suits
    ranks = np.array([card // 4 for card in cards])
    suits = np.array([card % 4 for card in cards])
    
    # Count rank occurrences
    rank_counts = np.zeros(15, dtype=np.int32)
    for rank in ranks:
        rank_counts[rank] += 1
    
    # Sort cards by rank (descending)
    sorted_indices = np.argsort(-ranks)
    sorted_ranks = ranks[sorted_indices]
    
    # Check for flush
    is_flush = False
    for suit in range(4):
        if np.sum(suits == suit) >= 5:
            is_flush = True
            flush_cards = cards[suits == suit]
            flush_ranks = np.array([card // 4 for card in flush_cards])
            # Sort flush cards by rank (descending)
            flush_ranks = np.sort(flush_ranks)[::-1]
            break
    
    # Check for straight
    is_straight = False
    straight_high = 0
    
    # Handle Ace low straight (A-5-4-3-2)
    if np.sum(rank_counts[2:6]) == 4 and rank_counts[14] == 1:
        is_straight = True
        straight_high = 5
    else:
        # Check for other straights
        for i in range(10):
            high_card = 14 - i
            if all(rank_counts[high_card - j] >= 1 for j in range(5)):
                is_straight = True
                straight_high = high_card
                break
    
    # Determine hand type and value
    
    # Straight flush
    if is_straight and is_flush:
        if straight_high == 14:
            return 9, 0  # Royal flush
        else:
            return 8, straight_high  # Straight flush
    
    # Four of a kind
    four_kind = np.where(rank_counts == 4)[0]
    if len(four_kind) > 0:
        kicker = np.max(np.where(rank_counts == 1)[0]) if np.any(rank_counts == 1) else 0
        return 7, four_kind[0] * 20 + kicker
    
    # Full house
    three_kind = np.where(rank_counts == 3)[0]
    pair = np.where(rank_counts == 2)[0]
    if len(three_kind) > 0 and len(pair) > 0:
        return 6, three_kind[-1] * 20 + pair[-1]
    
    # Check for full house with two three of a kinds
    if len(three_kind) > 1:
        return 6, three_kind[-1] * 20 + three_kind[-2]
    
    # Flush
    if is_flush:
        flush_value = sum(flush_ranks[i] * (10000 // (10 ** i)) for i in range(5))
        return 5, flush_value
    
    # Straight
    if is_straight:
        return 4, straight_high
    
    # Three of a kind
    if len(three_kind) > 0:
        kickers = sorted_ranks[np.where(sorted_ranks != three_kind[0])[0]][:2]
        value = three_kind[0] * 400 + kickers[0] * 20 + kickers[1]
        return 3, value
    
    # Two pair
    if len(pair) >= 2:
        kicker = np.max(np.where(rank_counts == 1)[0]) if np.any(rank_counts == 1) else 0
        value = pair[-1] * 400 + pair[-2] * 20 + kicker
        return 2, value
    
    # One pair
    if len(pair) == 1:
        kickers = sorted_ranks[np.where(sorted_ranks != pair[0])[0]][:3]
        value = pair[0] * 8000 + kickers[0] * 400 + kickers[1] * 20 + kickers[2]
        return 1, value
    
    # High card
    value = sum(sorted_ranks[i] * (10000 // (10 ** i)) for i in range(5))
    return 0, value


@njit(parallel=True)
def monte_carlo_equity(hole_cards: np.ndarray, num_players: int, board: np.ndarray, 
                      iterations: int) -> np.ndarray:
    """
    Calculate equity using Monte Carlo simulation.
    
    Args:
        hole_cards (np.ndarray): Array of hole cards for each player.
        num_players (int): Number of players.
        board (np.ndarray): Community cards on the board.
        iterations (int): Number of simulations to run.
    
    Returns:
        np.ndarray: Equity estimates for each player.
    """
    # Initialize equity counters
    equity = np.zeros(num_players, dtype=np.float64)
    
    # Create a deck of cards (0-51)
    full_deck = np.arange(52, dtype=np.int32)
    
    # Remove cards that are already dealt
    used_cards = np.concatenate((hole_cards.flatten(), board))
    used_cards = used_cards[used_cards >= 0]  # Remove placeholders (-1)
    
    for _ in prange(iterations):
        # Make a copy of the current board
        current_board = board.copy()
        remaining_board = 5 - len(current_board[current_board >= 0])
        
        # Create available cards
        available = np.setdiff1d(full_deck, used_cards)
        np.random.shuffle(available)
        
        # Complete the board
        completed_board = np.concatenate((current_board[current_board >= 0], 
                                        available[:remaining_board]))
        
        # Evaluate hands and determine winners
        best_hand = np.zeros(num_players, dtype=np.int32)
        best_value = np.zeros(num_players, dtype=np.int32)
        
        for p in range(num_players):
            player_hole = hole_cards[p]
            if player_hole[0] >= 0 and player_hole[1] >= 0:  # Valid hole cards
                player_cards = np.concatenate((player_hole, completed_board))
                hand_rank, hand_value = evaluate_hand_fast(player_cards)
                best_hand[p] = hand_rank
                best_value[p] = hand_value
            else:
                best_hand[p] = -1  # Invalid/folded player
        
        # Determine winners
        max_hand = np.max(best_hand)
        if max_hand >= 0:  # At least one valid player
            hand_winners = np.where(best_hand == max_hand)[0]
            
            if len(hand_winners) == 1:
                # Single winner
                equity[hand_winners[0]] += 1.0
            else:
                # Tie - check hand values
                max_value = np.max(best_value[hand_winners])
                value_winners = hand_winners[best_value[hand_winners] == max_value]
                
                # Split equity among winners
                split_equity = 1.0 / len(value_winners)
                for winner in value_winners:
                    equity[winner] += split_equity
    
    # Convert to probabilities
    return equity / iterations


@njit
def calculate_hand_strength(hole_cards: np.ndarray, board: np.ndarray) -> float:
    """
    Calculate the hand strength (probability of winning against random hands).
    
    Args:
        hole_cards (np.ndarray): Player's hole cards.
        board (np.ndarray): Community cards on the board.
    
    Returns:
        float: Hand strength (0.0-1.0).
    """
    total_hands = 0
    ahead_or_tied = 0
    
    # Create a deck of cards (0-51)
    full_deck = np.arange(52, dtype=np.int32)
    
    # Remove cards that are already dealt
    used_cards = np.concatenate((hole_cards, board))
    used_cards = used_cards[used_cards >= 0]  # Remove placeholders (-1)
    available = np.setdiff1d(full_deck, used_cards)
    
    # Evaluate our hand
    our_cards = np.concatenate((hole_cards, board[board >= 0]))
    our_rank, our_value = evaluate_hand_fast(our_cards)
    
    # Sample random opponent hands
    samples = min(1000, len(available) * (len(available) - 1) // 2)
    for _ in range(samples):
        # Choose random hole cards for opponent
        idx1 = np.random.randint(0, len(available))
        idx2 = np.random.randint(0, len(available))
        if idx1 == idx2:
            idx2 = (idx2 + 1) % len(available)
        
        opp_hole = np.array([available[idx1], available[idx2]])
        
        # Evaluate opponent's hand
        opp_cards = np.concatenate((opp_hole, board[board >= 0]))
        opp_rank, opp_value = evaluate_hand_fast(opp_cards)
        
        # Compare hands
        if our_rank > opp_rank or (our_rank == opp_rank and our_value >= opp_value):
            ahead_or_tied += 1
        
        total_hands += 1
    
    # Calculate strength
    return ahead_or_tied / total_hands if total_hands > 0 else 0.0


@njit
def calculate_pot_odds(pot_size: float, call_amount: float) -> float:
    """
    Calculate pot odds.
    
    Args:
        pot_size (float): Current pot size.
        call_amount (float): Amount needed to call.
    
    Returns:
        float: Pot odds (0.0-1.0).
    """
    return call_amount / (pot_size + call_amount) if (pot_size + call_amount) > 0 else 0.0


@njit
def calculate_expected_value(win_probability: float, pot_size: float, 
                           call_amount: float) -> float:
    """
    Calculate the expected value of a call.
    
    Args:
        win_probability (float): Probability of winning the hand.
        pot_size (float): Current pot size.
        call_amount (float): Amount needed to call.
    
    Returns:
        float: Expected value.
    """
    return win_probability * pot_size - (1 - win_probability) * call_amount


# Optimized functions for poker agent decision making
def optimize_agent_decision(f):
    """
    Decorator to optimize agent decision functions with Numba if available.
    
    Args:
        f: Function to optimize.
    
    Returns:
        Optimized function.
    """
    if HAS_NUMBA:
        return njit(f)
    return f


@optimize_agent_decision
def should_fold(hand_strength: float, pot_odds: float, aggression: float = 0.0) -> bool:
    """
    Determine if the agent should fold based on hand strength and pot odds.
    
    Args:
        hand_strength (float): Hand strength (0.0-1.0).
        pot_odds (float): Pot odds (0.0-1.0).
        aggression (float, optional): Aggression factor (-1.0 to 1.0). Defaults to 0.0.
    
    Returns:
        bool: Whether to fold.
    """
    threshold = pot_odds * (1.0 - 0.2 * aggression)
    return hand_strength < threshold