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

logger = logging.getLogger("pokersim.numba_optimizations")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

RANKS = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
SUITS = np.array([0, 1, 2, 3])

@njit
def card_to_int(rank: int, suit: int) -> int:
    """
    Convert a card's rank and suit to an integer representation.
    """
    return rank * 4 + suit

@njit
def int_to_card(card_int: int) -> Tuple[int, int]:
    """
    Convert an integer representation back to rank and suit.
    """
    rank = card_int // 4
    suit = card_int % 4
    return rank, suit

@njit
def evaluate_hand_fast(cards: np.ndarray) -> Tuple[int, int]:
    """
    Fast hand evaluation using Numba.
    """
    if len(cards) < 5:
        return 0, 0

    ranks = cards // 4
    suits = cards % 4
    rank_counts = np.zeros(15, dtype=np.int32)
    for i in range(len(ranks)):
        rank_counts[ranks[i]] += 1

    sorted_indices = np.argsort(-ranks)
    sorted_ranks = ranks[sorted_indices]

    is_flush = False
    for suit in range(4):
        if np.sum(suits == suit) >= 5:
            is_flush = True
            flush_cards = cards[suits == suit]
            flush_ranks = np.array([card // 4 for card in flush_cards])
            flush_ranks = np.sort(flush_ranks)[::-1]
            break

    is_straight = False
    straight_high = 0
    if np.sum(rank_counts[2:6]) == 4 and rank_counts[14] == 1:
        is_straight = True
        straight_high = 5
    else:
        for i in range(10):
            high_card = 14 - i
            if all(rank_counts[high_card - j] >= 1 for j in range(5)):
                is_straight = True
                straight_high = high_card
                break

    if is_straight and is_flush:
        if straight_high == 14:
            return 9, 0
        else:
            return 8, straight_high

    four_kind = np.where(rank_counts == 4)[0]
    if len(four_kind) > 0:
        kicker = np.max(np.where(rank_counts == 1)[0]) if np.any(rank_counts == 1) else 0
        return 7, four_kind[0] * 20 + kicker

    three_kind = np.where(rank_counts == 3)[0]
    pair = np.where(rank_counts == 2)[0]
    if len(three_kind) > 0 and len(pair) > 0:
        return 6, three_kind[-1] * 20 + pair[-1]
    if len(three_kind) > 1:
        return 6, three_kind[-1] * 20 + three_kind[-2]

    if is_flush:
        flush_value = sum(flush_ranks[i] * (10000 // (10 ** i)) for i in range(5))
        return 5, flush_value

    if is_straight:
        return 4, straight_high

    if len(three_kind) > 0:
        kickers = sorted_ranks[np.where(sorted_ranks != three_kind[0])[0]][:2]
        value = three_kind[0] * 400 + kickers[0] * 20 + kickers[1]
        return 3, value

    if len(pair) >= 2:
        kicker = np.max(np.where(rank_counts == 1)[0]) if np.any(rank_counts == 1) else 0
        value = pair[-1] * 400 + pair[-2] * 20 + kicker
        return 2, value

    if len(pair) == 1:
        kickers = sorted_ranks[np.where(sorted_ranks != pair[0])[0]][:3]
        value = pair[0] * 8000 + kickers[0] * 400 + kickers[1] * 20 + kickers[2]
        return 1, value

    value = sum(sorted_ranks[i] * (10000 // (10 ** i)) for i in range(5))
    return 0, value

@njit(parallel=True)
def monte_carlo_equity(hole_cards: np.ndarray, num_players: int, board: np.ndarray,
                      iterations: int = 1000) -> np.ndarray:
    """
    Calculate equity using Monte Carlo simulation.
    """
    if num_players < 2:
        logger.error(f"Invalid number of players: {num_players}")
        raise ValueError("Number of players must be at least 2")
    if hole_cards.shape[0] != num_players:
        logger.error(f"Expected {num_players} players, got {hole_cards.shape[0]}")
        raise ValueError(f"Expected {num_players} players, got {hole_cards.shape[0]}")
    if iterations < 1:
        logger.error("Iterations must be positive")
        raise ValueError("Iterations must be positive")

    logger.debug(f"Starting Monte Carlo equity calculation with {iterations} iterations")
    equity = np.zeros(num_players, dtype=np.float64)
    full_deck = np.arange(52, dtype=np.int32)
    used_cards = np.concatenate((hole_cards.flatten(), board))
    used_cards = used_cards[used_cards >= 0]

    for sim in prange(iterations):
        current_board = board.copy()
        remaining_board = 5 - len(current_board[current_board >= 0])
        available = np.setdiff1d(full_deck, used_cards)
        np.random.shuffle(available)
        completed_board = np.concatenate((current_board[current_board >= 0],
                                        available[:remaining_board]))
        best_hand = np.zeros(num_players, dtype=np.int32)
        best_value = np.zeros(num_players, dtype=np.int32)

        for p in range(num_players):
            player_hole = hole_cards[p]
            if player_hole[0] >= 0 and player_hole[1] >= 0:
                player_cards = np.concatenate((player_hole, completed_board))
                hand_rank, hand_value = evaluate_hand_fast(player_cards)
                best_hand[p] = hand_rank
                best_value[p] = hand_value
            else:
                best_hand[p] = -1

        max_hand = np.max(best_hand)
        if max_hand >= 0:
            hand_winners = np.where(best_hand == max_hand)[0]
            if len(hand_winners) == 1:
                equity[hand_winners[0]] += 1.0
            else:
                max_value = np.max(best_value[hand_winners])
                value_winners = hand_winners[best_value[hand_winners] == max_value]
                split_equity = 1.0 / len(value_winners)
                for winner in value_winners:
                    equity[winner] += split_equity

    result = equity / iterations
    logger.info(f"Equity calculation complete: {result}")
    return result

@njit
def calculate_hand_strength(hole_cards: np.ndarray, board: np.ndarray) -> float:
    """
    Calculate the hand strength (probability of winning against random hands).
    """
    total_hands = 0
    ahead_or_tied = 0
    full_deck = np.arange(52, dtype=np.int32)
    used_cards = np.concatenate((hole_cards, board))
    used_cards = used_cards[used_cards >= 0]
    available = np.setdiff1d(full_deck, used_cards)
    our_cards = np.concatenate((hole_cards, board[board >= 0]))
    our_rank, our_value = evaluate_hand_fast(our_cards)
    samples = min(1000, len(available) * (len(available) - 1) // 2)

    for _ in range(samples):
        idx1 = np.random.randint(0, len(available))
        idx2 = np.random.randint(0, len(available))
        if idx1 == idx2:
            idx2 = (idx2 + 1) % len(available)
        opp_hole = np.array([available[idx1], available[idx2]])
        opp_cards = np.concatenate((opp_hole, board[board >= 0]))
        opp_rank, opp_value = evaluate_hand_fast(opp_cards)
        if our_rank > opp_rank or (our_rank == opp_rank and our_value >= opp_value):
            ahead_or_tied += 1
        total_hands += 1

    strength = ahead_or_tied / total_hands if total_hands > 0 else 0.0
    logger.debug(f"Calculated hand strength: {strength:.4f}")
    return strength

@njit
def calculate_pot_odds(pot_size: float, call_amount: float) -> float:
    """
    Calculate pot odds.
    """
    odds = call_amount / (pot_size + call_amount) if (pot_size + call_amount) > 0 else 0.0
    logger.debug(f"Calculated pot odds: {odds:.4f}")
    return odds

@njit
def calculate_expected_value(win_probability: float, pot_size: float,
                           call_amount: float) -> float:
    """
    Calculate the expected value of a call.
    """
    ev = win_probability * pot_size - (1 - win_probability) * call_amount
    logger.debug(f"Calculated expected value: {ev:.4f}")
    return ev

def optimize_agent_decision(f):
    """
    Decorator to optimize agent decision functions with Numba if available.
    """
    if HAS_NUMBA:
        return njit(f)
    return f

@optimize_agent_decision
def should_fold(hand_strength: float, pot_odds: float, aggression: float = 0.0) -> bool:
    """
    Determine if the agent should fold based on hand strength and pot odds.
    """
    threshold = pot_odds * (1.0 - 0.2 * aggression)
    fold = hand_strength < threshold
    logger.debug(f"Should fold: {fold}, hand_strength={hand_strength:.4f}, pot_odds={pot_odds:.4f}")
    return fold
