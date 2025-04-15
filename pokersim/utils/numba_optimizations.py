"""
Optimized functions using Numba for poker hand evaluation and simulation.

This module provides high-performance implementations of computationally intensive
operations in poker, such as hand evaluation, monte carlo simulation, and game
state evaluation. These functions are optimized using Numba's just-in-time compilation.
"""

import numpy as np
from numba import njit, jit, prange
from typing import List, Tuple, Dict, Any, Optional

# Constants for card representation
RANKS = 13  # Ace through King
SUITS = 4   # Clubs, Diamonds, Hearts, Spades


@njit
def card_to_int(rank: int, suit: int) -> int:
    """
    Convert a card's rank and suit to an integer representation.
    
    Args:
        rank (int): Card rank (0-12 for 2-A)
        suit (int): Card suit (0-3 for clubs, diamonds, hearts, spades)
    
    Returns:
        int: Integer representation of the card
    """
    return rank * SUITS + suit


@njit
def int_to_card(card_int: int) -> Tuple[int, int]:
    """
    Convert an integer card representation back to rank and suit.
    
    Args:
        card_int (int): Integer representation of a card
    
    Returns:
        Tuple[int, int]: (rank, suit)
    """
    rank = card_int // SUITS
    suit = card_int % SUITS
    return rank, suit


@njit
def evaluate_hand_fast(hole_cards: np.ndarray, community_cards: np.ndarray) -> int:
    """
    Evaluate a poker hand quickly using Numba optimization.
    
    This function evaluates a Texas Hold'em hand (2 hole cards + up to 5 community cards)
    and returns a hand rank, where higher is better.
    
    Args:
        hole_cards (np.ndarray): Array of hole card integers (2 cards)
        community_cards (np.ndarray): Array of community card integers (up to 5 cards)
    
    Returns:
        int: Hand rank value, higher is better
    """
    # Combine hole cards and community cards
    all_cards = np.concatenate((hole_cards, community_cards))
    
    # Count ranks and suits
    rank_count = np.zeros(RANKS, dtype=np.int32)
    suit_count = np.zeros(SUITS, dtype=np.int32)
    
    for card in all_cards:
        rank, suit = int_to_card(card)
        rank_count[rank] += 1
        suit_count[suit] += 1
    
    # Check for flush
    flush_suit = -1
    for suit in range(SUITS):
        if suit_count[suit] >= 5:
            flush_suit = suit
            break
    
    # Check for straight
    straight_high = -1
    count = 0
    for rank in range(RANKS):
        if rank_count[rank] > 0:
            count += 1
            if count >= 5:
                straight_high = rank
        else:
            count = 0
    
    # Special case: A-5 straight (Ace can be low)
    if count == 4 and rank_count[12] > 0 and straight_high == 3:  # 12 is Ace, 3 is 5
        straight_high = 3  # 5-high straight
    
    # Check for straight flush
    straight_flush = False
    if flush_suit >= 0 and straight_high >= 0:
        # Check if there are 5 consecutive cards of the same suit
        flush_cards = []
        for card in all_cards:
            rank, suit = int_to_card(card)
            if suit == flush_suit:
                flush_cards.append(rank)
        
        flush_cards = np.sort(np.array(flush_cards))
        if len(flush_cards) >= 5:
            count = 1
            for i in range(1, len(flush_cards)):
                if flush_cards[i] == flush_cards[i-1] + 1:
                    count += 1
                    if count >= 5:
                        straight_flush = True
                        break
                else:
                    count = 1
            
            # Special case: A-5 straight flush
            if not straight_flush and 12 in flush_cards and 0 in flush_cards and 1 in flush_cards and 2 in flush_cards and 3 in flush_cards:
                straight_flush = True
    
    # Compute hand ranks based on poker hand hierarchy
    if straight_flush:
        return 8 * 10000000 + straight_high  # Straight flush
    
    # Count the number of pairs, trips, quads
    pairs = 0
    trips = 0
    quads = 0
    pair_ranks = []
    trip_ranks = []
    quad_ranks = []
    
    for rank in range(RANKS):
        if rank_count[rank] == 2:
            pairs += 1
            pair_ranks.append(rank)
        elif rank_count[rank] == 3:
            trips += 1
            trip_ranks.append(rank)
        elif rank_count[rank] == 4:
            quads += 1
            quad_ranks.append(rank)
    
    # Sort for kickers
    pair_ranks.sort(reverse=True)
    trip_ranks.sort(reverse=True)
    
    if quads > 0:
        return 7 * 10000000 + quad_ranks[0]  # Four of a kind
    
    if trips > 0 and pairs > 0:
        return 6 * 10000000 + trip_ranks[0] * 100 + pair_ranks[0]  # Full house
    
    if flush_suit >= 0:
        # Find the 5 highest cards in the flush suit
        flush_ranks = []
        for card in all_cards:
            rank, suit = int_to_card(card)
            if suit == flush_suit:
                flush_ranks.append(rank)
        
        flush_ranks.sort(reverse=True)
        flush_ranks = flush_ranks[:5]
        
        # Compute flush value based on the 5 highest cards
        flush_value = 0
        for i, rank in enumerate(flush_ranks):
            flush_value += rank * (100 ** (4 - i))
        
        return 5 * 10000000 + flush_value  # Flush
    
    if straight_high >= 0:
        return 4 * 10000000 + straight_high  # Straight
    
    if trips > 0:
        # Find kickers for three of a kind
        kickers = []
        for rank in range(RANKS-1, -1, -1):
            if rank_count[rank] > 0 and rank not in trip_ranks:
                kickers.append(rank)
                if len(kickers) == 2:
                    break
        
        return 3 * 10000000 + trip_ranks[0] * 10000 + kickers[0] * 100 + kickers[1]  # Three of a kind
    
    if pairs >= 2:
        # Find kicker for two pair
        kicker = -1
        for rank in range(RANKS-1, -1, -1):
            if rank_count[rank] > 0 and rank not in pair_ranks:
                kicker = rank
                break
        
        return 2 * 10000000 + pair_ranks[0] * 10000 + pair_ranks[1] * 100 + kicker  # Two pair
    
    if pairs == 1:
        # Find kickers for one pair
        kickers = []
        for rank in range(RANKS-1, -1, -1):
            if rank_count[rank] > 0 and rank not in pair_ranks:
                kickers.append(rank)
                if len(kickers) == 3:
                    break
        
        kicker_value = 0
        for i, rank in enumerate(kickers):
            kicker_value += rank * (100 ** (2 - i))
        
        return 1 * 10000000 + pair_ranks[0] * 1000000 + kicker_value  # One pair
    
    # High card
    high_cards = []
    for rank in range(RANKS-1, -1, -1):
        if rank_count[rank] > 0:
            high_cards.append(rank)
            if len(high_cards) == 5:
                break
    
    high_card_value = 0
    for i, rank in enumerate(high_cards):
        high_card_value += rank * (100 ** (4 - i))
    
    return high_card_value  # High card


@njit(parallel=True)
def monte_carlo_hand_equity(hole_cards: np.ndarray, community_cards: np.ndarray, 
                         num_players: int, num_simulations: int = 1000) -> float:
    """
    Estimate hand equity through Monte Carlo simulation.
    
    This function runs multiple simulations to estimate the probability of winning
    with a given hand against a specified number of opponents.
    
    Args:
        hole_cards (np.ndarray): Array of hole card integers (2 cards)
        community_cards (np.ndarray): Array of community card integers (up to 5 cards)
        num_players (int): Number of players in the hand
        num_simulations (int, optional): Number of simulations to run. Defaults to 1000.
    
    Returns:
        float: Estimated equity (probability of winning)
    """
    if len(community_cards) == 5:  # All community cards are out, no need to simulate
        return estimate_showdown_equity(hole_cards, community_cards, num_players)
    
    # Create a deck excluding known cards
    deck = []
    for rank in range(RANKS):
        for suit in range(SUITS):
            card = card_to_int(rank, suit)
            if card not in hole_cards and card not in community_cards:
                deck.append(card)
    
    deck = np.array(deck)
    
    # Run simulations
    wins = 0
    for sim in prange(num_simulations):
        np.random.shuffle(deck)
        
        # Deal cards to opponents
        used_cards = 0
        opponent_hole_cards = []
        for player in range(num_players - 1):  # Exclude hero
            opponent_hole_cards.append(deck[used_cards:used_cards+2])
            used_cards += 2
        
        # Deal remaining community cards
        remaining = 5 - len(community_cards)
        sim_community = np.concatenate((community_cards, deck[used_cards:used_cards+remaining]))
        
        # Evaluate hero's hand
        hero_rank = evaluate_hand_fast(hole_cards, sim_community)
        
        # Check if hero wins
        win = True
        for opp_cards in opponent_hole_cards:
            opp_rank = evaluate_hand_fast(opp_cards, sim_community)
            if opp_rank >= hero_rank:  # Tie or loss
                win = False
                break
        
        if win:
            wins += 1
    
    return wins / num_simulations


@njit
def estimate_showdown_equity(hole_cards: np.ndarray, community_cards: np.ndarray, 
                          num_players: int, num_samples: int = 100) -> float:
    """
    Estimate showdown equity when all community cards are known.
    
    This function samples possible opponent hands and calculates win probability.
    
    Args:
        hole_cards (np.ndarray): Array of hole card integers (2 cards)
        community_cards (np.ndarray): Array of community card integers (5 cards)
        num_players (int): Number of players in the hand
        num_samples (int, optional): Number of opponent hand combinations to sample. 
                                   Defaults to 100.
    
    Returns:
        float: Estimated equity (probability of winning)
    """
    # Create a deck excluding known cards
    deck = []
    for rank in range(RANKS):
        for suit in range(SUITS):
            card = card_to_int(rank, suit)
            if card not in hole_cards and card not in community_cards:
                deck.append(card)
    
    deck = np.array(deck)
    
    # Evaluate hero's hand
    hero_rank = evaluate_hand_fast(hole_cards, community_cards)
    
    # Sample possible opponent hands
    wins = 0
    for _ in range(num_samples):
        np.random.shuffle(deck)
        
        # Simulate opponents
        win = True
        used_cards = 0
        
        for _ in range(num_players - 1):  # Exclude hero
            opp_cards = deck[used_cards:used_cards+2]
            used_cards += 2
            
            opp_rank = evaluate_hand_fast(opp_cards, community_cards)
            if opp_rank >= hero_rank:  # Tie or loss
                win = False
                break
        
        if win:
            wins += 1
    
    return wins / num_samples


@jit
def optimal_strategy_icm(stacks: np.ndarray, payouts: np.ndarray, 
                       small_blind: int, big_blind: int) -> np.ndarray:
    """
    Calculate an optimal push/fold strategy using the Independent Chip Model (ICM).
    
    This function computes push/fold ranges for Spin & Go tournaments based on ICM equity.
    
    Args:
        stacks (np.ndarray): Array of player stack sizes
        payouts (np.ndarray): Array of tournament payouts
        small_blind (int): Small blind amount
        big_blind (int): Big blind amount
    
    Returns:
        np.ndarray: ICM push/fold thresholds for each player position
    """
    # Calculate ICM equity for each player
    total_chips = np.sum(stacks)
    equity_before = calculate_icm_equity(stacks, payouts, total_chips)
    
    # Calculate push/fold thresholds
    num_players = len(stacks)
    thresholds = np.zeros(num_players)
    
    for position in range(num_players):
        # Skip players who are already all-in
        if stacks[position] <= 0:
            thresholds[position] = 1.0  # Will never push
            continue
        
        # Simulate different push scenarios
        push_eq = np.zeros(169)  # 13*13 possible starting hands
        
        for hand_idx in range(169):
            # Simulate pushing with this hand
            new_stacks = stacks.copy()
            
            # Calculate chance of winning all-in
            # This is a simplified model - in reality, we'd use actual hand equities
            # We're using a simple approximation here based on hand ranking
            win_prob = (169 - hand_idx) / 169  # Higher ranking hands have higher probability
            
            # Simulate outcomes
            equity_sum = 0
            
            # Win scenario
            if win_prob > 0:
                new_stacks_win = new_stacks.copy()
                new_stacks_win[position] += big_blind  # Win blinds
                equity_win = calculate_icm_equity(new_stacks_win, payouts, total_chips)
                equity_sum += win_prob * equity_win[position]
            
            # Loss scenario
            if win_prob < 1:
                new_stacks_lose = new_stacks.copy()
                new_stacks_lose[position] = 0  # Bust
                equity_lose = calculate_icm_equity(new_stacks_lose, payouts, total_chips)
                equity_sum += (1 - win_prob) * equity_lose[position]
            
            # Store EV
            push_eq[hand_idx] = equity_sum
        
        # Find threshold where pushing becomes +EV compared to folding
        fold_equity = equity_before[position]
        threshold_idx = 0
        
        for hand_idx in range(169):
            if push_eq[hand_idx] > fold_equity:
                threshold_idx = hand_idx
                break
        
        # Convert to a percentage
        thresholds[position] = threshold_idx / 169
    
    return thresholds


@njit
def calculate_icm_equity(stacks: np.ndarray, payouts: np.ndarray, total_chips: int) -> np.ndarray:
    """
    Calculate ICM (Independent Chip Model) equity for tournament players.
    
    Args:
        stacks (np.ndarray): Array of player stack sizes
        payouts (np.ndarray): Array of tournament payouts (prize pool distribution)
        total_chips (int): Total chips in play
    
    Returns:
        np.ndarray: ICM equity for each player (dollar value)
    """
    num_players = len(stacks)
    equity = np.zeros(num_players)
    
    # If only one player left, they get all the money
    if np.sum(stacks > 0) == 1:
        for i in range(num_players):
            if stacks[i] > 0:
                equity[i] = np.sum(payouts)
                return equity
    
    # Calculate probability of finishing in each position
    finish_prob = np.zeros((num_players, len(payouts)))
    
    for player in range(num_players):
        if stacks[player] <= 0:
            continue
            
        # Probability of finishing 1st
        finish_prob[player, 0] = stacks[player] / total_chips
        
        # Recursive calculation for other positions
        for position in range(1, len(payouts)):
            # Calculate probability of finishing in position for each stack size
            prob_sum = 0
            for other_player in range(num_players):
                if other_player != player and stacks[other_player] > 0:
                    # Probability of other player finishing first, then this player finishing position
                    prob_first = stacks[other_player] / total_chips
                    
                    # Calculate remaining stack probabilities
                    new_stacks = stacks.copy()
                    new_stacks[other_player] = 0
                    new_total = total_chips - stacks[other_player]
                    
                    if new_total > 0:
                        prob_position = stacks[player] / new_total
                        prob_sum += prob_first * prob_position
            
            finish_prob[player, position] = prob_sum
    
    # Calculate equity from finish probabilities
    for player in range(num_players):
        for position in range(len(payouts)):
            equity[player] += finish_prob[player, position] * payouts[position]
    
    return equity