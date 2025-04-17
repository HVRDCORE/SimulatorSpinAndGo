"""
CUDA optimized kernels for poker hand evaluation and simulation.

This module provides custom CUDA kernels for accelerating poker computations
using PyTorch's CUDA capabilities. These kernels are optimized for parallel
hand evaluation, equity calculation, and Monte Carlo simulations.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging

# Настройка логирования
logger = logging.getLogger("pokersim.gpu_kernels")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Constants for card representation
RANKS = 13  # Ace through King
SUITS = 4   # Clubs, Diamonds, Hearts, Spades


class CUDAHandEvaluator:
    """
    CUDA-optimized hand evaluator for poker.

    This class leverages GPU parallelism to evaluate multiple poker hands
    simultaneously, providing significant speedup over CPU evaluation.

    Attributes:
        device (torch.device): GPU device to use for evaluation.
        batch_size (int): Batch size for evaluation.
    """

    def __init__(self, device: Optional[torch.device] = None, batch_size: int = 1024):
        """
        Initialize CUDA hand evaluator.

        Args:
            device (Optional[torch.device], optional): GPU device. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 1024.
        """
        # Use default GPU if none specified
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # Check if GPU is available
        if not torch.cuda.is_available() and self.device.type == "cuda":
            raise RuntimeError("CUDA is not available, but a CUDA device was requested.")

        # Pre-compute rank and suit masks for fast evaluation
        self._init_masks()
        logger.info(f"Initialized CUDAHandEvaluator with device: {self.device}, batch_size: {batch_size}")

    def _init_masks(self):
        """Initialize bit masks for ranks and suits."""
        # Create bit masks for each rank (2-A) and suit (clubs, diamonds, hearts, spades)
        # These will be used for fast hand evaluation

        # Rank masks: each bit represents a card of that rank
        self.rank_masks = torch.zeros(RANKS, dtype=torch.int64, device=self.device)

        # Suit masks: each bit represents a card of that suit
        self.suit_masks = torch.zeros(SUITS, dtype=torch.int64, device=self.device)

        # Populate masks
        for rank in range(RANKS):
            for suit in range(SUITS):
                card_bit = 1 << (rank * SUITS + suit)
                self.rank_masks[rank] |= card_bit
                self.suit_masks[suit] |= card_bit

    def evaluate_hands(self, hole_cards: torch.Tensor, community_cards: torch.Tensor) -> torch.Tensor:
        """
        Evaluate multiple poker hands in parallel using CUDA.

        Args:
            hole_cards (torch.Tensor): Tensor of hole card pairs, shape (N, 2).
            community_cards (torch.Tensor): Tensor of community cards, shape (5,).

        Returns:
            torch.Tensor: Hand strength values, shape (N,).
        """
        # Move tensors to device if needed
        hole_cards = hole_cards.to(self.device)
        community_cards = community_cards.to(self.device)

        # Process in batches for memory efficiency
        num_hands = hole_cards.shape[0]
        results = torch.zeros(num_hands, dtype=torch.int32, device=self.device)

        logger.debug(f"Evaluating {num_hands} hands in batches of {self.batch_size}")
        for i in range(0, num_hands, self.batch_size):
            end_idx = min(i + self.batch_size, num_hands)
            batch_hole_cards = hole_cards[i:end_idx]

            # Evaluate batch of hands
            batch_results = self._evaluate_batch(batch_hole_cards, community_cards)
            results[i:end_idx] = batch_results
            logger.debug(f"Processed batch {i//self.batch_size + 1}, size: {end_idx - i}")

        return results

    def _evaluate_batch(self, hole_cards: torch.Tensor, community_cards: torch.Tensor) -> torch.Tensor:
        """
        Evaluate a batch of poker hands using GPU parallelism.

        Args:
            hole_cards (torch.Tensor): Batch of hole card pairs.
            community_cards (torch.Tensor): Community cards.

        Returns:
            torch.Tensor: Hand strength values.
        """
        batch_size = hole_cards.shape[0]

        # Combine hole cards and community cards for each hand in the batch
        all_cards = torch.zeros((batch_size, 7), dtype=torch.int32, device=self.device)
        all_cards[:, :2] = hole_cards  # First 2 cards are hole cards
        all_cards[:, 2:2+community_cards.shape[0]] = community_cards.unsqueeze(0).expand(batch_size, -1)

        # Compute hand values in parallel
        hand_values = self._compute_hand_values(all_cards)

        return hand_values

    def _compute_hand_values(self, cards: torch.Tensor) -> torch.Tensor:
        """
        Compute hand values for multiple hands in parallel.

        Args:
            cards (torch.Tensor): Tensor of card values, shape (batch_size, 7).

        Returns:
            torch.Tensor: Hand strength values, shape (batch_size,).
        """
        batch_size = cards.shape[0]
        hand_values = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        rank_counts = torch.zeros((batch_size, RANKS), dtype=torch.int8, device=self.device)
        suit_counts = torch.zeros((batch_size, SUITS), dtype=torch.int8, device=self.device)

        # Count ranks and suits
        for batch_idx in range(batch_size):
            for card_idx in range(cards.shape[1]):
                if cards[batch_idx, card_idx] >= 0:
                    card = cards[batch_idx, card_idx].item()
                    rank = card // SUITS
                    suit = card % SUITS
                    rank_counts[batch_idx, rank] += 1
                    suit_counts[batch_idx, suit] += 1

        # Check for flush
        has_flush = (suit_counts >= 5).any(dim=1)
        flush_suits = torch.argmax(suit_counts, dim=1)

        # Check for hands from highest to lowest
        for batch_idx in range(batch_size):
            if has_flush[batch_idx]:
                flush_suit = flush_suits[batch_idx].item()
                flush_ranks = []
                for card_idx in range(cards.shape[1]):
                    if cards[batch_idx, card_idx] >= 0 and cards[batch_idx, card_idx].item() % SUITS == flush_suit:
                        flush_ranks.append(cards[batch_idx, card_idx].item() // SUITS)
                flush_ranks.sort(reverse=True)
                if len(flush_ranks) >= 5:
                    # Check for straight flush
                    for i in range(len(flush_ranks) - 4):
                        if flush_ranks[i] - flush_ranks[i + 4] == 4:
                            hand_values[batch_idx] = 8000000 + flush_ranks[i] * 10000  # Straight flush
                            logger.debug(f"Batch {batch_idx}: Straight flush, value={hand_values[batch_idx]}")
                            break
                    if hand_values[batch_idx] == 0:
                        flush_value = sum(flush_ranks[i] * (100 ** (4 - i)) for i in range(5))
                        hand_values[batch_idx] = 5000000 + flush_value  # Flush
                        logger.debug(f"Batch {batch_idx}: Flush, value={hand_values[batch_idx]}")

            # Four of a kind
            if hand_values[batch_idx] == 0 and (rank_counts[batch_idx] == 4).any():
                quad_rank = torch.argmax((rank_counts[batch_idx] == 4).float()).item()
                kicker = torch.max(torch.where(rank_counts[batch_idx] == 1)[0]) if (rank_counts[batch_idx] == 1).any() else 0
                hand_values[batch_idx] = 7000000 + quad_rank * 10000 + kicker
                logger.debug(f"Batch {batch_idx}: Four of a kind, value={hand_values[batch_idx]}")

            # Full house
            if hand_values[batch_idx] == 0 and (rank_counts[batch_idx] == 3).any() and ((rank_counts[batch_idx] == 2).any() or (rank_counts[batch_idx] == 3).sum() >= 2):
                trips_rank = torch.argmax((rank_counts[batch_idx] == 3).float()).item()
                pair_rank = torch.argmax((rank_counts[batch_idx] == 2).float()).item() if (rank_counts[batch_idx] == 2).any() else torch.argmax((rank_counts[batch_idx] == 3).float(), dim=0, keepdim=True)[0].item()
                hand_values[batch_idx] = 6000000 + trips_rank * 10000 + pair_rank
                logger.debug(f"Batch {batch_idx}: Full house, value={hand_values[batch_idx]}")

            # Straight
            if hand_values[batch_idx] == 0:
                ranks = torch.where(rank_counts[batch_idx] > 0)[0]
                for i in range(len(ranks) - 4):
                    if ranks[i] - ranks[i + 4] == 4:
                        hand_values[batch_idx] = 4000000 + ranks[i] * 10000
                        logger.debug(f"Batch {batch_idx}: Straight, value={hand_values[batch_idx]}")
                        break
                # Check Ace-low straight
                if hand_values[batch_idx] == 0 and all(r in ranks for r in [0, 1, 2, 3, 12]):  # 2-5 + Ace
                    hand_values[batch_idx] = 4000000 + 3 * 10000  # 5-high
                    logger.debug(f"Batch {batch_idx}: Ace-low straight, value={hand_values[batch_idx]}")

            # Three of a kind
            if hand_values[batch_idx] == 0 and (rank_counts[batch_idx] == 3).any():
                trips_rank = torch.argmax((rank_counts[batch_idx] == 3).float()).item()
                kickers = torch.sort(torch.where(rank_counts[batch_idx] == 1)[0], descending=True)[0][:2]
                kicker_value = sum(kickers[i].item() * (100 ** (1 - i)) for i in range(len(kickers)))
                hand_values[batch_idx] = 3000000 + trips_rank * 10000 + kicker_value
                logger.debug(f"Batch {batch_idx}: Three of a kind, value={hand_values[batch_idx]}")

            # Two pair
            if hand_values[batch_idx] == 0 and (rank_counts[batch_idx] == 2).sum() >= 2:
                pairs = torch.sort(torch.where(rank_counts[batch_idx] == 2)[0], descending=True)[0][:2]
                kicker = torch.max(torch.where(rank_counts[batch_idx] == 1)[0]) if (rank_counts[batch_idx] == 1).any() else 0
                hand_values[batch_idx] = 2000000 + pairs[0].item() * 10000 + pairs[1].item() * 100 + kicker
                logger.debug(f"Batch {batch_idx}: Two pair, value={hand_values[batch_idx]}")

            # One pair
            if hand_values[batch_idx] == 0 and (rank_counts[batch_idx] == 2).any():
                pair_rank = torch.argmax((rank_counts[batch_idx] == 2).float()).item()
                kickers = torch.sort(torch.where(rank_counts[batch_idx] == 1)[0], descending=True)[0][:3]
                kicker_value = sum(kickers[i].item() * (100 ** (2 - i)) for i in range(len(kickers)))
                hand_values[batch_idx] = 1000000 + pair_rank * 10000 + kicker_value
                logger.debug(f"Batch {batch_idx}: One pair, value={hand_values[batch_idx]}")

            # High card
            if hand_values[batch_idx] == 0:
                high_cards = torch.sort(torch.where(rank_counts[batch_idx] > 0)[0], descending=True)[0][:5]
                hand_value = sum(high_cards[i].item() * (100 ** (4 - i)) for i in range(len(high_cards)))
                hand_values[batch_idx] = hand_value
                logger.debug(f"Batch {batch_idx}: High card, value={hand_values[batch_idx]}")

        return hand_values


class CUDAEquityCalculator:
    """
    CUDA-accelerated poker equity calculator.

    This class uses GPU parallelism to run Monte Carlo simulations
    for estimating hand equity in poker.

    Attributes:
        device (torch.device): GPU device for computations.
        evaluator (CUDAHandEvaluator): Hand evaluator for GPU.
        num_simulations (int): Number of simulations per calculation.
    """

    def __init__(self, device: Optional[torch.device] = None, num_simulations: int = 1000):
        """
        Initialize CUDA equity calculator.

        Args:
            device (Optional[torch.device], optional): GPU device. Defaults to None.
            num_simulations (int, optional): Number of simulations. Defaults to 1000.
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluator = CUDAHandEvaluator(device=self.device)
        self.num_simulations = num_simulations
        logger.info(f"Initialized CUDAEquityCalculator with {num_simulations} simulations")

    def calculate_equity(self, hole_cards: List[int], community_cards: List[int],
                        num_players: int) -> float:
        """
        Calculate equity for a hand using Monte Carlo simulation on GPU.

        Args:
            hole_cards (List[int]): Player's hole cards.
            community_cards (List[int]): Community cards (0-5 cards).
            num_players (int): Number of players in the hand.

        Returns:
            float: Estimated equity (probability of winning).
        """
        # Convert inputs to tensors
        hole_tensor = torch.tensor(hole_cards, dtype=torch.int32, device=self.device)
        community_tensor = torch.tensor(community_cards, dtype=torch.int32, device=self.device)

        # Pad community cards to length 5 if needed
        if len(community_cards) < 5:
            padding = torch.full((5 - len(community_cards),), -1, dtype=torch.int32, device=self.device)
            community_tensor = torch.cat([community_tensor, padding])

        # Generate deck excluding known cards
        deck = self._generate_deck(hole_cards, community_cards)

        # Run Monte Carlo simulations
        wins = 0

        logger.debug(f"Starting {self.num_simulations} Monte Carlo simulations")
        for i in range(self.num_simulations):
            logger.debug(f"Simulation {i+1}/{self.num_simulations}")
            # Shuffle deck
            shuffled_deck = deck[torch.randperm(len(deck), device=self.device)]

            # Deal cards to opponents and complete community cards
            opponents_hole_cards = []
            used_cards = 0

            # Deal to opponents
            for player in range(num_players - 1):
                opponent_cards = shuffled_deck[used_cards:used_cards+2]
                opponents_hole_cards.append(opponent_cards)
                used_cards += 2

            # Complete community cards if needed
            remaining_community = 5 - len(community_cards)
            if remaining_community > 0:
                additional_cards = shuffled_deck[used_cards:used_cards+remaining_community]
                full_community = torch.cat([community_tensor[community_tensor >= 0], additional_cards])
            else:
                full_community = community_tensor

            # Evaluate hands
            hero_batch = hole_tensor.unsqueeze(0)  # Shape: (1, 2)
            opponent_batch = torch.stack(opponents_hole_cards)  # Shape: (num_players-1, 2)
            all_hole_cards = torch.cat([hero_batch, opponent_batch])  # Shape: (num_players, 2)

            # Evaluate all hands
            hand_values = self.evaluator.evaluate_hands(all_hole_cards, full_community)

            # Check if hero wins
            hero_value = hand_values[0]
            win = True
            for opp_value in hand_values[1:]:
                if opp_value >= hero_value:  # Tie or loss
                    win = False
                    break

            if win:
                wins += 1

        # Return equity
        equity = wins / self.num_simulations
        logger.info(f"Calculated equity: {equity:.4f}")
        return equity

    def _generate_deck(self, hole_cards: List[int], community_cards: List[int]) -> torch.Tensor:
        """
        Generate a deck of cards excluding known cards.

        Args:
            hole_cards (List[int]): Player's hole cards.
            community_cards (List[int]): Community cards.

        Returns:
            torch.Tensor: Deck of available cards.
        """
        # Create full deck
        full_deck = torch.arange(RANKS * SUITS, dtype=torch.int32, device=self.device)

        # Create mask for used cards
        used_cards = torch.zeros(RANKS * SUITS, dtype=torch.bool, device=self.device)
        for card in hole_cards + community_cards:
            if card >= 0:  # Skip padding
                used_cards[card] = True

        # Filter out used cards
        available_deck = full_deck[~used_cards]
        logger.debug(f"Generated deck with {len(available_deck)} available cards")
        return available_deck


class CUDACardGenerator:
    """
    CUDA-optimized card generator for poker simulations.

    This class uses GPU to generate and shuffle cards efficiently
    for poker simulations.

    Attributes:
        device (torch.device): GPU device for computations.
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize CUDA card generator.

        Args:
            device (Optional[torch.device], optional): GPU device. Defaults to None.
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initialized CUDACardGenerator with device: {self.device}")

    def generate_hands(self, num_hands: int, num_players: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate multiple random poker hands and community cards.

        Args:
            num_hands (int): Number of hands to generate.
            num_players (int): Number of players per hand.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Hole cards and community cards tensors.
        """
        # Create full deck
        full_deck = torch.arange(RANKS * SUITS, dtype=torch.int32, device=self.device)

        # Initialize result tensors
        hole_cards = torch.zeros((num_hands, num_players, 2), dtype=torch.int32, device=self.device)
        community_cards = torch.zeros((num_hands, 5), dtype=torch.int32, device=self.device)

        # Generate hands in parallel
        for hand_idx in range(num_hands):
            # Shuffle deck
            shuffled_deck = full_deck[torch.randperm(len(full_deck), device=self.device)]

            # Deal hole cards
            for player_idx in range(num_players):
                hole_cards[hand_idx, player_idx, 0] = shuffled_deck[player_idx * 2]
                hole_cards[hand_idx, player_idx, 1] = shuffled_deck[player_idx * 2 + 1]

            # Deal community cards
            community_start_idx = num_players * 2
            community_cards[hand_idx] = shuffled_deck[community_start_idx:community_start_idx + 5]

        logger.info(f"Generated {num_hands} hands for {num_players} players")
        return hole_cards, community_cards

    def generate_random_hands(self, num_hands: int) -> torch.Tensor:
        """
        Generate random poker hands (5-card hands).

        Args:
            num_hands (int): Number of hands to generate.

        Returns:
            torch.Tensor: Tensor of random hands, shape (num_hands, 5).
        """
        # Create full deck
        full_deck = torch.arange(RANKS * SUITS, dtype=torch.int32, device=self.device)

        # Initialize result tensor
        hands = torch.zeros((num_hands, 5), dtype=torch.int32, device=self.device)

        # Generate hands
        for i in range(num_hands):
            # Shuffle deck
            shuffled = full_deck[torch.randperm(len(full_deck), device=self.device)]

            # Take first 5 cards
            hands[i] = shuffled[:5]

        logger.info(f"Generated {num_hands} random 5-card hands")
        return hands