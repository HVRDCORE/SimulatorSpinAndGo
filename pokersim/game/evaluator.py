"""
Implementation of hand evaluation for poker simulations.
"""

from typing import List, Tuple, Dict, Set
import numba

from pokersim.game.card import Card, Rank, Suit
from pokersim.utils.optimization import njit_if_available


class HandEvaluator:
    """
    Evaluator for poker hands.
    
    This class provides methods for evaluating the strength of poker hands.
    """
    
    @staticmethod
    @njit_if_available
    def evaluate_hand(hand: List[Card]) -> Tuple[int, str]:
        """
        Evaluate the strength of a poker hand.
        
        Args:
            hand (List[Card]): The cards in the hand.
            
        Returns:
            Tuple[int, str]: A tuple containing the hand rank (higher is better)
                and a description of the hand.
        """
        if len(hand) < 5:
            raise ValueError("Hand must contain at least 5 cards.")
        
        # Convert cards to integers for faster evaluation
        card_ints = [card.to_int() for card in hand]
        
        # Check for different hand types in descending order of strength
        # Royal Flush
        royal_flush = HandEvaluator._is_royal_flush(hand)
        if royal_flush[0]:
            return (9, f"Royal Flush, {royal_flush[1]}")
        
        # Straight Flush
        straight_flush = HandEvaluator._is_straight_flush(hand)
        if straight_flush[0]:
            return (8, f"Straight Flush, {straight_flush[1]} high")
        
        # Four of a Kind
        four_of_a_kind = HandEvaluator._is_four_of_a_kind(hand)
        if four_of_a_kind[0]:
            return (7, f"Four of a Kind, {four_of_a_kind[1]}s")
        
        # Full House
        full_house = HandEvaluator._is_full_house(hand)
        if full_house[0]:
            return (6, f"Full House, {full_house[1]}s over {full_house[2]}s")
        
        # Flush
        flush = HandEvaluator._is_flush(hand)
        if flush[0]:
            return (5, f"Flush, {flush[1]} high")
        
        # Straight
        straight = HandEvaluator._is_straight(hand)
        if straight[0]:
            return (4, f"Straight, {straight[1]} high")
        
        # Three of a Kind
        three_of_a_kind = HandEvaluator._is_three_of_a_kind(hand)
        if three_of_a_kind[0]:
            return (3, f"Three of a Kind, {three_of_a_kind[1]}s")
        
        # Two Pair
        two_pair = HandEvaluator._is_two_pair(hand)
        if two_pair[0]:
            return (2, f"Two Pair, {two_pair[1]}s and {two_pair[2]}s")
        
        # One Pair
        one_pair = HandEvaluator._is_one_pair(hand)
        if one_pair[0]:
            return (1, f"One Pair, {one_pair[1]}s")
        
        # High Card
        high_card = HandEvaluator._get_high_card(hand)
        return (0, f"High Card, {high_card}")
    
    @staticmethod
    def _is_royal_flush(hand: List[Card]) -> Tuple[bool, str]:
        """Check if the hand is a royal flush."""
        if not HandEvaluator._is_flush(hand)[0]:
            return (False, "")
        
        if not all(card.rank in {Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE} for card in hand):
            return (False, "")
        
        if not HandEvaluator._is_straight(hand)[0]:
            return (False, "")
        
        return (True, hand[0].suit.name.capitalize())
    
    @staticmethod
    def _is_straight_flush(hand: List[Card]) -> Tuple[bool, str]:
        """Check if the hand is a straight flush."""
        if not HandEvaluator._is_flush(hand)[0]:
            return (False, "")
        
        straight = HandEvaluator._is_straight(hand)
        if not straight[0]:
            return (False, "")
        
        return (True, straight[1])
    
    @staticmethod
    def _is_four_of_a_kind(hand: List[Card]) -> Tuple[bool, str]:
        """Check if the hand contains four of a kind."""
        ranks = [card.rank for card in hand]
        for rank in set(ranks):
            if ranks.count(rank) >= 4:
                return (True, rank.name.capitalize())
        
        return (False, "")
    
    @staticmethod
    def _is_full_house(hand: List[Card]) -> Tuple[bool, str, str]:
        """Check if the hand is a full house."""
        ranks = [card.rank for card in hand]
        three_of_a_kind = None
        pair = None
        
        for rank in set(ranks):
            count = ranks.count(rank)
            if count >= 3 and three_of_a_kind is None:
                three_of_a_kind = rank
            elif count >= 2 and pair is None:
                pair = rank
        
        if three_of_a_kind is not None and pair is not None:
            return (True, three_of_a_kind.name.capitalize(), pair.name.capitalize())
        
        return (False, "", "")
    
    @staticmethod
    def _is_flush(hand: List[Card]) -> Tuple[bool, str]:
        """Check if the hand is a flush."""
        suits = [card.suit for card in hand]
        for suit in set(suits):
            if suits.count(suit) >= 5:
                flush_cards = [card for card in hand if card.suit == suit]
                flush_cards.sort(key=lambda card: card.rank.value, reverse=True)
                return (True, flush_cards[0].rank.name.capitalize())
        
        return (False, "")
    
    @staticmethod
    def _is_straight(hand: List[Card]) -> Tuple[bool, str]:
        """Check if the hand is a straight."""
        ranks = sorted({card.rank.value for card in hand}, reverse=True)
        
        # Check for A-5 straight
        if set(ranks).issuperset({14, 5, 4, 3, 2}):
            return (True, "Five")
        
        # Check for normal straight
        for i in range(len(ranks) - 4):
            if ranks[i] - ranks[i+4] == 4:
                return (True, Rank(ranks[i]).name.capitalize())
        
        return (False, "")
    
    @staticmethod
    def _is_three_of_a_kind(hand: List[Card]) -> Tuple[bool, str]:
        """Check if the hand contains three of a kind."""
        ranks = [card.rank for card in hand]
        for rank in set(ranks):
            if ranks.count(rank) >= 3:
                return (True, rank.name.capitalize())
        
        return (False, "")
    
    @staticmethod
    def _is_two_pair(hand: List[Card]) -> Tuple[bool, str, str]:
        """Check if the hand contains two pairs."""
        ranks = [card.rank for card in hand]
        pairs = []
        
        for rank in set(ranks):
            if ranks.count(rank) >= 2:
                pairs.append(rank)
        
        if len(pairs) >= 2:
            pairs.sort(key=lambda r: r.value, reverse=True)
            return (True, pairs[0].name.capitalize(), pairs[1].name.capitalize())
        
        return (False, "", "")
    
    @staticmethod
    def _is_one_pair(hand: List[Card]) -> Tuple[bool, str]:
        """Check if the hand contains a pair."""
        ranks = [card.rank for card in hand]
        for rank in set(ranks):
            if ranks.count(rank) >= 2:
                return (True, rank.name.capitalize())
        
        return (False, "")
    
    @staticmethod
    def _get_high_card(hand: List[Card]) -> str:
        """Get the highest card in the hand."""
        return max(hand, key=lambda card: card.rank.value).rank.name.capitalize()
    
    @staticmethod
    def compare_hands(hand1: List[Card], hand2: List[Card]) -> int:
        """
        Compare two poker hands.
        
        Args:
            hand1 (List[Card]): The first hand.
            hand2 (List[Card]): The second hand.
            
        Returns:
            int: 1 if hand1 is better, -1 if hand2 is better, 0 if they are equal.
        """
        eval1 = HandEvaluator.evaluate_hand(hand1)
        eval2 = HandEvaluator.evaluate_hand(hand2)
        
        if eval1[0] > eval2[0]:
            return 1
        elif eval1[0] < eval2[0]:
            return -1
        else:
            # If the hand types are the same, compare kickers
            return HandEvaluator._compare_kickers(hand1, hand2)
    
    @staticmethod
    def _compare_kickers(hand1: List[Card], hand2: List[Card]) -> int:
        """
        Compare the kickers of two hands.
        
        Args:
            hand1 (List[Card]): The first hand.
            hand2 (List[Card]): The second hand.
            
        Returns:
            int: 1 if hand1 is better, -1 if hand2 is better, 0 if they are equal.
        """
        ranks1 = sorted([card.rank.value for card in hand1], reverse=True)
        ranks2 = sorted([card.rank.value for card in hand2], reverse=True)
        
        for r1, r2 in zip(ranks1, ranks2):
            if r1 > r2:
                return 1
            elif r1 < r2:
                return -1
        
        return 0
