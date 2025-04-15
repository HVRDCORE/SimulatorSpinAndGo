"""
Implementation of a deck of cards for poker simulations.
"""

import random
from typing import List, Optional
import numba

from pokersim.game.card import Card, Rank, Suit


class Deck:
    """
    A deck of playing cards.
    
    Attributes:
        cards (List[Card]): The cards in the deck.
    """
    
    def __init__(self, shuffle: bool = True):
        """
        Initialize a deck of cards.
        
        Args:
            shuffle (bool, optional): Whether to shuffle the deck. Defaults to True.
        """
        self.cards = [Card(rank, suit) for rank in Rank for suit in Suit]
        if shuffle:
            self.shuffle()
    
    def shuffle(self) -> None:
        """Shuffle the deck."""
        random.shuffle(self.cards)
    
    def deal(self, n: int = 1) -> List[Card]:
        """
        Deal cards from the deck.
        
        Args:
            n (int, optional): The number of cards to deal. Defaults to 1.
            
        Returns:
            List[Card]: The dealt cards.
            
        Raises:
            ValueError: If there are not enough cards in the deck.
        """
        if n > len(self.cards):
            raise ValueError(f"Cannot deal {n} cards from a deck of {len(self.cards)} cards.")
        
        dealt_cards = self.cards[-n:]
        self.cards = self.cards[:-n]
        return dealt_cards
    
    def deal_one(self) -> Card:
        """
        Deal a single card from the deck.
        
        Returns:
            Card: The dealt card.
            
        Raises:
            ValueError: If there are no cards in the deck.
        """
        if not self.cards:
            raise ValueError("Cannot deal from an empty deck.")
        
        return self.cards.pop()
    
    def __len__(self) -> int:
        """Return the number of cards in the deck."""
        return len(self.cards)
    
    def __str__(self) -> str:
        """Return a string representation of the deck."""
        return f"Deck of {len(self.cards)} cards"
    
    def __repr__(self) -> str:
        """Return a string representation of the deck."""
        return str(self)
