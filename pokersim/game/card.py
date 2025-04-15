"""
Implementation of playing cards for poker simulations.
"""

from enum import Enum, auto
from typing import List, Set, Dict, Tuple
import numba


class Suit(Enum):
    """Enumeration of card suits."""
    CLUBS = auto()
    DIAMONDS = auto()
    HEARTS = auto()
    SPADES = auto()
    
    def __str__(self) -> str:
        return self.name.capitalize()
    
    def __repr__(self) -> str:
        return str(self)
    
    @property
    def symbol(self) -> str:
        """Return the Unicode symbol for the suit."""
        symbols = {
            Suit.CLUBS: "♣",
            Suit.DIAMONDS: "♦",
            Suit.HEARTS: "♥",
            Suit.SPADES: "♠"
        }
        return symbols[self]


class Rank(Enum):
    """Enumeration of card ranks."""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    
    def __str__(self) -> str:
        if self.value <= 10:
            return str(self.value)
        return self.name[0]
    
    def __repr__(self) -> str:
        return str(self)


class Card:
    """
    A playing card with a rank and suit.
    
    Attributes:
        rank (Rank): The rank of the card.
        suit (Suit): The suit of the card.
    """
    
    def __init__(self, rank: Rank, suit: Suit):
        """
        Initialize a card with a rank and suit.
        
        Args:
            rank (Rank): The rank of the card.
            suit (Suit): The suit of the card.
        """
        self.rank = rank
        self.suit = suit
    
    def __str__(self) -> str:
        """Return a string representation of the card."""
        return f"{self.rank}{self.suit.symbol}"
    
    def __repr__(self) -> str:
        """Return a string representation of the card."""
        return str(self)
    
    def __eq__(self, other) -> bool:
        """Check if two cards are equal."""
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self) -> int:
        """Return a hash of the card."""
        return hash((self.rank, self.suit))
    
    def to_int(self) -> int:
        """
        Convert the card to an integer representation.
        
        Returns:
            int: An integer representation of the card.
        """
        suit_value = {
            Suit.CLUBS: 0,
            Suit.DIAMONDS: 1,
            Suit.HEARTS: 2,
            Suit.SPADES: 3
        }[self.suit]
        
        return (self.rank.value - 2) + suit_value * 13
    
    @staticmethod
    def from_int(value: int) -> 'Card':
        """
        Create a card from an integer representation.
        
        Args:
            value (int): An integer representation of a card.
            
        Returns:
            Card: The corresponding card.
        """
        rank_value = (value % 13) + 2
        suit_value = value // 13
        
        rank = next(r for r in Rank if r.value == rank_value)
        suit = {
            0: Suit.CLUBS,
            1: Suit.DIAMONDS,
            2: Suit.HEARTS,
            3: Suit.SPADES
        }[suit_value]
        
        return Card(rank, suit)
