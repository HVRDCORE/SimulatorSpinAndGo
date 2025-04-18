"""
PokerSim: A Python API for poker simulation with ML integration.

This package provides a comprehensive framework for simulating poker games,
implementing poker-playing agents, and training them using machine learning.
It is designed to be useful for reinforcement learning research.
"""

__version__ = "0.1.0"

from pokersim.game.card import Card, Suit, Rank
from pokersim.game.deck import Deck
from pokersim.game.state import GameState, Action, ActionType
from pokersim.game.evaluator import HandEvaluator

__all__ = [
    "Card", "Suit", "Rank", "Deck", "GameState", "Action", "ActionType", "HandEvaluator"
]
