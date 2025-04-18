"""
Base agent class for poker-playing agents.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from pokersim.game.state import GameState, Action


class Agent(ABC):
    """
    Base class for poker-playing agents.
    
    Attributes:
        player_id (int): The ID of the player controlled by this agent.
    """
    
    def __init__(self, player_id: int):
        """
        Initialize an agent.
        
        Args:
            player_id (int): The ID of the player controlled by this agent.
        """
        self.player_id = player_id
    
    @abstractmethod
    def act(self, game_state: GameState) -> Action:
        """
        Choose an action based on the current game state.
        
        Args:
            game_state (GameState): The current game state.
            
        Returns:
            Action: The chosen action.
        """
        pass
    
    def observe(self, game_state: GameState) -> None:
        """
        Update the agent's internal state based on the current game state.
        
        Args:
            game_state (GameState): The current game state.
        """
        pass
    
    def reset(self) -> None:
        """Reset the agent's internal state."""
        pass
    
    def end_hand(self, game_state: GameState) -> None:
        """
        Update the agent's internal state at the end of a hand.
        
        Args:
            game_state (GameState): The final game state.
        """
        pass
