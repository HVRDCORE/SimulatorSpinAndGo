"""
Implementation of a call agent for poker simulations.
"""

from typing import List, Dict, Any

from pokersim.agents.base_agent import Agent
from pokersim.game.state import GameState, Action, ActionType


class CallAgent(Agent):
    """
    An agent that always calls the current bet.
    
    Attributes:
        player_id (int): The ID of the player controlled by this agent.
    """
    
    def act(self, game_state: GameState) -> Action:
        """
        Choose an action based on the current game state.
        
        Args:
            game_state (GameState): The current game state.
            
        Returns:
            Action: The chosen action.
        """
        legal_actions = game_state.get_legal_actions()
        if not legal_actions:
            raise ValueError("No legal actions available")
        
        # Prefer check over call
        for action in legal_actions:
            if action.action_type == ActionType.CHECK:
                return action
        
        # Prefer call over other actions
        for action in legal_actions:
            if action.action_type == ActionType.CALL:
                return action
        
        # If can't check or call, choose the first legal action (likely fold)
        return legal_actions[0]
