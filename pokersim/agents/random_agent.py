"""
Implementation of a random agent for poker simulations.
"""

import random
from typing import List, Dict, Any, Optional

from pokersim.agents.base_agent import Agent
from pokersim.game.state import GameState, Action


class RandomAgent(Agent):
    """
    An agent that chooses actions uniformly at random from the legal actions.

    Attributes:
        player_id (int): The ID of the player controlled by this agent.
    """

    def __init__(self, player_id: int, seed: Optional[int] = None):
        """
        Initialize the random agent.

        Args:
            player_id (int): The ID of the player controlled by this agent.
            seed (Optional[int]): Seed for random number generator. Defaults to None.
        """
        super().__init__(player_id)
        if seed is not None:
            random.seed(seed)

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
        return random.choice(legal_actions)