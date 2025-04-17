"""
Implementation of Spin and Go game format for poker simulations.

Spin and Go is a popular poker format with the following characteristics:
- 3 players
- Winner-takes-all prize structure
- Randomly determined prize pool multiplier
- Fast blinds structure
- Starting stacks typically 25-50 big blinds
"""

import random
import logging
from typing import List, Dict, Tuple, Optional, Any

from pokersim.game.state import GameState, Action, ActionType, Stage

logger = logging.getLogger(__name__)

class SpinGoGame:
    """
    Spin and Go game format simulator.

    Attributes:
        num_players (int): Number of players (typically 3).
        buy_in (int): The buy-in amount.
        multiplier (float): Random prize pool multiplier.
        prize_pool (int): Total prize pool.
        starting_stack (int): Starting chip stack for each player.
        current_blinds (Tuple[int, int]): Current small and big blind levels.
        blind_levels (List[Tuple[int, int]]): Schedule of blind levels.
        blind_level_duration (int): Number of hands before blind increase.
        hands_played (int): Number of hands played so far.
        remaining_players (List[int]): IDs of players still in the tournament.
        eliminated_players (List[int]): IDs of eliminated players.
        player_stacks (List[int]): Chip stacks of each player.
    """

    MULTIPLIER_TABLE = {
        1.5: 0.7499,
        3: 0.15,
        5: 0.07,
        10: 0.02,
        100: 0.01,
        1000: 0.0001
    }

    def __init__(self, buy_in: int = 10, num_players: int = 3, starting_stack: int = 500,
                 blind_schedule: Optional[List[Tuple[int, int]]] = None,
                 blind_level_duration: int = 5):
        """
        Initialize a Spin and Go game.

        Args:
            buy_in (int): The buy-in amount. Defaults to 10.
            num_players (int): Number of players. Defaults to 3.
            starting_stack (int): Starting chip stack per player. Defaults to 500.
            blind_schedule (Optional[List[Tuple[int, int]]]): Blind level schedule.
            blind_level_duration (int): Hands per blind level. Defaults to 5.
        """
        self.num_players = num_players
        self.buy_in = buy_in
        self.multiplier = self._determine_multiplier()
        self.prize_pool = int(buy_in * num_players * self.multiplier)
        self.starting_stack = starting_stack

        if blind_schedule is None:
            blind_schedule = [
                (5, 10), (10, 20), (15, 30), (20, 40), (30, 60),
                (50, 100), (75, 150), (100, 200), (150, 300), (200, 400)
            ]
        self.blind_levels = blind_schedule
        self.current_blinds = self.blind_levels[0]
        self.blind_level_duration = blind_level_duration
        self.hands_played = 0
        self.remaining_players = list(range(num_players))
        self.eliminated_players = []
        self.player_stacks = [starting_stack] * num_players
        self.current_game = None
        self.button_position = 0
        logger.info(f"Initialized SpinGoGame with {num_players} players, prize pool {self.prize_pool}")

    def _determine_multiplier(self) -> float:
        """
        Determine the prize pool multiplier based on probabilities.

        Returns:
            float: The selected multiplier.
        """
        multipliers = list(self.MULTIPLIER_TABLE.keys())
        probabilities = list(self.MULTIPLIER_TABLE.values())
        return random.choices(multipliers, weights=probabilities, k=1)[0]

    def _update_blinds(self) -> None:
        """
        Update the blind levels based on the number of hands played.
        """
        if self.hands_played % self.blind_level_duration == 0:
            level_idx = min(self.hands_played // self.blind_level_duration, len(self.blind_levels) - 1)
            self.current_blinds = self.blind_levels[level_idx]
            logger.debug(f"Blinds updated to {self.current_blinds}")

    def start_new_hand(self) -> GameState:
        """
        Start a new hand in the tournament.

        Returns:
            GameState: The initial state of the new hand.

        Raises:
            ValueError: If there are fewer than 2 players to start a hand.
        """
        if len(self.remaining_players) < 2:
            raise ValueError("Cannot start new hand with fewer than 2 players")

        self._update_blinds()
        self.button_position = (self.button_position + 1) % len(self.remaining_players)
        small_blind, big_blind = self.current_blinds
        active_players = self.remaining_players
        stacks = [self.player_stacks[p] for p in active_players]

        self.current_game = GameState(
            num_players=len(active_players),
            small_blind=small_blind,
            big_blind=big_blind,
            initial_stacks=stacks,
            button=self.button_position,
            player_ids=active_players
        )
        self.hands_played += 1
        logger.info(f"Started new hand #{self.hands_played}, blinds: {self.current_blinds}, button: {self.button_position}")
        return self.current_game

    def update_stacks_after_hand(self) -> None:
        """
        Update player stacks after a hand and handle eliminations.

        Raises:
            ValueError: If there is no completed game to process.
        """
        if self.current_game is None:
            logger.error("No game initialized")
            raise ValueError("No completed game to process")
        if not self.current_game.is_terminal():
            logger.error("Current game is not terminal")
            raise ValueError("No completed game to process")

        payouts = self.current_game.get_payouts()
        logger.info(f"Payouts: {payouts}")

        eliminated_this_hand = []
        for i, player_id in enumerate(self.current_game.player_ids):
            self.player_stacks[player_id] += payouts[i]
            logger.debug(f"Player {player_id} stack updated to {self.player_stacks[player_id]}")
            if self.player_stacks[player_id] <= 0:
                eliminated_this_hand.append(player_id)

        for player_id in eliminated_this_hand:
            if player_id in self.remaining_players:
                self.remaining_players.remove(player_id)
                self.eliminated_players.append(player_id)
                logger.info(f"Player {player_id} eliminated")

        self.current_game = None  # Reset game state
        logger.debug(f"Active players: {self.remaining_players}")

    def is_tournament_over(self) -> bool:
        """
        Check if the tournament is over.

        Returns:
            bool: True if one or zero players remain, False otherwise.
        """
        result = len(self.remaining_players) <= 1 or not self.remaining_players
        if result:
            logger.info("Tournament over")
        return result

    def get_winner(self) -> Optional[int]:
        """
        Get the winner of the tournament.

        Returns:
            Optional[int]: The ID of the winner, or None if the tournament is not over.
        """
        if self.is_tournament_over() and self.remaining_players:
            logger.info(f"Winner: Player {self.remaining_players[0]}")
            return self.remaining_players[0]
        return None

    def get_prize(self, player_id: int) -> int:
        """
        Get the prize for a specific player.

        Args:
            player_id (int): The ID of the player.

        Returns:
            int: The prize amount (prize_pool if winner, 0 otherwise).
        """
        if player_id == self.get_winner():
            logger.info(f"Player {player_id} wins prize pool: {self.prize_pool}")
            return self.prize_pool
        return 0

    def get_tournament_state(self) -> Dict[str, Any]:
        """
        Get the current state of the tournament.

        Returns:
            Dict[str, Any]: Tournament state information.
        """
        return {
            'buy_in': self.buy_in,
            'multiplier': self.multiplier,
            'prize_pool': self.prize_pool,
            'current_blinds': self.current_blinds,
            'hands_played': self.hands_played,
            'remaining_players': self.remaining_players,
            'eliminated_players': self.eliminated_players,
            'player_stacks': self.player_stacks
        }