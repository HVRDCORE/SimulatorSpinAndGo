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
from typing import List, Dict, Tuple, Optional, Any

from pokersim.game.state import GameState, Action, ActionType, Stage


class SpinGoGame:
    """
    Spin and Go game format simulator.
    
    This class manages a complete Spin and Go tournament, including:
    - Prize pool determination via multiplier
    - Tournament progression with increasing blinds
    - Eliminating players and determining the winner
    
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
    
    # Possible multipliers and their probabilities
    MULTIPLIER_TABLE = {
        1.5: 0.75,    # 75% chance of 1.5x
        3: 0.15,      # 15% chance of 3x
        5: 0.07,      # 7% chance of 5x
        10: 0.02,     # 2% chance of 10x
        100: 0.01,    # 1% chance of 100x
        1000: 0.001   # 0.1% chance of 1000x
    }
    
    def __init__(self, buy_in: int = 10, num_players: int = 3, starting_stack: int = 500,
                blind_schedule: Optional[List[Tuple[int, int]]] = None,
                blind_level_duration: int = 5):
        """
        Initialize a Spin and Go game.
        
        Args:
            buy_in (int, optional): Buy-in amount. Defaults to 10.
            num_players (int, optional): Number of players. Defaults to 3.
            starting_stack (int, optional): Starting chip stack. Defaults to 500.
            blind_schedule (List[Tuple[int, int]], optional): Custom blind schedule.
                Defaults to None, which uses the standard schedule.
            blind_level_duration (int, optional): Hands per blind level. Defaults to 5.
        """
        self.num_players = num_players
        self.buy_in = buy_in
        
        # Determine prize pool multiplier
        self.multiplier = self._determine_multiplier()
        self.prize_pool = int(buy_in * num_players * self.multiplier)
        
        # Setup tournament structure
        self.starting_stack = starting_stack
        
        # Initialize with standard blind schedule if none provided
        if blind_schedule is None:
            blind_schedule = [
                (5, 10),     # Starting blinds 5/10
                (10, 20),    # Level 2
                (15, 30),    # Level 3
                (20, 40),    # Level 4
                (30, 60),    # Level 5
                (50, 100),   # Level 6
                (75, 150),   # Level 7
                (100, 200),  # Level 8
                (150, 300),  # Level 9
                (200, 400)   # Level 10
            ]
        
        self.blind_levels = blind_schedule
        self.current_blinds = self.blind_levels[0]
        self.blind_level_duration = blind_level_duration
        self.hands_played = 0
        
        # Initialize player states
        self.remaining_players = list(range(num_players))
        self.eliminated_players = []
        self.player_stacks = [starting_stack] * num_players
        
        # Current game state
        self.current_game = None
        self.button_position = 0
        
    def _determine_multiplier(self) -> float:
        """
        Determine the prize pool multiplier using weighted random selection.
        
        Returns:
            float: The selected multiplier.
        """
        multipliers = list(self.MULTIPLIER_TABLE.keys())
        probabilities = list(self.MULTIPLIER_TABLE.values())
        
        # Normalize probabilities to ensure they sum to 1
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]
        
        return random.choices(multipliers, weights=probabilities, k=1)[0]
    
    def _update_blinds(self) -> None:
        """Update the blinds if it's time to increase them."""
        if self.hands_played % self.blind_level_duration == 0:
            level_idx = self.hands_played // self.blind_level_duration
            if level_idx < len(self.blind_levels):
                self.current_blinds = self.blind_levels[level_idx]
    
    def start_new_hand(self) -> GameState:
        """
        Start a new hand of poker.
        
        Returns:
            GameState: The initial game state for the new hand.
        """
        # Update blinds if needed
        self._update_blinds()
        
        # Advance button position
        self.button_position = (self.button_position + 1) % len(self.remaining_players)
        
        # Create new game state
        small_blind, big_blind = self.current_blinds
        
        # Map remaining player indices to their actual stacks
        active_players = self.remaining_players
        stacks = [self.player_stacks[p] for p in active_players]
        
        # Create a new game state for the active players
        self.current_game = GameState(
            num_players=len(active_players),
            small_blind=small_blind,
            big_blind=big_blind,
            initial_stacks=stacks,
            button=self.button_position
        )
        
        self.hands_played += 1
        
        return self.current_game
        
    def update_stacks_after_hand(self) -> None:
        """Update player stacks and eliminate players after a hand is completed."""
        if self.current_game is None or not self.current_game.is_terminal():
            raise ValueError("No completed game to process")
        
        # Get payouts from the completed game
        payouts = self.current_game.get_payouts()
        
        # Update player stacks and check for eliminations
        eliminated_this_hand = []
        
        for i, player_id in enumerate(self.remaining_players):
            self.player_stacks[player_id] += payouts[i]
            
            # Check if player is eliminated
            if self.player_stacks[player_id] <= 0:
                eliminated_this_hand.append(player_id)
        
        # Remove eliminated players
        for player_id in eliminated_this_hand:
            self.remaining_players.remove(player_id)
            self.eliminated_players.append(player_id)
    
    def is_tournament_over(self) -> bool:
        """
        Check if the tournament is over (only one player remaining).
        
        Returns:
            bool: True if the tournament is over, False otherwise.
        """
        return len(self.remaining_players) <= 1
    
    def get_winner(self) -> Optional[int]:
        """
        Get the winner of the tournament.
        
        Returns:
            Optional[int]: The ID of the winner, or None if tournament is not over.
        """
        if self.is_tournament_over() and self.remaining_players:
            return self.remaining_players[0]
        return None
    
    def get_prize(self, player_id: int) -> int:
        """
        Get the prize for a player.
        
        Args:
            player_id (int): The player ID.
            
        Returns:
            int: The prize amount.
        """
        # In Spin and Go, winner takes all
        if player_id == self.get_winner():
            return self.prize_pool
        return 0
    
    def get_tournament_state(self) -> Dict[str, Any]:
        """
        Get the current state of the tournament.
        
        Returns:
            Dict[str, Any]: The tournament state.
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