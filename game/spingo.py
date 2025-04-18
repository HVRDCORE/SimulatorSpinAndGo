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
from pokersim.config.config_manager import get_config

from pokersim.game.state import GameState, Action, ActionType, Stage

logger = logging.getLogger(__name__)

class SpinGoGame:
    def __init__(self, buy_in: int = 10, num_players: int = 3, starting_stack: int = 500,
                 blind_schedule: Optional[List[Tuple[int, int]]] = None,
                 blind_level_duration: int = 5):
        config = get_config()
        self.num_players = 3  # Фиксированное значение для Spin and Go
        if num_players != 3:
            logger.warning(f"SpinGoGame требует 3 игроков, заданное значение {num_players} игнорируется")
        self.buy_in = config.get('tournament.buy_in', buy_in)
        self.starting_stack = config.get('tournament.starting_chips', starting_stack)
        self.blind_level_duration = config.get('tournament.level_duration', blind_level_duration) // 36

        if self.num_players < 2:
            raise ValueError("Number of players must be at least 2")

        self.multiplier = self._determine_multiplier()
        self.prize_pool = int(self.buy_in * self.num_players * self.multiplier)

        if blind_schedule is None:
            blind_schedule = [
                (5, 10), (10, 20), (15, 30), (20, 40), (30, 60),
                (50, 100), (75, 150), (100, 200), (150, 300), (200, 400)
            ]
        self.blind_levels = blind_schedule
        self.current_blinds = self.blind_levels[0]
        self.hands_played = 0
        self.remaining_players = list(range(self.num_players))
        self.eliminated_players = []
        self.player_stacks = [self.starting_stack] * self.num_players
        self.current_game = None
        self.button_position = 0
        logger.info(f"Initialized SpinGoGame with {self.num_players} players, prize pool {self.prize_pool}")

    def _determine_multiplier(self) -> float:
        config = get_config()
        multipliers = config.get('tournament.multipliers', [2, 6, 25, 120, 240, 1200])
        weights = config.get('tournament.multiplier_weights', [75, 15, 5, 4, 0.9, 0.1])
        return random.choices(multipliers, weights=weights, k=1)[0]

    def _update_blinds(self) -> None:
        if self.hands_played % self.blind_level_duration == 0:
            level_idx = min(self.hands_played // self.blind_level_duration, len(self.blind_levels) - 1)
            self.current_blinds = self.blind_levels[level_idx]
            logger.debug(f"Blinds updated to {self.current_blinds}")

    def start_new_hand(self) -> GameState:
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
        if self.current_game is None:
            logger.warning("No game initialized, skipping stack update")
            return

        try:
            if not self.current_game.is_terminal():
                logger.debug("Игра не терминальна, распределяем банк между активными игроками")
                active_players = [i for i, active in enumerate(self.current_game.active) if
                                  active and self.current_game.stacks[i] > 0]
                if active_players:
                    pot_per_player = self.current_game.pot // len(active_players)
                    for p in active_players:
                        self.current_game.payouts[p] = pot_per_player
                    logger.info(
                        f"Банк {self.current_game.pot} разделён между игроками {active_players}, каждому по {pot_per_player}")

            payouts = self.current_game.get_payouts()
            logger.info(f"Payouts: {payouts}, текущие стеки: {self.player_stacks}")
        except Exception as e:
            logger.error(f"Error getting payouts: {str(e)}")
            self.current_game = None
            return

        eliminated_this_hand = []
        for i, player_id in enumerate(self.current_game.player_ids):
            self.player_stacks[player_id] += payouts[i]
            if self.player_stacks[player_id] < 0:
                logger.error(f"Отрицательный стек для игрока {player_id}: {self.player_stacks[player_id]}")
                self.player_stacks[player_id] = 0  # Корректируем отрицательный стек
            logger.debug(f"Player {player_id} stack updated to {self.player_stacks[player_id]}")
            if self.player_stacks[player_id] <= 0:
                eliminated_this_hand.append(player_id)

        for player_id in eliminated_this_hand:
            if player_id in self.remaining_players:
                self.remaining_players.remove(player_id)
                self.eliminated_players.append(player_id)
                logger.info(f"Player {player_id} eliminated")

        # Проверка инварианта: сумма стеков = 1500
        total_stacks = sum(self.player_stacks)
        if total_stacks != 1500:
            logger.error(f"Нарушение инварианта: сумма стеков = {total_stacks}, ожидается 1500")
            # Можно добавить корректировку, но лучше найти источник ошибки

        logger.debug(f"Current stacks: {self.player_stacks}")
        self.current_game = None

    def is_tournament_over(self) -> bool:
        result = len(self.remaining_players) <= 1 or not self.remaining_players
        if result:
            winner = self.get_winner()
            if winner is not None:
                logger.info(f"Tournament over. Winner: Player {winner} with stack {self.player_stacks[winner]}")
            else:
                logger.info("Tournament over with no winner (all players eliminated)")
        return result

    def get_winner(self) -> Optional[int]:
        if len(self.remaining_players) == 1:
            winner = self.remaining_players[0]
            if self.player_stacks[winner] == 1500:
                logger.info(f"Winner: Player {winner} with 1500 chips")
                return winner
            else:
                logger.warning(f"Winner {winner} has {self.player_stacks[winner]} chips, expected 1500")
                return winner
        elif not self.remaining_players:
            logger.warning("No remaining players")
            return None
        return None

    def get_prize(self, player_id: int) -> int:
        if player_id == self.get_winner():
            logger.info(f"Player {player_id} wins prize pool: {self.prize_pool}")
            return self.prize_pool
        return 0

    def get_tournament_state(self) -> Dict[str, Any]:
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