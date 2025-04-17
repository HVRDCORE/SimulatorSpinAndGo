"""
Implementation of rule-based agents for poker simulations.
"""

from typing import List, Dict, Any
import random

from pokersim.agents.base_agent import Agent
from pokersim.game.state import GameState, Action, ActionType, Stage
from pokersim.game.evaluator import HandEvaluator


class RuleBased1Agent(Agent):
    """
    A rule-based agent that plays based on hand strength and pot odds.

    Attributes:
        player_id (int): The ID of the player controlled by this agent.
        aggression (float): How aggressively the agent plays (0.0-1.0).
    """

    def __init__(self, player_id: int, aggression: float = 0.5):
        super().__init__(player_id)
        self.aggression = max(0.0, min(1.0, aggression))

    def act(self, game_state: GameState) -> Action:
        legal_actions = game_state.get_legal_actions()
        if not legal_actions:
            raise ValueError("No legal actions available")

        hand_strength = self._evaluate_hand_strength(game_state)
        hand_strength = hand_strength * (1.0 + self.aggression)
        pot_odds = self._calculate_pot_odds(game_state)

        return self._select_action(legal_actions, hand_strength, pot_odds)

    def _evaluate_hand_strength(self, game_state: GameState) -> float:
        hole_cards = game_state.hole_cards[self.player_id]
        community_cards = game_state.community_cards
        if game_state.stage == Stage.PREFLOP:
            return self._evaluate_preflop(hole_cards)
        else:
            hand = hole_cards + community_cards
            hand_rank, _ = HandEvaluator.evaluate_hand(hand)
            return min(1.0, hand_rank / 8.0)

    def _evaluate_preflop(self, hole_cards: List) -> float:
        if len(hole_cards) != 2:
            return 0.0
        if hole_cards[0].rank == hole_cards[1].rank:
            rank_value = hole_cards[0].rank.value
            return 0.5 + (rank_value - 2) / 24.0
        suited = hole_cards[0].suit == hole_cards[1].suit
        rank1 = hole_cards[0].rank.value
        rank2 = hole_cards[1].rank.value
        rank_strength = (rank1 + rank2 - 4) / 26.0
        suited_bonus = 0.1 if suited else 0.0
        gap = abs(rank1 - rank2)
        connected_bonus = max(0.0, 0.1 - 0.02 * gap)
        return min(1.0, rank_strength + suited_bonus + connected_bonus)

    def _calculate_pot_odds(self, game_state: GameState) -> float:
        pot_size = game_state.pot + sum(game_state.current_bets)
        amount_to_call = max(game_state.current_bets) - game_state.current_bets[self.player_id] if max(game_state.current_bets) > game_state.current_bets[self.player_id] else 0
        return amount_to_call / (pot_size + amount_to_call) if pot_size > 0 and amount_to_call > 0 else 0.0

    def _select_action(self, legal_actions: List[Action], hand_strength: float, pot_odds: float) -> Action:
        action_priority = [ActionType.RAISE, ActionType.BET, ActionType.CALL, ActionType.CHECK, ActionType.FOLD]
        for action_type in action_priority:
            for action in legal_actions:
                if action.action_type == action_type:
                    if action_type in [ActionType.RAISE, ActionType.BET] and hand_strength > pot_odds + 0.2:
                        legal_actions_of_type = [a for a in legal_actions if a.action_type == action_type]
                        idx = min(int(hand_strength * len(legal_actions_of_type)), len(legal_actions_of_type) - 1)
                        return legal_actions_of_type[idx]
                    if action_type in [ActionType.CALL, ActionType.CHECK] and hand_strength > pot_odds - 0.1:
                        return action
                    if action_type == ActionType.FOLD and hand_strength <= pot_odds - 0.1:
                        return action
        return legal_actions[0]


class RuleBased2Agent(Agent):
    """
    A more sophisticated rule-based agent that plays based on hand strength, pot odds, and position.

    Attributes:
        player_id (int): The ID of the player controlled by this agent.
        aggression (float): How aggressively the agent plays (0.0-1.0).
        bluff_frequency (float): How often the agent bluffs (0.0-1.0).
    """

    def __init__(self, player_id: int, aggression: float = 0.5, bluff_frequency: float = 0.1):
        super().__init__(player_id)
        self.aggression = max(0.0, min(1.0, aggression))
        self.bluff_frequency = max(0.0, min(1.0, bluff_frequency))

    def act(self, game_state: GameState) -> Action:
        legal_actions = game_state.get_legal_actions()
        if not legal_actions:
            raise ValueError("No legal actions available")

        spr = self._calculate_stack_to_pot_ratio(game_state)
        bluffing = random.random() < self.bluff_frequency and spr > 2.0 and sum(game_state.active) <= 2
        hand_strength = self._evaluate_hand_strength(game_state)
        position_value = self._evaluate_position(game_state)
        hand_strength = hand_strength * (1.0 + 0.2 * position_value) * (1.0 + 0.5 * self.aggression)
        if bluffing:
            hand_strength = 0.8 + 0.2 * random.random()
        pot_odds = self._calculate_pot_odds(game_state)

        return self._select_action(legal_actions, hand_strength, pot_odds)

    def _evaluate_hand_strength(self, game_state: GameState) -> float:
        hole_cards = game_state.hole_cards[self.player_id]
        community_cards = game_state.community_cards
        if game_state.stage == Stage.PREFLOP:
            return self._evaluate_preflop(hole_cards)
        else:
            hand = hole_cards + community_cards
            hand_rank, _ = HandEvaluator.evaluate_hand(hand)
            return min(1.0, hand_rank / 8.0)

    def _evaluate_preflop(self, hole_cards: List) -> float:
        if len(hole_cards) != 2:
            return 0.0
        if hole_cards[0].rank == hole_cards[1].rank:
            rank_value = hole_cards[0].rank.value
            return 0.5 + (rank_value - 2) / 24.0
        suited = hole_cards[0].suit == hole_cards[1].suit
        rank1 = hole_cards[0].rank.value
        rank2 = hole_cards[1].rank.value
        rank_strength = (rank1 + rank2 - 4) / 26.0
        suited_bonus = 0.1 if suited else 0.0
        gap = abs(rank1 - rank2)
        connected_bonus = max(0.0, 0.1 - 0.02 * gap)
        return min(1.0, rank_strength + suited_bonus + connected_bonus)

    def _calculate_pot_odds(self, game_state: GameState) -> float:
        pot_size = game_state.pot + sum(game_state.current_bets)
        amount_to_call = max(game_state.current_bets) - game_state.current_bets[self.player_id] if max(game_state.current_bets) > game_state.current_bets[self.player_id] else 0
        return amount_to_call / (pot_size + amount_to_call) if pot_size > 0 and amount_to_call > 0 else 0.0

    def _calculate_stack_to_pot_ratio(self, game_state: GameState) -> float:
        pot_size = game_state.pot + sum(game_state.current_bets)
        stack = game_state.stacks[self.player_id]
        return stack / pot_size if pot_size > 0 else float('inf')

    def _evaluate_position(self, game_state: GameState) -> float:
        active_players = sum(game_state.active)
        if active_players <= 1:
            return 1.0
        position = (self.player_id - game_state.button) % game_state.num_players
        return position / (game_state.num_players - 1)

    def _select_action(self, legal_actions: List[Action], hand_strength: float, pot_odds: float) -> Action:
        action_priority = [ActionType.RAISE, ActionType.BET, ActionType.CALL, ActionType.CHECK, ActionType.FOLD]
        for action_type in action_priority:
            for action in legal_actions:
                if action.action_type == action_type:
                    if action_type in [ActionType.RAISE, ActionType.BET] and hand_strength > pot_odds + 0.2:
                        legal_actions_of_type = [a for a in legal_actions if a.action_type == action_type]
                        idx = min(int(hand_strength * len(legal_actions_of_type)), len(legal_actions_of_type) - 1)
                        return legal_actions_of_type[idx]
                    if action_type in [ActionType.CALL, ActionType.CHECK] and hand_strength > pot_odds - 0.1:
                        return action
                    if action_type == ActionType.FOLD and hand_strength <= pot_odds - 0.1:
                        return action
        return legal_actions[0]