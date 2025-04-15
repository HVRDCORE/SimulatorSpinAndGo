"""
Implementation of the game state for poker simulations.
"""

from enum import Enum, auto
from typing import List, Dict, Tuple, Optional, Any
import copy
import random
import json
import numpy as np

from pokersim.game.card import Card, Rank, Suit
from pokersim.game.deck import Deck
from pokersim.game.evaluator import HandEvaluator


class ActionType(Enum):
    """Enumeration of possible player actions."""
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    BET = auto()
    RAISE = auto()
    
    def __str__(self) -> str:
        return self.name.capitalize()
    
    def __repr__(self) -> str:
        return str(self)


class Action:
    """
    An action taken by a player in a poker game.
    
    Attributes:
        action_type (ActionType): The type of action.
        amount (int, optional): The amount bet or raised, if applicable.
    """
    
    def __init__(self, action_type: ActionType, amount: int = 0):
        """
        Initialize an action.
        
        Args:
            action_type (ActionType): The type of action.
            amount (int, optional): The amount bet or raised, if applicable.
        """
        self.action_type = action_type
        self.amount = amount
    
    def __str__(self) -> str:
        """Return a string representation of the action."""
        if self.action_type in {ActionType.BET, ActionType.RAISE}:
            return f"{self.action_type} {self.amount}"
        return str(self.action_type)
    
    def __repr__(self) -> str:
        """Return a string representation of the action."""
        return str(self)
    
    def __eq__(self, other) -> bool:
        """Check if two actions are equal."""
        if not isinstance(other, Action):
            return False
        return (self.action_type == other.action_type and 
                self.amount == other.amount)


class Stage(Enum):
    """Enumeration of game stages."""
    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()
    SHOWDOWN = auto()
    
    def __str__(self) -> str:
        return self.name.capitalize()
    
    def __repr__(self) -> str:
        return str(self)


class GameState:
    """
    State of a poker game.
    
    Attributes:
        num_players (int): The number of players in the game.
        small_blind (int): The small blind amount.
        big_blind (int): The big blind amount.
        stacks (List[int]): The chip stacks of each player.
        deck (Deck): The deck of cards.
        hole_cards (List[List[Card]]): The hole cards of each player.
        community_cards (List[Card]): The community cards.
        pot (int): The pot size.
        current_bets (List[int]): The current bets of each player.
        stage (Stage): The current stage of the game.
        button (int): The player with the dealer button.
        current_player (int): The player to act.
        last_raiser (int): The last player to raise.
        min_raise (int): The minimum raise amount.
        active (List[bool]): Whether each player is active (not folded).
        history (List[Tuple[int, Action]]): The history of actions.
        stage_history (List[List[Tuple[int, Action]]]): The history of actions by stage.
        payouts (List[int]): The payouts to each player at the end of the hand.
    """
    
    def __init__(self, num_players: int = 2, small_blind: int = 1, big_blind: int = 2, 
                initial_stacks: List[int] = None, button: int = 0):
        """
        Initialize a game state.
        
        Args:
            num_players (int, optional): The number of players. Defaults to 2.
            small_blind (int, optional): The small blind amount. Defaults to 1.
            big_blind (int, optional): The big blind amount. Defaults to 2.
            initial_stacks (List[int], optional): The initial chip stacks. Defaults to 100 for each player.
            button (int, optional): The player with the dealer button. Defaults to 0.
        """
        self.num_players = num_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        
        if initial_stacks is None:
            initial_stacks = [100] * num_players
        
        self.stacks = initial_stacks.copy()
        self.deck = Deck()
        self.hole_cards = [[] for _ in range(num_players)]
        self.community_cards = []
        self.pot = 0
        self.current_bets = [0] * num_players
        self.stage = Stage.PREFLOP
        self.button = button
        self.current_player = self._first_to_act()
        self.last_raiser = -1
        self.min_raise = big_blind
        self.active = [True] * num_players
        self.history = []
        self.stage_history = [[] for _ in range(5)]  # One for each stage + showdown
        self.payouts = [0] * num_players
        
        # Deal hole cards
        self._deal_hole_cards()
        
        # Post blinds
        self._post_blinds()
    
    def _deal_hole_cards(self) -> None:
        """Deal hole cards to each player."""
        for i in range(self.num_players):
            self.hole_cards[i] = self.deck.deal(2)
    
    def _post_blinds(self) -> None:
        """Post blinds."""
        sb_pos = (self.button + 1) % self.num_players
        bb_pos = (self.button + 2) % self.num_players
        
        # Post small blind
        self.stacks[sb_pos] -= self.small_blind
        self.current_bets[sb_pos] = self.small_blind
        self.history.append((sb_pos, Action(ActionType.BET, self.small_blind)))
        self.stage_history[0].append((sb_pos, Action(ActionType.BET, self.small_blind)))
        
        # Post big blind
        self.stacks[bb_pos] -= self.big_blind
        self.current_bets[bb_pos] = self.big_blind
        self.history.append((bb_pos, Action(ActionType.BET, self.big_blind)))
        self.stage_history[0].append((bb_pos, Action(ActionType.BET, self.big_blind)))
        
        self.last_raiser = bb_pos
    
    def _first_to_act(self) -> int:
        """Determine the first player to act."""
        if self.stage == Stage.PREFLOP:
            return (self.button + 3) % self.num_players
        else:
            return (self.button + 1) % self.num_players
    
    def _next_player(self) -> int:
        """Determine the next player to act."""
        p = (self.current_player + 1) % self.num_players
        while not self.active[p] or self.stacks[p] == 0:
            p = (p + 1) % self.num_players
            if p == self.current_player:
                return -1  # No active players left
        return p
    
    def _stage_complete(self) -> bool:
        """Check if the current stage is complete."""
        # All players folded except one
        if sum(self.active) == 1:
            return True
        
        # All active players have bet the same amount or are all-in
        active_players = [i for i, active in enumerate(self.active) if active]
        for p in active_players:
            if self.current_bets[p] != self.current_bets[self.last_raiser] and self.stacks[p] > 0:
                return False
        
        # All players have acted
        if self.current_player == -1 or self.current_player == self.last_raiser:
            return True
        
        return False
    
    def _advance_stage(self) -> None:
        """Advance to the next stage of the game."""
        # Move bets to pot
        self.pot += sum(self.current_bets)
        self.current_bets = [0] * self.num_players
        
        # Reset last raiser and min raise
        self.last_raiser = -1
        self.min_raise = self.big_blind
        
        # Deal community cards
        if self.stage == Stage.PREFLOP:
            self.stage = Stage.FLOP
            self.community_cards.extend(self.deck.deal(3))
        elif self.stage == Stage.FLOP:
            self.stage = Stage.TURN
            self.community_cards.extend(self.deck.deal(1))
        elif self.stage == Stage.TURN:
            self.stage = Stage.RIVER
            self.community_cards.extend(self.deck.deal(1))
        elif self.stage == Stage.RIVER:
            self.stage = Stage.SHOWDOWN
            self._showdown()
        
        # Set first to act
        self.current_player = self._first_to_act()
        while not self.active[self.current_player] and self.current_player != -1:
            self.current_player = self._next_player()
    
    def _showdown(self) -> None:
        """Determine the winner(s) at showdown."""
        if sum(self.active) == 1:
            # Only one player left, they win the pot
            winner = self.active.index(True)
            self.payouts[winner] = self.pot
            return
        
        # Evaluate hands of active players
        active_players = [i for i, active in enumerate(self.active) if active]
        best_hand_rank = -1
        winners = []
        
        for p in active_players:
            hand = self.hole_cards[p] + self.community_cards
            hand_rank, _ = HandEvaluator.evaluate_hand(hand)
            
            if hand_rank > best_hand_rank:
                best_hand_rank = hand_rank
                winners = [p]
            elif hand_rank == best_hand_rank:
                winners.append(p)
        
        # Split pot among winners
        for winner in winners:
            self.payouts[winner] = self.pot // len(winners)
    
    def get_legal_actions(self) -> List[Action]:
        """
        Get the legal actions for the current player.
        
        Returns:
            List[Action]: The legal actions.
        """
        if self.current_player == -1 or self.stage == Stage.SHOWDOWN:
            return []
        
        actions = []
        
        # Fold is always legal
        actions.append(Action(ActionType.FOLD))
        
        # Check is legal if no one has bet
        if max(self.current_bets) == 0 or self.current_bets[self.current_player] == max(self.current_bets):
            actions.append(Action(ActionType.CHECK))
        
        # Call is legal if someone has bet
        if max(self.current_bets) > self.current_bets[self.current_player]:
            call_amount = min(max(self.current_bets) - self.current_bets[self.current_player], 
                            self.stacks[self.current_player])
            actions.append(Action(ActionType.CALL, call_amount))
        
        # Bet is legal if no one has bet
        if max(self.current_bets) == 0 and self.stacks[self.current_player] > 0:
            bet_amount = min(self.big_blind, self.stacks[self.current_player])
            actions.append(Action(ActionType.BET, bet_amount))
            
            # Can also bet more if there's enough in the stack
            if self.stacks[self.current_player] > self.big_blind:
                actions.append(Action(ActionType.BET, self.stacks[self.current_player]))
        
        # Raise is legal if someone has bet and we have enough chips
        if (max(self.current_bets) > 0 and 
            self.stacks[self.current_player] > max(self.current_bets) - self.current_bets[self.current_player]):
            # Min raise
            min_raise_amount = min(self.min_raise, self.stacks[self.current_player])
            actions.append(Action(ActionType.RAISE, min_raise_amount))
            
            # Can also raise more if there's enough in the stack
            if self.stacks[self.current_player] > self.min_raise:
                actions.append(Action(ActionType.RAISE, self.stacks[self.current_player]))
        
        return actions
    
    def apply_action(self, action: Action) -> 'GameState':
        """
        Apply an action to the game state.
        
        Args:
            action (Action): The action to apply.
            
        Returns:
            GameState: The new game state.
            
        Raises:
            ValueError: If the action is not legal.
        """
        if action not in self.get_legal_actions():
            raise ValueError(f"Illegal action: {action}")
        
        new_state = copy.deepcopy(self)
        
        if action.action_type == ActionType.FOLD:
            new_state.active[new_state.current_player] = False
        
        elif action.action_type == ActionType.CHECK:
            pass  # No change to the game state
        
        elif action.action_type == ActionType.CALL:
            call_amount = min(max(new_state.current_bets) - new_state.current_bets[new_state.current_player], 
                            new_state.stacks[new_state.current_player])
            new_state.stacks[new_state.current_player] -= call_amount
            new_state.current_bets[new_state.current_player] += call_amount
        
        elif action.action_type == ActionType.BET:
            new_state.stacks[new_state.current_player] -= action.amount
            new_state.current_bets[new_state.current_player] = action.amount
            new_state.last_raiser = new_state.current_player
            new_state.min_raise = action.amount
        
        elif action.action_type == ActionType.RAISE:
            raise_amount = action.amount - (max(new_state.current_bets) - new_state.current_bets[new_state.current_player])
            new_state.stacks[new_state.current_player] -= raise_amount
            new_state.current_bets[new_state.current_player] = max(new_state.current_bets) + action.amount
            new_state.last_raiser = new_state.current_player
            new_state.min_raise = action.amount
        
        # Record action in history
        new_state.history.append((new_state.current_player, action))
        new_state.stage_history[new_state.stage.value - 1].append((new_state.current_player, action))
        
        # Update current player
        new_state.current_player = new_state._next_player()
        
        # Check if stage is complete
        if new_state._stage_complete():
            new_state._advance_stage()
        
        return new_state
    
    def is_terminal(self) -> bool:
        """
        Check if the game state is terminal.
        
        Returns:
            bool: True if the game state is terminal, False otherwise.
        """
        return self.stage == Stage.SHOWDOWN or sum(self.active) == 1
    
    def get_observation(self, player_id: int) -> Dict[str, Any]:
        """
        Get an observation of the game state from a player's perspective.
        
        Args:
            player_id (int): The player ID.
            
        Returns:
            Dict[str, Any]: The observation.
        """
        observation = {
            'player_id': player_id,
            'num_players': self.num_players,
            'small_blind': self.small_blind,
            'big_blind': self.big_blind,
            'stacks': self.stacks,
            'hole_cards': self.hole_cards[player_id] if 0 <= player_id < self.num_players else None,
            'community_cards': self.community_cards,
            'pot': self.pot,
            'current_bets': self.current_bets,
            'stage': self.stage,
            'button': self.button,
            'current_player': self.current_player,
            'active': self.active,
            'history': self.history,
            'legal_actions': self.get_legal_actions() if self.current_player == player_id else []
        }
        
        return observation
    
    def get_payouts(self) -> List[int]:
        """
        Get the payouts for each player.
        
        Returns:
            List[int]: The payouts.
        """
        return self.payouts
    
    def get_rewards(self) -> List[float]:
        """
        Get the rewards for each player.
        
        Returns:
            List[float]: The rewards.
        """
        if not self.is_terminal():
            return [0.0] * self.num_players
        
        return [float(payout) for payout in self.payouts]
    
    def to_feature_vector(self, player_id: int) -> np.ndarray:
        """
        Convert the game state to a feature vector from a player's perspective.
        
        Args:
            player_id (int): The player ID.
            
        Returns:
            np.ndarray: The feature vector.
        """
        # Features:
        # - One-hot encoding of hole cards (2 cards x 52 possibilities)
        # - One-hot encoding of community cards (5 cards x 52 possibilities)
        # - Pot size (normalized)
        # - Stack sizes (normalized for each player)
        # - Current bets (normalized for each player)
        # - Button position (one-hot encoding)
        # - Current player (one-hot encoding)
        # - Active players (binary for each player)
        
        # One-hot encoding of hole cards
        hole_cards_features = np.zeros(2 * 52)
        if 0 <= player_id < self.num_players:
            for i, card in enumerate(self.hole_cards[player_id]):
                hole_cards_features[i * 52 + card.to_int()] = 1
        
        # One-hot encoding of community cards
        community_cards_features = np.zeros(5 * 52)
        for i, card in enumerate(self.community_cards):
            community_cards_features[i * 52 + card.to_int()] = 1
        
        # Pot size (normalized)
        pot_feature = np.array([self.pot / (sum(self.stacks) + self.pot)])
        
        # Stack sizes (normalized for each player)
        stack_features = np.array([stack / (sum(self.stacks) + self.pot) for stack in self.stacks])
        
        # Current bets (normalized for each player)
        bet_features = np.array([bet / (sum(self.stacks) + self.pot) for bet in self.current_bets])
        
        # Button position (one-hot encoding)
        button_features = np.zeros(self.num_players)
        button_features[self.button] = 1
        
        # Current player (one-hot encoding)
        current_player_features = np.zeros(self.num_players)
        if self.current_player != -1:
            current_player_features[self.current_player] = 1
        
        # Active players (binary for each player)
        active_features = np.array([1 if active else 0 for active in self.active])
        
        # Concatenate all features
        features = np.concatenate([
            hole_cards_features,
            community_cards_features,
            pot_feature,
            stack_features,
            bet_features,
            button_features,
            current_player_features,
            active_features
        ])
        
        return features
    
    def __str__(self) -> str:
        """Return a string representation of the game state."""
        s = f"Stage: {self.stage}\n"
        s += f"Pot: {self.pot}\n"
        s += f"Community cards: {self.community_cards}\n"
        s += "Players:\n"
        
        for i in range(self.num_players):
            s += f"  Player {i}: "
            if i == self.button:
                s += "(Button) "
            if i == self.current_player:
                s += "(To act) "
            s += f"Stack: {self.stacks[i]} "
            s += f"Bet: {self.current_bets[i]} "
            if self.active[i]:
                s += f"Cards: {self.hole_cards[i]}\n"
            else:
                s += "Folded\n"
        
        return s
