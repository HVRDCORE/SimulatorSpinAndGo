"""
Implementation of advanced poker-playing opponent agents.

This module provides sophisticated opponent agents with various playing styles and
strategies for realistic poker simulations. These agents use a combination of
card strength evaluation, positional awareness, betting patterns, and adaptive
strategies based on game context.
"""

import random
from typing import List, Dict, Tuple, Any, Optional, Set

from pokersim.agents.base_agent import Agent
from pokersim.game.state import GameState, Action, ActionType, Stage
from pokersim.game.evaluator import HandEvaluator
from pokersim.game.card import Card


class AdvancedOpponentProfile:
    """
    Profile for an advanced opponent, defining their playing style.
    
    Attributes:
        aggression (float): How aggressively the agent plays (0.0-1.0).
        bluff_frequency (float): How often the agent bluffs (0.0-1.0).
        call_threshold (float): Threshold for calling (0.0-1.0).
        raise_threshold (float): Threshold for raising (0.0-1.0).
        position_importance (float): Importance of position in decision making (0.0-1.0).
        stack_sensitivity (float): How sensitive the agent is to stack sizes (0.0-1.0).
        adapt_to_opponents (float): How much the agent adapts to opponents (0.0-1.0).
        hand_reading_skill (float): Ability to read opponent hands (0.0-1.0).
    """
    
    def __init__(self, 
                aggression: float = 0.5,
                bluff_frequency: float = 0.2,
                call_threshold: float = 0.4,
                raise_threshold: float = 0.6,
                position_importance: float = 0.5,
                stack_sensitivity: float = 0.5,
                adapt_to_opponents: float = 0.5,
                hand_reading_skill: float = 0.5):
        """
        Initialize an opponent profile.
        
        Args:
            aggression (float, optional): How aggressively the agent plays. Defaults to 0.5.
            bluff_frequency (float, optional): How often the agent bluffs. Defaults to 0.2.
            call_threshold (float, optional): Threshold for calling. Defaults to 0.4.
            raise_threshold (float, optional): Threshold for raising. Defaults to 0.6.
            position_importance (float, optional): Importance of position. Defaults to 0.5.
            stack_sensitivity (float, optional): Sensitivity to stack sizes. Defaults to 0.5.
            adapt_to_opponents (float, optional): Adaptability to opponents. Defaults to 0.5.
            hand_reading_skill (float, optional): Hand reading skill. Defaults to 0.5.
        """
        self.aggression = aggression
        self.bluff_frequency = bluff_frequency
        self.call_threshold = call_threshold
        self.raise_threshold = raise_threshold
        self.position_importance = position_importance
        self.stack_sensitivity = stack_sensitivity
        self.adapt_to_opponents = adapt_to_opponents
        self.hand_reading_skill = hand_reading_skill


class AdvancedOpponentAgent(Agent):
    """
    An advanced opponent agent with sophisticated decision-making.
    
    This agent uses a combination of hand strength evaluation, positional awareness,
    opponent modeling, and adaptive strategies to make poker decisions.
    
    Attributes:
        player_id (int): The ID of the player controlled by this agent.
        profile (AdvancedOpponentProfile): The playing style profile.
        opponent_models (Dict[int, Dict]): Models of opponent playing styles.
        hand_history (List[Tuple]): History of hands played by this agent.
        action_history (Dict[int, List]): History of actions by each player.
    """
    
    def __init__(self, player_id: int, profile: Optional[AdvancedOpponentProfile] = None):
        """
        Initialize an advanced opponent agent.
        
        Args:
            player_id (int): The ID of the player controlled by this agent.
            profile (AdvancedOpponentProfile, optional): The playing style profile.
                Defaults to None (uses default profile).
        """
        super().__init__(player_id)
        self.profile = profile if profile is not None else AdvancedOpponentProfile()
        self.opponent_models = {}  # player_id -> model
        self.hand_history = []
        self.action_history = {}  # player_id -> list of actions
        
        # Track statistics for adapting
        self.hands_played = 0
        self.hands_won = 0
        self.total_profit = 0
        self.position_stats = {
            'early': {'played': 0, 'won': 0},
            'middle': {'played': 0, 'won': 0},
            'late': {'played': 0, 'won': 0}
        }
    
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
        
        # Update opponent models
        self._update_opponent_models(game_state)
        
        # Calculate key decision factors
        hand_strength = self._evaluate_hand_strength(game_state)
        position_value = self._evaluate_position(game_state)
        bluff_chance = self._should_bluff(game_state)
        pot_odds = self._calculate_pot_odds(game_state)
        opponent_weakness = self._evaluate_opponent_weakness(game_state)
        
        # Combine factors based on profile
        decision_value = (
            hand_strength * (1.0 - self.profile.aggression * 0.3) +
            position_value * self.profile.position_importance * 0.2 +
            bluff_chance * self.profile.bluff_frequency * 0.2 +
            opponent_weakness * self.profile.aggression * 0.3
        )
        
        # Choose action based on decision value and pot odds
        return self._select_action(decision_value, pot_odds, legal_actions, game_state)
    
    def _evaluate_hand_strength(self, game_state: GameState) -> float:
        """
        Evaluate the strength of the current hand.
        
        Args:
            game_state (GameState): The current game state.
            
        Returns:
            float: The hand strength (0.0-1.0).
        """
        hole_cards = game_state.hole_cards[self.player_id]
        community_cards = game_state.community_cards
        
        # No cards yet
        if not hole_cards:
            return 0.5
        
        # Preflop evaluation
        if game_state.stage == Stage.PREFLOP:
            return self._evaluate_preflop_hand(hole_cards)
        
        # Postflop evaluation
        all_cards = hole_cards + community_cards
        hand_value, _ = HandEvaluator.evaluate_hand(all_cards)
        
        # Scale hand value to 0-1 range (9000 is approximately the max hand value)
        return min(hand_value / 9000, 1.0)
    
    def _evaluate_preflop_hand(self, hole_cards: List[Card]) -> float:
        """
        Evaluate the strength of a preflop hand.
        
        Args:
            hole_cards (List[Card]): The hole cards.
            
        Returns:
            float: The hand strength (0.0-1.0).
        """
        # Group into pairs, suited, and high cards
        if len(hole_cards) != 2:
            return 0.5
        
        c1, c2 = hole_cards
        
        # Check for pairs
        if c1.rank == c2.rank:
            # Scale pair value based on rank (2 -> 0.5, A -> 1.0)
            rank_value = c1.rank.value / 14.0
            return 0.5 + rank_value / 2.0  # Pairs range from 0.5 to 1.0
        
        # Check for suited cards
        suited = c1.suit == c2.suit
        
        # Calculate rank values
        r1, r2 = c1.rank.value, c2.rank.value
        high_rank = max(r1, r2) / 14.0
        low_rank = min(r1, r2) / 14.0
        
        # Calculate connectedness (closer = better)
        connected = 1.0 - (abs(r1 - r2) / 13.0)
        
        # Combine factors
        base_value = high_rank * 0.4 + low_rank * 0.2 + connected * 0.2
        if suited:
            base_value += 0.15
            
        return base_value
    
    def _evaluate_position(self, game_state: GameState) -> float:
        """
        Evaluate the value of the current position.
        
        Args:
            game_state (GameState): The current game state.
            
        Returns:
            float: The position value (0.0-1.0).
        """
        # Determine relative position
        players_to_act = 0
        current_player = game_state.current_player
        
        # Count players still to act after us in this round
        for i in range(game_state.num_players):
            next_player = (current_player + i + 1) % game_state.num_players
            if game_state.active[next_player] and game_state.stacks[next_player] > 0:
                players_to_act += 1
        
        # Later position (fewer players to act) is better
        return 1.0 - (players_to_act / max(1, game_state.num_players - 1))
    
    def _should_bluff(self, game_state: GameState) -> float:
        """
        Determine if the agent should bluff.
        
        Args:
            game_state (GameState): The current game state.
            
        Returns:
            float: The bluff value (0.0-1.0).
        """
        # Higher value means more likely to bluff
        # Base chance on profile's bluff frequency
        base_chance = self.profile.bluff_frequency
        
        # Factors that increase bluff chance:
        # - Late position
        # - Few opponents
        # - Small pot relative to stack
        position_value = self._evaluate_position(game_state)
        active_opponents = sum(game_state.active) - 1
        pot_size_ratio = game_state.pot / max(1, self.profile.stack_sensitivity * game_state.stacks[self.player_id])
        
        # Combine factors
        bluff_value = (
            base_chance * 0.4 +
            position_value * 0.3 +
            (1.0 - active_opponents / max(1, game_state.num_players - 1)) * 0.2 +
            (1.0 - min(1.0, pot_size_ratio)) * 0.1
        )
        
        # Random component for unpredictability
        bluff_value += random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, bluff_value))
    
    def _calculate_pot_odds(self, game_state: GameState) -> float:
        """
        Calculate the pot odds for the current decision.
        
        Args:
            game_state (GameState): The current game state.
            
        Returns:
            float: The pot odds (0.0-1.0).
        """
        # Calculate how much needs to be called
        current_player = game_state.current_player
        to_call = max(0, max(game_state.current_bets) - game_state.current_bets[current_player])
        
        # If no need to call, pot odds are favorable
        if to_call == 0:
            return 1.0
        
        # Calculate pot odds: pot / (pot + to_call)
        pot_with_bets = game_state.pot + sum(game_state.current_bets)
        pot_odds = pot_with_bets / (pot_with_bets + to_call)
        
        return pot_odds
    
    def _evaluate_opponent_weakness(self, game_state: GameState) -> float:
        """
        Evaluate the weakness of opponents.
        
        Args:
            game_state (GameState): The current game state.
            
        Returns:
            float: The opponent weakness value (0.0-1.0).
        """
        # Higher value means opponents are perceived as weaker
        # Default to moderate weakness if no data
        if not self.opponent_models:
            return 0.5
        
        # Extract active opponents
        active_opponents = [
            p for p in range(game_state.num_players) 
            if p != self.player_id and game_state.active[p]
        ]
        
        # If no active opponents, return neutral value
        if not active_opponents:
            return 0.5
        
        # Calculate average perceived weakness
        weakness_sum = 0.0
        for opp_id in active_opponents:
            # If we have a model for this opponent
            if opp_id in self.opponent_models:
                model = self.opponent_models[opp_id]
                # Factors that suggest weakness:
                # - Low aggression
                # - High folding frequency
                # - Predictable betting patterns
                weakness = (
                    (1.0 - model.get('aggression', 0.5)) * 0.4 +
                    model.get('fold_frequency', 0.3) * 0.4 +
                    (1.0 - model.get('unpredictability', 0.5)) * 0.2
                )
                weakness_sum += weakness
            else:
                # Default value for unknown opponents
                weakness_sum += 0.5
        
        return weakness_sum / len(active_opponents)
    
    def _select_action(self, decision_value: float, pot_odds: float, 
                      legal_actions: List[Action], game_state: GameState) -> Action:
        """
        Select an action based on the decision value and pot odds.
        
        Args:
            decision_value (float): The combined decision value (0.0-1.0).
            pot_odds (float): The pot odds (0.0-1.0).
            legal_actions (List[Action]): The legal actions.
            game_state (GameState): The current game state.
            
        Returns:
            Action: The selected action.
        """
        # Adjust decision thresholds based on pot odds
        fold_threshold = self.profile.call_threshold - (pot_odds * 0.3)
        call_threshold = self.profile.raise_threshold - (pot_odds * 0.2)
        
        # Check if check is available
        check_action = next((a for a in legal_actions if a.action_type == ActionType.CHECK), None)
        
        # For betting/raising, size based on hand strength, position, and aggression
        bet_raise_sizing = self._determine_sizing(decision_value, game_state)
        
        # Action selection logic
        if decision_value < fold_threshold:
            # Weak hand: fold or check
            if check_action:
                return check_action
            fold_action = next((a for a in legal_actions if a.action_type == ActionType.FOLD), None)
            if fold_action:
                return fold_action
        
        if decision_value < call_threshold:
            # Medium hand: check or call
            if check_action:
                return check_action
            call_action = next((a for a in legal_actions if a.action_type == ActionType.CALL), None)
            if call_action:
                return call_action
        
        # Strong hand or bluff: bet or raise
        bet_action = next((a for a in legal_actions if a.action_type == ActionType.BET), None)
        if bet_action:
            bet_options = [a for a in legal_actions if a.action_type == ActionType.BET]
            if bet_options:
                # Choose the bet size closest to our target
                return min(bet_options, key=lambda a: abs(a.amount - bet_raise_sizing * game_state.pot))
        
        raise_action = next((a for a in legal_actions if a.action_type == ActionType.RAISE), None)
        if raise_action:
            raise_options = [a for a in legal_actions if a.action_type == ActionType.RAISE]
            if raise_options:
                # Choose the raise size closest to our target
                return min(raise_options, key=lambda a: abs(a.amount - bet_raise_sizing * game_state.pot))
        
        # If we can't bet/raise, call if possible
        call_action = next((a for a in legal_actions if a.action_type == ActionType.CALL), None)
        if call_action:
            return call_action
        
        # Fallback to first legal action (shouldn't get here)
        return legal_actions[0]
    
    def _determine_sizing(self, decision_value: float, game_state: GameState) -> float:
        """
        Determine the bet or raise sizing.
        
        Args:
            decision_value (float): The combined decision value (0.0-1.0).
            game_state (GameState): The current game state.
            
        Returns:
            float: The bet/raise size as a multiple of the pot.
        """
        # Base sizing on hand strength and aggression
        # Range from 0.5x pot to 2x pot
        min_sizing = 0.5
        max_sizing = 2.0
        
        # Scale sizing based on decision value and aggression
        sizing_scale = decision_value * (0.5 + self.profile.aggression)
        sizing = min_sizing + sizing_scale * (max_sizing - min_sizing)
        
        # Adjust for stage (larger bets in later streets)
        if game_state.stage == Stage.TURN:
            sizing *= 1.2
        elif game_state.stage == Stage.RIVER:
            sizing *= 1.5
            
        # Add some randomness
        sizing *= random.uniform(0.8, 1.2)
        
        return sizing
    
    def _update_opponent_models(self, game_state: GameState) -> None:
        """
        Update models of opponents based on their actions.
        
        Args:
            game_state (GameState): The current game state.
        """
        # Extract recent actions
        recent_actions = game_state.history[-game_state.num_players*2:] if game_state.history else []
        
        for player_id, action in recent_actions:
            # Skip own actions
            if player_id == self.player_id:
                continue
                
            # Initialize model if needed
            if player_id not in self.opponent_models:
                self.opponent_models[player_id] = {
                    'aggression': 0.5,
                    'fold_frequency': 0.3,
                    'unpredictability': 0.5,
                    'actions': []
                }
            
            # Update action history
            if player_id not in self.action_history:
                self.action_history[player_id] = []
            self.action_history[player_id].append((game_state.stage, action))
            
            # Update model based on action
            model = self.opponent_models[player_id]
            
            if action.action_type == ActionType.FOLD:
                model['fold_frequency'] = 0.9 * model['fold_frequency'] + 0.1 * 1.0
                model['aggression'] = 0.9 * model['aggression']  # Folding decreases aggression
            
            elif action.action_type in (ActionType.BET, ActionType.RAISE):
                model['fold_frequency'] = 0.9 * model['fold_frequency']
                model['aggression'] = 0.9 * model['aggression'] + 0.1 * 1.0
                
                # Track bet sizing pattern
                if 'bet_sizing' not in model:
                    model['bet_sizing'] = []
                if game_state.pot > 0:
                    relative_size = action.amount / game_state.pot
                    model['bet_sizing'].append(relative_size)
            
            elif action.action_type == ActionType.CALL:
                # Calling is less aggressive
                model['fold_frequency'] = 0.9 * model['fold_frequency']
                model['aggression'] = 0.9 * model['aggression'] + 0.1 * 0.5
            
            # Update unpredictability
            if len(model['actions']) >= 10:
                unique_actions = len(set([a[1].action_type for a in model['actions'][-10:]]))
                model['unpredictability'] = unique_actions / 10.0
            
            # Add to tracked actions
            model['actions'].append((game_state.stage, action))
    
    def observe(self, game_state: GameState) -> None:
        """
        Update the agent's internal state based on the observed game state.
        
        Args:
            game_state (GameState): The current game state.
        """
        # Track hand history if game is over
        if game_state.is_terminal():
            result = {
                'hole_cards': game_state.hole_cards[self.player_id] if self.player_id < len(game_state.hole_cards) else [],
                'community_cards': game_state.community_cards,
                'payout': game_state.get_payouts()[self.player_id] if self.player_id < len(game_state.get_payouts()) else 0
            }
            self.hand_history.append(result)
            
            # Update statistics
            self.hands_played += 1
            if result['payout'] > 0:
                self.hands_won += 1
                self.total_profit += result['payout']
            else:
                self.total_profit -= sum(stake for i, stake in enumerate(game_state.current_bets) 
                                       if i == self.player_id)
                
            # Adapt strategy based on performance
            self._adapt_strategy()
    
    def _adapt_strategy(self) -> None:
        """Adapt the agent's strategy based on performance."""
        # Only adapt after enough hands
        if self.hands_played < 10:
            return
            
        win_rate = self.hands_won / self.hands_played
        
        # Adapt aggression based on win rate
        if win_rate > 0.6:
            # Winning a lot - increase aggression slightly
            self.profile.aggression = min(1.0, self.profile.aggression + 0.05)
        elif win_rate < 0.3:
            # Losing a lot - decrease aggression slightly
            self.profile.aggression = max(0.0, self.profile.aggression - 0.05)
            
        # Adapt bluffing based on profit per hand
        avg_profit = self.total_profit / self.hands_played
        if avg_profit > 0:
            # Profitable - bluff a bit more
            self.profile.bluff_frequency = min(1.0, self.profile.bluff_frequency + 0.03)
        else:
            # Unprofitable - bluff a bit less
            self.profile.bluff_frequency = max(0.0, self.profile.bluff_frequency - 0.03)
            

# Predefined opponent profiles
LOOSE_AGGRESSIVE_PROFILE = AdvancedOpponentProfile(
    aggression=0.8,
    bluff_frequency=0.4,
    call_threshold=0.2,
    raise_threshold=0.4,
    position_importance=0.6,
    stack_sensitivity=0.3,
    adapt_to_opponents=0.7,
    hand_reading_skill=0.6
)

TIGHT_PASSIVE_PROFILE = AdvancedOpponentProfile(
    aggression=0.2,
    bluff_frequency=0.1,
    call_threshold=0.6,
    raise_threshold=0.8,
    position_importance=0.4,
    stack_sensitivity=0.7,
    adapt_to_opponents=0.3,
    hand_reading_skill=0.5
)

LOOSE_PASSIVE_PROFILE = AdvancedOpponentProfile(
    aggression=0.3,
    bluff_frequency=0.2,
    call_threshold=0.2,
    raise_threshold=0.7,
    position_importance=0.3,
    stack_sensitivity=0.5,
    adapt_to_opponents=0.4,
    hand_reading_skill=0.3
)

TIGHT_AGGRESSIVE_PROFILE = AdvancedOpponentProfile(
    aggression=0.7,
    bluff_frequency=0.2,
    call_threshold=0.5,
    raise_threshold=0.7,
    position_importance=0.7,
    stack_sensitivity=0.6,
    adapt_to_opponents=0.6,
    hand_reading_skill=0.8
)

BALANCED_PROFILE = AdvancedOpponentProfile(
    aggression=0.5,
    bluff_frequency=0.25,
    call_threshold=0.4,
    raise_threshold=0.6,
    position_importance=0.6,
    stack_sensitivity=0.5,
    adapt_to_opponents=0.6,
    hand_reading_skill=0.7
)

MANIAC_PROFILE = AdvancedOpponentProfile(
    aggression=0.9,
    bluff_frequency=0.5,
    call_threshold=0.1,
    raise_threshold=0.3,
    position_importance=0.3,
    stack_sensitivity=0.2,
    adapt_to_opponents=0.4,
    hand_reading_skill=0.4
)

ROCK_PROFILE = AdvancedOpponentProfile(
    aggression=0.1,
    bluff_frequency=0.05,
    call_threshold=0.7,
    raise_threshold=0.9,
    position_importance=0.5,
    stack_sensitivity=0.8,
    adapt_to_opponents=0.3,
    hand_reading_skill=0.6
)

ADAPTIVE_PROFILE = AdvancedOpponentProfile(
    aggression=0.5,
    bluff_frequency=0.3,
    call_threshold=0.4,
    raise_threshold=0.6,
    position_importance=0.6,
    stack_sensitivity=0.5,
    adapt_to_opponents=0.9,
    hand_reading_skill=0.7
)