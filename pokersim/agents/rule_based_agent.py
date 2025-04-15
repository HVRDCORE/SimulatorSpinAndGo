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
    A rule-based agent that plays based on hand strength.
    
    This agent evaluates its hand strength and makes decisions accordingly:
    - Strong hands: raises/bets
    - Medium hands: calls/checks
    - Weak hands: checks/folds
    
    Attributes:
        player_id (int): The ID of the player controlled by this agent.
        aggression (float): How aggressively the agent plays (0.0-1.0).
    """
    
    def __init__(self, player_id: int, aggression: float = 0.5):
        """
        Initialize a rule-based agent.
        
        Args:
            player_id (int): The ID of the player controlled by this agent.
            aggression (float, optional): How aggressively the agent plays (0.0-1.0). Defaults to 0.5.
        """
        super().__init__(player_id)
        self.aggression = max(0.0, min(1.0, aggression))
    
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
        
        # Evaluate hand strength
        hand_strength = self._evaluate_hand_strength(game_state)
        
        # Adjust for aggression
        hand_strength = hand_strength * (1.0 + self.aggression)
        
        # Decide action based on hand strength
        if hand_strength > 0.7:  # Strong hand
            # Try to raise/bet
            for action in legal_actions:
                if action.action_type == ActionType.RAISE:
                    # Raise by a random amount
                    raise_action = next((a for a in legal_actions if a.action_type == ActionType.RAISE), None)
                    if raise_action:
                        legal_raises = [a for a in legal_actions if a.action_type == ActionType.RAISE]
                        return random.choice(legal_raises)
                    
                if action.action_type == ActionType.BET:
                    # Bet by a random amount
                    bet_action = next((a for a in legal_actions if a.action_type == ActionType.BET), None)
                    if bet_action:
                        legal_bets = [a for a in legal_actions if a.action_type == ActionType.BET]
                        return random.choice(legal_bets)
            
            # If can't raise/bet, try to call/check
            for action in legal_actions:
                if action.action_type == ActionType.CALL:
                    return action
                if action.action_type == ActionType.CHECK:
                    return action
            
            # If can't call/check, fold
            return legal_actions[0]
            
        elif hand_strength > 0.3:  # Medium hand
            # Try to call/check
            for action in legal_actions:
                if action.action_type == ActionType.CHECK:
                    return action
                if action.action_type == ActionType.CALL:
                    return action
            
            # If can't call/check, try to raise/bet with some probability
            if random.random() < hand_strength:
                for action in legal_actions:
                    if action.action_type == ActionType.RAISE:
                        legal_raises = [a for a in legal_actions if a.action_type == ActionType.RAISE]
                        return legal_raises[0]  # Min raise
                    
                    if action.action_type == ActionType.BET:
                        legal_bets = [a for a in legal_actions if a.action_type == ActionType.BET]
                        return legal_bets[0]  # Min bet
            
            # If can't or don't want to raise/bet, fold
            for action in legal_actions:
                if action.action_type == ActionType.FOLD:
                    return action
            
            # If can't fold, choose the first legal action
            return legal_actions[0]
            
        else:  # Weak hand
            # Try to check
            for action in legal_actions:
                if action.action_type == ActionType.CHECK:
                    return action
            
            # If can't check, call with some probability
            if random.random() < hand_strength * 2:
                for action in legal_actions:
                    if action.action_type == ActionType.CALL:
                        return action
            
            # If can't or don't want to call, fold
            for action in legal_actions:
                if action.action_type == ActionType.FOLD:
                    return action
            
            # If can't fold, choose the first legal action
            return legal_actions[0]
    
    def _evaluate_hand_strength(self, game_state: GameState) -> float:
        """
        Evaluate the strength of the agent's hand.
        
        Args:
            game_state (GameState): The current game state.
            
        Returns:
            float: The hand strength, from 0.0 (weakest) to 1.0 (strongest).
        """
        # Get the agent's hole cards
        hole_cards = game_state.hole_cards[self.player_id]
        
        # Get the community cards
        community_cards = game_state.community_cards
        
        # Evaluate based on stage
        if game_state.stage == Stage.PREFLOP:
            return self._evaluate_preflop(hole_cards)
        else:
            # Evaluate based on the full hand
            hand = hole_cards + community_cards
            hand_rank, _ = HandEvaluator.evaluate_hand(hand)
            
            # Map hand rank to strength
            return min(1.0, hand_rank / 8.0)
    
    def _evaluate_preflop(self, hole_cards: List) -> float:
        """
        Evaluate the strength of the agent's hole cards.
        
        Args:
            hole_cards (List): The agent's hole cards.
            
        Returns:
            float: The hand strength, from 0.0 (weakest) to 1.0 (strongest).
        """
        # Check for pocket pair
        if hole_cards[0].rank == hole_cards[1].rank:
            # Map rank to strength (higher pairs are stronger)
            rank_value = hole_cards[0].rank.value
            return 0.5 + (rank_value - 2) / 24.0  # Map 2-14 to 0.5-1.0
        
        # Check for suited cards
        suited = hole_cards[0].suit == hole_cards[1].suit
        
        # Calculate strength based on ranks
        rank1 = hole_cards[0].rank.value
        rank2 = hole_cards[1].rank.value
        
        # Higher ranks are better
        rank_strength = (rank1 + rank2 - 4) / 26.0  # Map 4-28 to 0.0-0.923
        
        # Suited cards are better
        suited_bonus = 0.1 if suited else 0.0
        
        # Connected cards are better
        gap = abs(rank1 - rank2)
        connected_bonus = max(0.0, 0.1 - 0.02 * gap)
        
        return min(1.0, rank_strength + suited_bonus + connected_bonus)


class RuleBased2Agent(Agent):
    """
    A more sophisticated rule-based agent that plays based on hand strength and pot odds.
    
    This agent evaluates its hand strength, pot odds, and position to make decisions.
    
    Attributes:
        player_id (int): The ID of the player controlled by this agent.
        aggression (float): How aggressively the agent plays (0.0-1.0).
        bluff_frequency (float): How often the agent bluffs (0.0-1.0).
    """
    
    def __init__(self, player_id: int, aggression: float = 0.5, bluff_frequency: float = 0.1):
        """
        Initialize a rule-based agent.
        
        Args:
            player_id (int): The ID of the player controlled by this agent.
            aggression (float, optional): How aggressively the agent plays (0.0-1.0). Defaults to 0.5.
            bluff_frequency (float, optional): How often the agent bluffs (0.0-1.0). Defaults to 0.1.
        """
        super().__init__(player_id)
        self.aggression = max(0.0, min(1.0, aggression))
        self.bluff_frequency = max(0.0, min(1.0, bluff_frequency))
    
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
        
        # Decide whether to bluff
        bluffing = random.random() < self.bluff_frequency
        
        # Evaluate hand strength
        hand_strength = self._evaluate_hand_strength(game_state)
        
        # Adjust for position
        position_value = self._evaluate_position(game_state)
        hand_strength = hand_strength * (1.0 + 0.2 * position_value)
        
        # Adjust for aggression
        hand_strength = hand_strength * (1.0 + 0.5 * self.aggression)
        
        # If bluffing, pretend we have a stronger hand
        if bluffing:
            hand_strength = 0.8 + 0.2 * random.random()
        
        # Decide action based on hand strength and pot odds
        pot_odds = self._calculate_pot_odds(game_state)
        
        if hand_strength > pot_odds + 0.2:  # Strong hand relative to pot odds
            # Try to raise/bet
            for action in legal_actions:
                if action.action_type == ActionType.RAISE:
                    # Choose raise amount based on hand strength
                    legal_raises = [a for a in legal_actions if a.action_type == ActionType.RAISE]
                    raise_idx = min(int(hand_strength * len(legal_raises)), len(legal_raises) - 1)
                    return legal_raises[raise_idx]
                
                if action.action_type == ActionType.BET:
                    # Choose bet amount based on hand strength
                    legal_bets = [a for a in legal_actions if a.action_type == ActionType.BET]
                    bet_idx = min(int(hand_strength * len(legal_bets)), len(legal_bets) - 1)
                    return legal_bets[bet_idx]
            
            # If can't raise/bet, try to call/check
            for action in legal_actions:
                if action.action_type == ActionType.CALL:
                    return action
                if action.action_type == ActionType.CHECK:
                    return action
            
            # If can't call/check, fold
            return legal_actions[0]
            
        elif hand_strength > pot_odds - 0.1:  # Hand strength close to pot odds
            # Try to call/check
            for action in legal_actions:
                if action.action_type == ActionType.CHECK:
                    return action
                if action.action_type == ActionType.CALL:
                    return action
            
            # If can't call/check, try to raise/bet with some probability
            if random.random() < hand_strength:
                for action in legal_actions:
                    if action.action_type == ActionType.RAISE:
                        legal_raises = [a for a in legal_actions if a.action_type == ActionType.RAISE]
                        return legal_raises[0]  # Min raise
                    
                    if action.action_type == ActionType.BET:
                        legal_bets = [a for a in legal_actions if a.action_type == ActionType.BET]
                        return legal_bets[0]  # Min bet
            
            # If can't or don't want to raise/bet, fold
            for action in legal_actions:
                if action.action_type == ActionType.FOLD:
                    return action
            
            # If can't fold, choose the first legal action
            return legal_actions[0]
            
        else:  # Weak hand relative to pot odds
            # Try to check
            for action in legal_actions:
                if action.action_type == ActionType.CHECK:
                    return action
            
            # If can't check, fold
            for action in legal_actions:
                if action.action_type == ActionType.FOLD:
                    return action
            
            # If can't fold, call with some probability
            if random.random() < hand_strength * 2:
                for action in legal_actions:
                    if action.action_type == ActionType.CALL:
                        return action
            
            # If can't call, choose the first legal action
            return legal_actions[0]
    
    def _evaluate_hand_strength(self, game_state: GameState) -> float:
        """
        Evaluate the strength of the agent's hand.
        
        Args:
            game_state (GameState): The current game state.
            
        Returns:
            float: The hand strength, from 0.0 (weakest) to 1.0 (strongest).
        """
        # Get the agent's hole cards
        hole_cards = game_state.hole_cards[self.player_id]
        
        # Get the community cards
        community_cards = game_state.community_cards
        
        # Evaluate based on stage
        if game_state.stage == Stage.PREFLOP:
            return self._evaluate_preflop(hole_cards)
        elif game_state.stage == Stage.FLOP:
            return self._evaluate_flop(hole_cards, community_cards)
        elif game_state.stage == Stage.TURN:
            return self._evaluate_turn(hole_cards, community_cards)
        else:  # River or Showdown
            return self._evaluate_river(hole_cards, community_cards)
    
    def _evaluate_preflop(self, hole_cards: List) -> float:
        """
        Evaluate the strength of the agent's hole cards.
        
        Args:
            hole_cards (List): The agent's hole cards.
            
        Returns:
            float: The hand strength, from 0.0 (weakest) to 1.0 (strongest).
        """
        # Check for pocket pair
        if hole_cards[0].rank == hole_cards[1].rank:
            # Map rank to strength (higher pairs are stronger)
            rank_value = hole_cards[0].rank.value
            return 0.5 + (rank_value - 2) / 24.0  # Map 2-14 to 0.5-1.0
        
        # Check for suited cards
        suited = hole_cards[0].suit == hole_cards[1].suit
        
        # Calculate strength based on ranks
        rank1 = hole_cards[0].rank.value
        rank2 = hole_cards[1].rank.value
        
        # Higher ranks are better
        rank_strength = (rank1 + rank2 - 4) / 26.0  # Map 4-28 to 0.0-0.923
        
        # Suited cards are better
        suited_bonus = 0.1 if suited else 0.0
        
        # Connected cards are better
        gap = abs(rank1 - rank2)
        connected_bonus = max(0.0, 0.1 - 0.02 * gap)
        
        return min(1.0, rank_strength + suited_bonus + connected_bonus)
    
    def _evaluate_flop(self, hole_cards: List, community_cards: List) -> float:
        """
        Evaluate the strength of the agent's hand on the flop.
        
        Args:
            hole_cards (List): The agent's hole cards.
            community_cards (List): The community cards.
            
        Returns:
            float: The hand strength, from 0.0 (weakest) to 1.0 (strongest).
        """
        # Evaluate the full hand
        hand = hole_cards + community_cards
        hand_rank, _ = HandEvaluator.evaluate_hand(hand)
        
        # Map hand rank to strength
        return min(1.0, hand_rank / 8.0)
    
    def _evaluate_turn(self, hole_cards: List, community_cards: List) -> float:
        """
        Evaluate the strength of the agent's hand on the turn.
        
        Args:
            hole_cards (List): The agent's hole cards.
            community_cards (List): The community cards.
            
        Returns:
            float: The hand strength, from 0.0 (weakest) to 1.0 (strongest).
        """
        # Evaluate the full hand
        hand = hole_cards + community_cards
        hand_rank, _ = HandEvaluator.evaluate_hand(hand)
        
        # Map hand rank to strength
        return min(1.0, hand_rank / 8.0)
    
    def _evaluate_river(self, hole_cards: List, community_cards: List) -> float:
        """
        Evaluate the strength of the agent's hand on the river.
        
        Args:
            hole_cards (List): The agent's hole cards.
            community_cards (List): The community cards.
            
        Returns:
            float: The hand strength, from 0.0 (weakest) to 1.0 (strongest).
        """
        # Evaluate the full hand
        hand = hole_cards + community_cards
        hand_rank, _ = HandEvaluator.evaluate_hand(hand)
        
        # Map hand rank to strength
        return min(1.0, hand_rank / 8.0)
    
    def _calculate_pot_odds(self, game_state: GameState) -> float:
        """
        Calculate the pot odds.
        
        Args:
            game_state (GameState): The current game state.
            
        Returns:
            float: The pot odds, from 0.0 to 1.0.
        """
        # Get the current pot size
        pot_size = game_state.pot + sum(game_state.current_bets)
        
        # Get the amount to call
        amount_to_call = 0
        if max(game_state.current_bets) > game_state.current_bets[self.player_id]:
            amount_to_call = max(game_state.current_bets) - game_state.current_bets[self.player_id]
        
        # Calculate pot odds
        if amount_to_call == 0:
            return 0.0
        
        pot_odds = amount_to_call / (pot_size + amount_to_call)
        return min(1.0, pot_odds)
    
    def _evaluate_position(self, game_state: GameState) -> float:
        """
        Evaluate the agent's position.
        
        Args:
            game_state (GameState): The current game state.
            
        Returns:
            float: The position value, from 0.0 (worst) to 1.0 (best).
        """
        # Count active players
        active_players = sum(game_state.active)
        
        # Calculate position value
        if active_players <= 1:
            return 1.0
        
        # Find the agent's position relative to the button
        position = (self.player_id - game_state.button) % game_state.num_players
        
        # Map position to value (later positions are better)
        return position / (game_state.num_players - 1)
