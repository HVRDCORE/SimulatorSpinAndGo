"""
Unit tests for the game module of the pokersim package.

This module contains tests for card, deck, and game state functionality.
"""

import pytest
import random
import numpy as np
from typing import List, Dict, Tuple

from pokersim.game.card import Card, Suit, Rank
from pokersim.game.deck import Deck
from pokersim.game.state import GameState, Action, ActionType, Stage
from pokersim.game.evaluator import HandEvaluator


class TestCard:
    """Tests for the Card class."""
    
    def test_card_initialization(self):
        """Test card initialization with rank and suit."""
        card = Card(Rank.ACE, Suit.SPADES)
        assert card.rank == Rank.ACE
        assert card.suit == Suit.SPADES
    
    def test_card_string_representation(self):
        """Test string representation of a card."""
        card = Card(Rank.ACE, Suit.SPADES)
        assert str(card) == "A♠"
        
        card = Card(Rank.TEN, Suit.HEARTS)
        assert str(card) == "10♥"
    
    def test_card_equality(self):
        """Test equality comparison between cards."""
        card1 = Card(Rank.ACE, Suit.SPADES)
        card2 = Card(Rank.ACE, Suit.SPADES)
        card3 = Card(Rank.KING, Suit.SPADES)
        
        assert card1 == card2
        assert card1 != card3
        assert card2 != card3
    
    def test_card_to_int(self):
        """Test conversion of card to integer representation."""
        card = Card(Rank.ACE, Suit.SPADES)
        assert card.to_int() == 51  # Ace of Spades should be the highest value
        
        card = Card(Rank.TWO, Suit.CLUBS)
        assert card.to_int() == 0   # Two of Clubs should be the lowest value
    
    def test_card_from_int(self):
        """Test creation of card from integer representation."""
        card = Card.from_int(51)
        assert card.rank == Rank.ACE
        assert card.suit == Suit.SPADES
        
        card = Card.from_int(0)
        assert card.rank == Rank.TWO
        assert card.suit == Suit.CLUBS
    
    def test_all_cards_unique(self):
        """Test that all 52 cards have unique integer representations."""
        int_values = set()
        for rank in Rank:
            for suit in Suit:
                card = Card(rank, suit)
                int_value = card.to_int()
                assert int_value not in int_values
                int_values.add(int_value)
        
        assert len(int_values) == 52


class TestDeck:
    """Tests for the Deck class."""
    
    def test_deck_initialization(self):
        """Test deck initialization with standard 52 cards."""
        deck = Deck(shuffle=False)
        assert len(deck.cards) == 52
    
    def test_deck_shuffling(self):
        """Test that shuffling changes the order of cards."""
        deck1 = Deck(shuffle=False)
        deck2 = Deck(shuffle=True)
        
        # The probability of two shuffled decks being identical is extremely low
        assert deck1.cards != deck2.cards
    
    def test_deal_one(self):
        """Test dealing a single card from the deck."""
        deck = Deck()
        initial_size = len(deck)
        card = deck.deal_one()
        
        assert len(deck) == initial_size - 1
        assert isinstance(card, Card)
    
    def test_deal_multiple(self):
        """Test dealing multiple cards from the deck."""
        deck = Deck()
        initial_size = len(deck)
        cards = deck.deal(5)
        
        assert len(deck) == initial_size - 5
        assert len(cards) == 5
        assert all(isinstance(card, Card) for card in cards)
    
    def test_deal_too_many(self):
        """Test dealing more cards than are in the deck."""
        deck = Deck()
        with pytest.raises(ValueError):
            deck.deal(53)  # Try to deal 53 cards from a 52-card deck
    
    def test_deal_from_empty(self):
        """Test dealing from an empty deck."""
        deck = Deck()
        deck.deal(52)  # Deal all cards
        
        with pytest.raises(ValueError):
            deck.deal_one()  # Try to deal from an empty deck


class TestHandEvaluator:
    """Tests for the HandEvaluator class."""
    
    def test_high_card(self):
        """Test evaluation of a high card hand."""
        hand = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.CLUBS),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.TEN, Suit.DIAMONDS),
            Card(Rank.EIGHT, Suit.SPADES)
        ]
        
        rank, description = HandEvaluator.evaluate_hand(hand)
        assert rank == 0  # High card is the lowest rank
        assert "High Card" in description
        assert "Ace" in description
    
    def test_one_pair(self):
        """Test evaluation of a one pair hand."""
        hand = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.CLUBS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.DIAMONDS),
            Card(Rank.TEN, Suit.SPADES)
        ]
        
        rank, description = HandEvaluator.evaluate_hand(hand)
        assert rank == 1  # One pair rank
        assert "One Pair" in description
        assert "Aces" in description
    
    def test_two_pair(self):
        """Test evaluation of a two pair hand."""
        hand = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.CLUBS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.KING, Suit.DIAMONDS),
            Card(Rank.TEN, Suit.SPADES)
        ]
        
        rank, description = HandEvaluator.evaluate_hand(hand)
        assert rank == 2  # Two pair rank
        assert "Two Pair" in description
        assert "Aces" in description
        assert "Kings" in description
    
    def test_three_of_a_kind(self):
        """Test evaluation of a three of a kind hand."""
        hand = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.CLUBS),
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.DIAMONDS),
            Card(Rank.TEN, Suit.SPADES)
        ]
        
        rank, description = HandEvaluator.evaluate_hand(hand)
        assert rank == 3  # Three of a kind rank
        assert "Three of a Kind" in description
        assert "Aces" in description
    
    def test_straight(self):
        """Test evaluation of a straight hand."""
        hand = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.CLUBS),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.DIAMONDS),
            Card(Rank.TEN, Suit.SPADES)
        ]
        
        rank, description = HandEvaluator.evaluate_hand(hand)
        assert rank == 4  # Straight rank
        assert "Straight" in description
        assert "Ace" in description
    
    def test_flush(self):
        """Test evaluation of a flush hand."""
        hand = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.SPADES),
            Card(Rank.QUEEN, Suit.SPADES),
            Card(Rank.JACK, Suit.SPADES),
            Card(Rank.NINE, Suit.SPADES)
        ]
        
        rank, description = HandEvaluator.evaluate_hand(hand)
        assert rank == 5  # Flush rank
        assert "Flush" in description
        assert "Ace" in description
    
    def test_full_house(self):
        """Test evaluation of a full house hand."""
        hand = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.CLUBS),
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.DIAMONDS),
            Card(Rank.KING, Suit.SPADES)
        ]
        
        rank, description = HandEvaluator.evaluate_hand(hand)
        assert rank == 6  # Full house rank
        assert "Full House" in description
        assert "Aces" in description
        assert "Kings" in description
    
    def test_four_of_a_kind(self):
        """Test evaluation of a four of a kind hand."""
        hand = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.CLUBS),
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.KING, Suit.SPADES)
        ]
        
        rank, description = HandEvaluator.evaluate_hand(hand)
        assert rank == 7  # Four of a kind rank
        assert "Four of a Kind" in description
        assert "Aces" in description
    
    def test_straight_flush(self):
        """Test evaluation of a straight flush hand."""
        hand = [
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.TEN, Suit.HEARTS),
            Card(Rank.NINE, Suit.HEARTS)
        ]
        
        rank, description = HandEvaluator.evaluate_hand(hand)
        assert rank == 8  # Straight flush rank
        assert "Straight Flush" in description
        assert "King" in description
    
    def test_royal_flush(self):
        """Test evaluation of a royal flush hand."""
        hand = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.TEN, Suit.HEARTS)
        ]
        
        rank, description = HandEvaluator.evaluate_hand(hand)
        assert rank == 9  # Royal flush rank (highest)
        assert "Royal Flush" in description
        assert "Hearts" in description
    
    def test_compare_hands(self):
        """Test comparison between different hands."""
        royal_flush = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.TEN, Suit.HEARTS)
        ]
        
        straight_flush = [
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.TEN, Suit.HEARTS),
            Card(Rank.NINE, Suit.HEARTS)
        ]
        
        four_of_a_kind = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.CLUBS),
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.KING, Suit.SPADES)
        ]
        
        # Royal flush beats straight flush
        assert HandEvaluator.compare_hands(royal_flush, straight_flush) == 1
        assert HandEvaluator.compare_hands(straight_flush, royal_flush) == -1
        
        # Straight flush beats four of a kind
        assert HandEvaluator.compare_hands(straight_flush, four_of_a_kind) == 1
        assert HandEvaluator.compare_hands(four_of_a_kind, straight_flush) == -1
        
        # Royal flush beats four of a kind
        assert HandEvaluator.compare_hands(royal_flush, four_of_a_kind) == 1
        assert HandEvaluator.compare_hands(four_of_a_kind, royal_flush) == -1


class TestGameState:
    """Tests for the GameState class."""
    
    def test_initial_state(self):
        """Test initial game state setup."""
        game_state = GameState(num_players=6, small_blind=1, big_blind=2)
        
        assert game_state.num_players == 6
        assert game_state.small_blind == 1
        assert game_state.big_blind == 2
        assert len(game_state.stacks) == 6
        assert all(stack == 100 for stack in game_state.stacks)  # Default stack size
        assert len(game_state.hole_cards) == 6
        assert all(len(cards) == 2 for cards in game_state.hole_cards)  # 2 hole cards per player
        assert len(game_state.community_cards) == 0  # No community cards initially
        assert game_state.pot == 0  # Pot starts at 0
        assert game_state.stage == Stage.PREFLOP  # Start at preflop
        
        # Check blinds are posted
        assert game_state.current_bets[(game_state.button + 1) % 6] == 1  # Small blind
        assert game_state.current_bets[(game_state.button + 2) % 6] == 2  # Big blind
    
    def test_legal_actions(self):
        """Test getting legal actions."""
        game_state = GameState(num_players=3)
        
        # UTG player should be able to fold, call, or raise
        legal_actions = game_state.get_legal_actions()
        action_types = [action.action_type for action in legal_actions]
        
        assert ActionType.FOLD in action_types
        assert ActionType.CALL in action_types
        assert ActionType.RAISE in action_types
        assert ActionType.CHECK not in action_types  # Can't check facing a bet
    
    def test_apply_action(self):
        """Test applying actions to the game state."""
        game_state = GameState(num_players=3)
        initial_player = game_state.current_player
        
        # Test fold action
        fold_action = Action(ActionType.FOLD)
        new_state = game_state.apply_action(fold_action)
        
        assert not new_state.active[initial_player]  # Player should be inactive after folding
        assert new_state.current_player != initial_player  # Should move to next player
        
        # Test call action
        call_action = Action(ActionType.CALL)
        new_state = game_state.apply_action(call_action)
        
        assert new_state.current_bets[initial_player] == game_state.big_blind  # Should match the big blind
        assert new_state.stacks[initial_player] == game_state.stacks[initial_player] - game_state.big_blind
    
    def test_stage_progression(self):
        """Test progression through game stages."""
        game_state = GameState(num_players=2)
        
        # Players call/check to complete preflop
        while game_state.stage == Stage.PREFLOP:
            if game_state.current_player == -1:
                break
            
            legal_actions = game_state.get_legal_actions()
            # Prefer check, then call, then fold
            action = None
            for a in legal_actions:
                if a.action_type == ActionType.CHECK:
                    action = a
                    break
            if action is None:
                for a in legal_actions:
                    if a.action_type == ActionType.CALL:
                        action = a
                        break
            if action is None:
                action = legal_actions[0]  # Default to first legal action
            
            game_state = game_state.apply_action(action)
        
        # Should now be at flop with 3 community cards
        assert game_state.stage == Stage.FLOP
        assert len(game_state.community_cards) == 3
        
        # Continue to turn
        while game_state.stage == Stage.FLOP:
            if game_state.current_player == -1:
                break
            
            legal_actions = game_state.get_legal_actions()
            action = next((a for a in legal_actions if a.action_type == ActionType.CHECK), legal_actions[0])
            game_state = game_state.apply_action(action)
        
        # Should now be at turn with 4 community cards
        assert game_state.stage == Stage.TURN
        assert len(game_state.community_cards) == 4
    
    def test_terminal_state(self):
        """Test detecting terminal game states."""
        # Case 1: All players fold except one
        game_state = GameState(num_players=3)
        
        # First player folds
        fold_action = Action(ActionType.FOLD)
        game_state = game_state.apply_action(fold_action)
        
        # Second player folds, leaving only the big blind active
        game_state = game_state.apply_action(fold_action)
        
        # Should be terminal
        assert game_state.is_terminal()
        assert sum(game_state.active) == 1
        
        # Case 2: Play to showdown
        game_state = GameState(num_players=2)
        
        # Complete all streets with checks
        while not game_state.is_terminal():
            if game_state.current_player == -1:
                break
            
            legal_actions = game_state.get_legal_actions()
            # Prefer check/call to advance stages
            action = None
            for a in legal_actions:
                if a.action_type in {ActionType.CHECK, ActionType.CALL}:
                    action = a
                    break
            if action is None:
                action = legal_actions[0]
            
            game_state = game_state.apply_action(action)
        
        # Should be terminal with showdown
        assert game_state.is_terminal()
        assert game_state.stage == Stage.SHOWDOWN
        assert len(game_state.community_cards) == 5
    
    def test_payouts(self):
        """Test calculation of payouts."""
        # Single winner due to folds
        game_state = GameState(num_players=3)
        
        # Two players fold
        fold_action = Action(ActionType.FOLD)
        game_state = game_state.apply_action(fold_action)
        game_state = game_state.apply_action(fold_action)
        
        # Check payouts
        payouts = game_state.get_payouts()
        assert sum(payouts) == 3  # Small blind (1) + big blind (2)
        assert payouts[2] == 3  # Last player (BB) should win everything
        
        # Verify zero for folded players
        assert payouts[0] == 0
        assert payouts[1] == 0
    
    def test_observation(self):
        """Test getting observations from a player's perspective."""
        game_state = GameState(num_players=3)
        
        # Get observation for each player
        for i in range(3):
            obs = game_state.get_observation(i)
            
            # Check that hole cards are only visible to the player
            assert obs['hole_cards'] == game_state.hole_cards[i]
            
            # Check that common elements are the same
            assert obs['community_cards'] == game_state.community_cards
            assert obs['pot'] == game_state.pot
            assert obs['current_bets'] == game_state.current_bets
            assert obs['stage'] == game_state.stage
            
            # Legal actions should only be available for current player
            if i == game_state.current_player:
                assert len(obs['legal_actions']) > 0
            else:
                assert len(obs['legal_actions']) == 0
