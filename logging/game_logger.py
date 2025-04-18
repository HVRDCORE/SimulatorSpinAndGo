"""
Game logger for the poker simulator.

This module provides a specialized logger for poker games, capturing hand
histories, game events, and statistics for later analysis and visualization.
"""

import os
import json
import csv
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import uuid

from pokersim.config.config_manager import get_config


class HandHistory:
    """
    Poker hand history recorder.
    
    This class records detailed information about a poker hand,
    including players, actions, community cards, and results.
    
    Attributes:
        hand_id (str): Unique identifier for the hand.
        game_type (str): Type of poker game.
        table_id (str): Identifier for the poker table.
        timestamp (float): Unix timestamp when the hand started.
        players (List[Dict]): Information about players.
        blinds (Dict): Small and big blind amounts.
        ante (float): Ante amount, if any.
        dealer_position (int): Position of the dealer.
        actions (List[Dict]): List of actions during the hand.
        cards (Dict): Community and player cards.
        results (Dict): Results of the hand.
    """
    
    def __init__(self, game_type: str, table_id: Optional[str] = None):
        """
        Initialize a hand history record.
        
        Args:
            game_type (str): Type of poker game.
            table_id (Optional[str], optional): Table identifier. Defaults to None.
        """
        self.hand_id = str(uuid.uuid4())
        self.game_type = game_type
        self.table_id = table_id if table_id else f"Table-{uuid.uuid4().hex[:8]}"
        self.timestamp = time.time()
        self.players = []
        self.blinds = {"small": 0, "big": 0}
        self.ante = 0
        self.dealer_position = 0
        self.actions = []
        self.cards = {
            "community": [],
            "players": {}
        }
        self.results = {
            "pot": 0,
            "winners": [],
            "player_stacks": {}
        }
    
    def set_table_info(self, blinds: Dict[str, float], ante: float, dealer_position: int):
        """
        Set table information.
        
        Args:
            blinds (Dict[str, float]): Small and big blind amounts.
            ante (float): Ante amount.
            dealer_position (int): Position of the dealer.
        """
        self.blinds = blinds
        self.ante = ante
        self.dealer_position = dealer_position
    
    def add_player(self, player_id: int, position: int, stack: float, name: Optional[str] = None):
        """
        Add a player to the hand.
        
        Args:
            player_id (int): Player identifier.
            position (int): Player position at the table.
            stack (float): Player's stack size.
            name (Optional[str], optional): Player name. Defaults to None.
        """
        player_info = {
            "id": player_id,
            "position": position,
            "stack": stack,
            "name": name if name else f"Player-{player_id}"
        }
        self.players.append(player_info)
        
        # Initialize player result
        self.results["player_stacks"][player_id] = stack
    
    def set_player_cards(self, player_id: int, cards: List[int]):
        """
        Set a player's hole cards.
        
        Args:
            player_id (int): Player identifier.
            cards (List[int]): Hole cards.
        """
        self.cards["players"][player_id] = cards
    
    def set_community_cards(self, cards: List[int], stage: str):
        """
        Set community cards for a stage.
        
        Args:
            cards (List[int]): Community cards.
            stage (str): Stage of the hand (flop, turn, river).
        """
        # Store community cards by stage
        self.cards["community"] = cards
        self.cards[stage] = cards
    
    def add_action(self, player_id: int, action_type: str, amount: float, stage: str):
        """
        Add a player action to the hand.
        
        Args:
            player_id (int): Player identifier.
            action_type (str): Type of action.
            amount (float): Amount involved in the action.
            stage (str): Stage of the hand.
        """
        action = {
            "player_id": player_id,
            "type": action_type,
            "amount": amount,
            "stage": stage,
            "timestamp": time.time()
        }
        self.actions.append(action)
    
    def set_results(self, pot: float, winners: List[Dict[str, Any]], player_stacks: Dict[int, float]):
        """
        Set the results of the hand.
        
        Args:
            pot (float): Total pot size.
            winners (List[Dict[str, Any]]): List of winners.
            player_stacks (Dict[int, float]): Final stack sizes.
        """
        self.results["pot"] = pot
        self.results["winners"] = winners
        self.results["player_stacks"] = player_stacks
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert hand history to a dictionary.
        
        Returns:
            Dict[str, Any]: Hand history as a dictionary.
        """
        return {
            "hand_id": self.hand_id,
            "game_type": self.game_type,
            "table_id": self.table_id,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "players": self.players,
            "blinds": self.blinds,
            "ante": self.ante,
            "dealer_position": self.dealer_position,
            "actions": self.actions,
            "cards": self.cards,
            "results": self.results
        }
    
    def get_player_summary(self, player_id: int) -> Dict[str, Any]:
        """
        Get a summary of a player's participation in the hand.
        
        Args:
            player_id (int): Player identifier.
        
        Returns:
            Dict[str, Any]: Player summary.
        """
        # Find player
        player = None
        for p in self.players:
            if p["id"] == player_id:
                player = p
                break
        
        if not player:
            return {}
        
        # Get player actions
        player_actions = [a for a in self.actions if a["player_id"] == player_id]
        
        # Calculate profit/loss
        initial_stack = player["stack"]
        final_stack = self.results["player_stacks"].get(player_id, initial_stack)
        profit = final_stack - initial_stack
        
        # Determine win status
        won = False
        for winner in self.results["winners"]:
            if winner["player_id"] == player_id:
                won = True
                break
        
        return {
            "player_id": player_id,
            "position": player["position"],
            "initial_stack": initial_stack,
            "final_stack": final_stack,
            "profit": profit,
            "won": won,
            "cards": self.cards["players"].get(player_id, []),
            "actions": player_actions
        }


class GameLogger:
    """
    Logger for poker games and tournaments.
    
    This class provides methods for logging game events, saving hand histories,
    and collecting statistics for analysis.
    
    Attributes:
        logger (logging.Logger): Python logger for text logs.
        config (Dict): Configuration settings.
        session_id (str): Unique identifier for the session.
        hand_counter (int): Counter for hands in the session.
        current_hand (Optional[HandHistory]): Current hand being recorded.
        hand_histories (List[HandHistory]): All recorded hand histories.
        stats (Dict): Session statistics.
    """
    
    def __init__(self, session_id: Optional[str] = None, log_level: Optional[str] = None):
        """
        Initialize a game logger.
        
        Args:
            session_id (Optional[str], optional): Session identifier. Defaults to None.
            log_level (Optional[str], optional): Logging level. Defaults to None.
        """
        # Get configuration
        config = get_config()
        self.config = config.to_dict()
        
        # Set up Python logger
        log_level_str = log_level if log_level else config.get("general.log_level", "INFO")
        log_level_value = getattr(logging, log_level_str.upper(), logging.INFO)
        
        self.logger = logging.getLogger("pokersim.game")
        self.logger.setLevel(log_level_value)
        
        # Session information
        self.session_id = session_id if session_id else f"Session-{uuid.uuid4().hex[:8]}"
        self.hand_counter = 0
        self.current_hand = None
        
        # Storage for hand histories and statistics
        self.hand_histories = []
        self.stats = {
            "session_id": self.session_id,
            "start_time": time.time(),
            "end_time": None,
            "hands_played": 0,
            "player_stats": {},
            "game_stats": {}
        }
        
        self.logger.info(f"Starting new game logging session: {self.session_id}")
    
    def start_hand(self, game_type: str, table_id: Optional[str] = None) -> HandHistory:
        """
        Start recording a new hand.
        
        Args:
            game_type (str): Type of poker game.
            table_id (Optional[str], optional): Table identifier. Defaults to None.
        
        Returns:
            HandHistory: The new hand history object.
        """
        self.hand_counter += 1
        self.current_hand = HandHistory(game_type, table_id)
        self.logger.info(f"Starting hand #{self.hand_counter} - ID: {self.current_hand.hand_id}")
        return self.current_hand
    
    def end_hand(self) -> Optional[HandHistory]:
        """
        End the current hand and store its history.
        
        Returns:
            Optional[HandHistory]: The completed hand history.
        """
        if self.current_hand:
            self.hand_histories.append(self.current_hand)
            self.stats["hands_played"] += 1
            
            # Log basic information about the hand
            winners_str = ", ".join([f"Player-{w['player_id']}" for w in self.current_hand.results["winners"]])
            self.logger.info(f"Hand #{self.hand_counter} completed - Winners: {winners_str}, Pot: {self.current_hand.results['pot']}")
            
            # Save hand history if configured to do so
            if self.config["data"]["save_hands"]:
                self._save_hand_history(self.current_hand)
            
            # Update statistics
            self._update_stats(self.current_hand)
            
            result = self.current_hand
            self.current_hand = None
            return result
        
        return None
    
    def log_action(self, player_id: int, action_type: str, amount: float, stage: str):
        """
        Log a player action.
        
        Args:
            player_id (int): Player identifier.
            action_type (str): Type of action.
            amount (float): Amount involved in the action.
            stage (str): Stage of the hand.
        """
        if self.current_hand:
            self.current_hand.add_action(player_id, action_type, amount, stage)
            self.logger.debug(f"Player {player_id} {action_type} {amount} at {stage}")
    
    def log_cards(self, community_cards: List[int], stage: str):
        """
        Log community cards.
        
        Args:
            community_cards (List[int]): Community cards.
            stage (str): Stage of the hand.
        """
        if self.current_hand:
            self.current_hand.set_community_cards(community_cards, stage)
            cards_str = ", ".join([str(c) for c in community_cards])
            self.logger.debug(f"{stage.capitalize()}: {cards_str}")
    
    def log_player_cards(self, player_id: int, cards: List[int]):
        """
        Log a player's hole cards.
        
        Args:
            player_id (int): Player identifier.
            cards (List[int]): Hole cards.
        """
        if self.current_hand:
            self.current_hand.set_player_cards(player_id, cards)
            cards_str = ", ".join([str(c) for c in cards])
            self.logger.debug(f"Player {player_id} cards: {cards_str}")
    
    def log_results(self, pot: float, winners: List[Dict[str, Any]], player_stacks: Dict[int, float]):
        """
        Log the results of a hand.
        
        Args:
            pot (float): Total pot size.
            winners (List[Dict[str, Any]]): List of winners.
            player_stacks (Dict[int, float]): Final stack sizes.
        """
        if self.current_hand:
            self.current_hand.set_results(pot, winners, player_stacks)
            
            # Log detailed results
            winners_str = ", ".join([f"Player-{w['player_id']} ({w.get('amount', 0)})" for w in winners])
            self.logger.info(f"Hand results - Pot: {pot}, Winners: {winners_str}")
            
            for player_id, stack in player_stacks.items():
                self.logger.debug(f"Player {player_id} stack: {stack}")
    
    def log_tournament_result(self, tournament_id: str, players: List[Dict[str, Any]], 
                           prizes: List[float], duration: float):
        """
        Log the results of a tournament.
        
        Args:
            tournament_id (str): Tournament identifier.
            players (List[Dict[str, Any]]): List of players and their rankings.
            prizes (List[float]): Prize pool distribution.
            duration (float): Duration of the tournament in seconds.
        """
        tournament_info = {
            "tournament_id": tournament_id,
            "timestamp": time.time(),
            "duration": duration,
            "players": players,
            "prizes": prizes
        }
        
        # Log tournament result
        winner = next((p for p in players if p["rank"] == 1), None)
        winner_str = f"Player-{winner['id']}" if winner else "Unknown"
        self.logger.info(f"Tournament {tournament_id} completed - Winner: {winner_str}, Duration: {duration:.2f}s")
        
        # Save tournament data
        if self.config["data"]["save_hands"]:
            self._save_tournament_result(tournament_info)
        
        # Update statistics
        self._update_tournament_stats(tournament_info)
    
    def log_training_metrics(self, algorithm: str, metrics: Dict[str, Any], step: int):
        """
        Log training metrics.
        
        Args:
            algorithm (str): Training algorithm.
            metrics (Dict[str, Any]): Training metrics.
            step (int): Training step or iteration.
        """
        # Log metrics
        metrics_str = ", ".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        self.logger.info(f"Training {algorithm} - Step {step} - {metrics_str}")
        
        # Add to statistics
        if "training" not in self.stats:
            self.stats["training"] = {}
        
        if algorithm not in self.stats["training"]:
            self.stats["training"][algorithm] = []
        
        self.stats["training"][algorithm].append({
            "step": step,
            "timestamp": time.time(),
            **metrics
        })
    
    def end_session(self):
        """End the current logging session and finalize statistics."""
        self.stats["end_time"] = time.time()
        duration = self.stats["end_time"] - self.stats["start_time"]
        
        self.logger.info(f"Session {self.session_id} completed - Hands played: {self.stats['hands_played']}, Duration: {duration:.2f}s")
        
        # Save session statistics
        if self.config["data"]["collect_stats"]:
            self._save_session_stats()
    
    def _save_hand_history(self, hand: HandHistory):
        """
        Save a hand history to disk.
        
        Args:
            hand (HandHistory): Hand history to save.
        """
        data_dir = self.config["general"]["data_dir"]
        format = self.config["data"]["save_format"].lower()
        
        # Create directory if it doesn't exist
        session_dir = os.path.join(data_dir, self.session_id, "hands")
        os.makedirs(session_dir, exist_ok=True)
        
        # Convert hand to dictionary
        hand_dict = hand.to_dict()
        
        # Save in the specified format
        if format == "json":
            filepath = os.path.join(session_dir, f"{hand.hand_id}.json")
            with open(filepath, 'w') as f:
                json.dump(hand_dict, f, indent=2)
        elif format == "csv":
            # For CSV, we need to flatten the dictionary
            csv_data = self._flatten_dict(hand_dict)
            
            filepath = os.path.join(session_dir, f"{hand.hand_id}.csv")
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_data.keys())
                writer.writeheader()
                writer.writerow(csv_data)
        else:
            self.logger.warning(f"Unsupported hand history format: {format}")
    
    def _save_tournament_result(self, tournament_info: Dict[str, Any]):
        """
        Save tournament results to disk.
        
        Args:
            tournament_info (Dict[str, Any]): Tournament information.
        """
        data_dir = self.config["general"]["data_dir"]
        format = self.config["data"]["save_format"].lower()
        
        # Create directory if it doesn't exist
        session_dir = os.path.join(data_dir, self.session_id, "tournaments")
        os.makedirs(session_dir, exist_ok=True)
        
        # Save in the specified format
        if format == "json":
            filepath = os.path.join(session_dir, f"{tournament_info['tournament_id']}.json")
            with open(filepath, 'w') as f:
                json.dump(tournament_info, f, indent=2)
        elif format == "csv":
            # For CSV, we need to flatten the dictionary
            csv_data = self._flatten_dict(tournament_info)
            
            filepath = os.path.join(session_dir, f"{tournament_info['tournament_id']}.csv")
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_data.keys())
                writer.writeheader()
                writer.writerow(csv_data)
        else:
            self.logger.warning(f"Unsupported tournament result format: {format}")
    
    def _save_session_stats(self):
        """Save session statistics to disk."""
        data_dir = self.config["general"]["data_dir"]
        
        # Create directory if it doesn't exist
        stats_dir = os.path.join(data_dir, "stats")
        os.makedirs(stats_dir, exist_ok=True)
        
        # Save statistics as JSON
        filepath = os.path.join(stats_dir, f"{self.session_id}_stats.json")
        with open(filepath, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def _update_stats(self, hand: HandHistory):
        """
        Update session statistics with a completed hand.
        
        Args:
            hand (HandHistory): Completed hand history.
        """
        # Update per-player statistics
        for player in hand.players:
            player_id = player["id"]
            
            if player_id not in self.stats["player_stats"]:
                self.stats["player_stats"][player_id] = {
                    "hands_played": 0,
                    "hands_won": 0,
                    "profit": 0.0,
                    "vpip": 0,  # Voluntarily Put In Pot
                    "pfr": 0,   # Pre-Flop Raise
                    "actions": {
                        "fold": 0,
                        "check": 0,
                        "call": 0,
                        "bet": 0,
                        "raise": 0
                    }
                }
            
            # Get player summary for this hand
            summary = hand.get_player_summary(player_id)
            
            # Update statistics
            self.stats["player_stats"][player_id]["hands_played"] += 1
            
            if summary.get("won", False):
                self.stats["player_stats"][player_id]["hands_won"] += 1
            
            self.stats["player_stats"][player_id]["profit"] += summary.get("profit", 0)
            
            # Update action counts
            for action in summary.get("actions", []):
                action_type = action["type"]
                if action_type in self.stats["player_stats"][player_id]["actions"]:
                    self.stats["player_stats"][player_id]["actions"][action_type] += 1
            
            # Check for voluntary pot contribution and pre-flop raise
            preflop_actions = [a for a in summary.get("actions", []) if a["stage"] == "preflop"]
            if any(a["type"] in ["call", "bet", "raise"] for a in preflop_actions):
                self.stats["player_stats"][player_id]["vpip"] += 1
            
            if any(a["type"] == "raise" for a in preflop_actions):
                self.stats["player_stats"][player_id]["pfr"] += 1
        
        # Update game statistics
        game_type = hand.game_type
        if game_type not in self.stats["game_stats"]:
            self.stats["game_stats"][game_type] = {
                "hands_played": 0,
                "avg_pot_size": 0.0,
                "avg_winners": 0.0,
                "stages_reached": {
                    "preflop": 0,
                    "flop": 0,
                    "turn": 0,
                    "river": 0,
                    "showdown": 0
                }
            }
        
        self.stats["game_stats"][game_type]["hands_played"] += 1
        
        # Update average pot size
        prev_avg = self.stats["game_stats"][game_type]["avg_pot_size"]
        prev_count = self.stats["game_stats"][game_type]["hands_played"] - 1
        new_pot = hand.results["pot"]
        
        if prev_count == 0:
            self.stats["game_stats"][game_type]["avg_pot_size"] = new_pot
        else:
            self.stats["game_stats"][game_type]["avg_pot_size"] = (prev_avg * prev_count + new_pot) / self.stats["game_stats"][game_type]["hands_played"]
        
        # Update average winners
        prev_avg = self.stats["game_stats"][game_type]["avg_winners"]
        num_winners = len(hand.results["winners"])
        
        if prev_count == 0:
            self.stats["game_stats"][game_type]["avg_winners"] = num_winners
        else:
            self.stats["game_stats"][game_type]["avg_winners"] = (prev_avg * prev_count + num_winners) / self.stats["game_stats"][game_type]["hands_played"]
        
        # Update stages reached
        stages = set(a["stage"] for a in hand.actions)
        for stage in stages:
            if stage in self.stats["game_stats"][game_type]["stages_reached"]:
                self.stats["game_stats"][game_type]["stages_reached"][stage] += 1
        
        # Check for showdown
        if any(a["type"] == "showdown" for a in hand.actions):
            self.stats["game_stats"][game_type]["stages_reached"]["showdown"] += 1
    
    def _update_tournament_stats(self, tournament_info: Dict[str, Any]):
        """
        Update session statistics with tournament results.
        
        Args:
            tournament_info (Dict[str, Any]): Tournament information.
        """
        if "tournaments" not in self.stats:
            self.stats["tournaments"] = {
                "count": 0,
                "avg_duration": 0.0,
                "player_rankings": {}
            }
        
        # Update tournament count
        self.stats["tournaments"]["count"] += 1
        
        # Update average duration
        prev_avg = self.stats["tournaments"]["avg_duration"]
        prev_count = self.stats["tournaments"]["count"] - 1
        new_duration = tournament_info["duration"]
        
        if prev_count == 0:
            self.stats["tournaments"]["avg_duration"] = new_duration
        else:
            self.stats["tournaments"]["avg_duration"] = (prev_avg * prev_count + new_duration) / self.stats["tournaments"]["count"]
        
        # Update player rankings
        for player in tournament_info["players"]:
            player_id = player["id"]
            
            if player_id not in self.stats["tournaments"]["player_rankings"]:
                self.stats["tournaments"]["player_rankings"][player_id] = {
                    "tournaments": 0,
                    "avg_rank": 0.0,
                    "wins": 0,
                    "profit": 0.0
                }
            
            self.stats["tournaments"]["player_rankings"][player_id]["tournaments"] += 1
            
            # Update average rank
            prev_avg = self.stats["tournaments"]["player_rankings"][player_id]["avg_rank"]
            prev_count = self.stats["tournaments"]["player_rankings"][player_id]["tournaments"] - 1
            new_rank = player["rank"]
            
            if prev_count == 0:
                self.stats["tournaments"]["player_rankings"][player_id]["avg_rank"] = new_rank
            else:
                self.stats["tournaments"]["player_rankings"][player_id]["avg_rank"] = (prev_avg * prev_count + new_rank) / self.stats["tournaments"]["player_rankings"][player_id]["tournaments"]
            
            # Update wins
            if player["rank"] == 1:
                self.stats["tournaments"]["player_rankings"][player_id]["wins"] += 1
            
            # Update profit (prize - buy_in)
            prize = tournament_info["prizes"][player["rank"] - 1] if player["rank"] <= len(tournament_info["prizes"]) else 0
            buy_in = tournament_info.get("buy_in", 0)
            profit = prize - buy_in
            
            self.stats["tournaments"]["player_rankings"][player_id]["profit"] += profit
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Flatten a nested dictionary.
        
        Args:
            d (Dict[str, Any]): Dictionary to flatten.
            parent_key (str, optional): Parent key. Defaults to ''.
            sep (str, optional): Separator between keys. Defaults to '_'.
        
        Returns:
            Dict[str, Any]: Flattened dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(self._flatten_dict(item, f"{new_key}{sep}{i}", sep).items())
                    else:
                        items.append((f"{new_key}{sep}{i}", item))
            else:
                items.append((new_key, v))
        
        return dict(items)


# Singleton instance
_instance = None

def get_logger() -> GameLogger:
    """
    Get the singleton game logger instance.
    
    Returns:
        GameLogger: Game logger instance.
    """
    global _instance
    if _instance is None:
        _instance = GameLogger()
    return _instance