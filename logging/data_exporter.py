"""
Data exporter for the poker simulator.

This module provides utilities for exporting poker game data and statistics
in various formats for analysis and visualization.
"""

import os
import json
import csv
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime

from pokersim.config.config_manager import get_config


class DataExporter:
    """
    Exporter for poker game data and statistics.
    
    This class provides methods for exporting hand histories, player statistics,
    and other game data in various formats for analysis and visualization.
    
    Attributes:
        logger (logging.Logger): Python logger.
        config (Dict): Configuration settings.
        export_formats (List[str]): Supported export formats.
    """
    
    def __init__(self):
        """Initialize a data exporter."""
        # Get configuration
        config = get_config()
        self.config = config.to_dict()
        
        # Set up logger
        self.logger = logging.getLogger("pokersim.exporter")
        
        # Supported export formats
        self.export_formats = ["json", "csv", "pickle", "pandas"]
    
    def export_hand_histories(self, hand_histories: List[Dict[str, Any]], 
                            format: str = "json", output_dir: Optional[str] = None) -> str:
        """
        Export hand histories to a file.
        
        Args:
            hand_histories (List[Dict[str, Any]]): Hand histories to export.
            format (str, optional): Export format. Defaults to "json".
            output_dir (Optional[str], optional): Output directory. Defaults to None.
        
        Returns:
            str: Path to the exported file.
        """
        if not hand_histories:
            self.logger.warning("No hand histories to export")
            return ""
        
        if format not in self.export_formats:
            self.logger.warning(f"Unsupported export format: {format}")
            format = "json"
        
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.join(self.config["general"]["data_dir"], "exports")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"hands_{timestamp}"
        
        # Export based on format
        if format == "json":
            filepath = os.path.join(output_dir, f"{filename}.json")
            with open(filepath, 'w') as f:
                json.dump(hand_histories, f, indent=2)
        
        elif format == "csv":
            filepath = os.path.join(output_dir, f"{filename}.csv")
            
            # Flatten hand histories for CSV
            flattened_hands = []
            for hand in hand_histories:
                flat_hand = self._flatten_dict(hand)
                flattened_hands.append(flat_hand)
            
            # Write to CSV
            if flattened_hands:
                fieldnames = flattened_hands[0].keys()
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(flattened_hands)
        
        elif format == "pickle":
            filepath = os.path.join(output_dir, f"{filename}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(hand_histories, f)
        
        elif format == "pandas":
            filepath = os.path.join(output_dir, f"{filename}.parquet")
            
            # Convert to pandas DataFrame
            try:
                # For simplicity, we'll just use the basic fields common to all hands
                data = []
                for hand in hand_histories:
                    hand_data = {
                        "hand_id": hand.get("hand_id", ""),
                        "game_type": hand.get("game_type", ""),
                        "timestamp": hand.get("timestamp", 0),
                        "pot": hand.get("results", {}).get("pot", 0),
                        "num_players": len(hand.get("players", [])),
                        "num_actions": len(hand.get("actions", [])),
                        "num_winners": len(hand.get("results", {}).get("winners", []))
                    }
                    data.append(hand_data)
                
                if data:
                    df = pd.DataFrame(data)
                    df.to_parquet(filepath)
                else:
                    self.logger.warning("No data to export")
                    return ""
            
            except Exception as e:
                self.logger.error(f"Error exporting to pandas DataFrame: {e}")
                return ""
        
        self.logger.info(f"Exported {len(hand_histories)} hand histories to {filepath}")
        return filepath
    
    def export_player_stats(self, player_stats: Dict[str, Any], 
                          format: str = "json", output_dir: Optional[str] = None) -> str:
        """
        Export player statistics to a file.
        
        Args:
            player_stats (Dict[str, Any]): Player statistics to export.
            format (str, optional): Export format. Defaults to "json".
            output_dir (Optional[str], optional): Output directory. Defaults to None.
        
        Returns:
            str: Path to the exported file.
        """
        if not player_stats:
            self.logger.warning("No player statistics to export")
            return ""
        
        if format not in self.export_formats:
            self.logger.warning(f"Unsupported export format: {format}")
            format = "json"
        
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.join(self.config["general"]["data_dir"], "exports")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"player_stats_{timestamp}"
        
        # Export based on format
        if format == "json":
            filepath = os.path.join(output_dir, f"{filename}.json")
            with open(filepath, 'w') as f:
                json.dump(player_stats, f, indent=2)
        
        elif format == "csv":
            filepath = os.path.join(output_dir, f"{filename}.csv")
            
            # Create a list of rows for CSV
            rows = []
            for player_id, stats in player_stats.items():
                row = {"player_id": player_id}
                row.update(self._flatten_dict(stats))
                rows.append(row)
            
            # Write to CSV
            if rows:
                fieldnames = rows[0].keys()
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
        
        elif format == "pickle":
            filepath = os.path.join(output_dir, f"{filename}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(player_stats, f)
        
        elif format == "pandas":
            filepath = os.path.join(output_dir, f"{filename}.parquet")
            
            # Convert to pandas DataFrame
            try:
                data = []
                for player_id, stats in player_stats.items():
                    player_data = {"player_id": player_id}
                    
                    # Extract common statistics
                    player_data["hands_played"] = stats.get("hands_played", 0)
                    player_data["hands_won"] = stats.get("hands_won", 0)
                    player_data["profit"] = stats.get("profit", 0.0)
                    player_data["vpip"] = stats.get("vpip", 0)
                    player_data["pfr"] = stats.get("pfr", 0)
                    
                    # Extract action counts
                    actions = stats.get("actions", {})
                    for action_type, count in actions.items():
                        player_data[f"action_{action_type}"] = count
                    
                    data.append(player_data)
                
                if data:
                    df = pd.DataFrame(data)
                    df.to_parquet(filepath)
                else:
                    self.logger.warning("No data to export")
                    return ""
            
            except Exception as e:
                self.logger.error(f"Error exporting to pandas DataFrame: {e}")
                return ""
        
        self.logger.info(f"Exported statistics for {len(player_stats)} players to {filepath}")
        return filepath
    
    def export_training_metrics(self, metrics: List[Dict[str, Any]], 
                              format: str = "json", output_dir: Optional[str] = None) -> str:
        """
        Export training metrics to a file.
        
        Args:
            metrics (List[Dict[str, Any]]): Training metrics to export.
            format (str, optional): Export format. Defaults to "json".
            output_dir (Optional[str], optional): Output directory. Defaults to None.
        
        Returns:
            str: Path to the exported file.
        """
        if not metrics:
            self.logger.warning("No training metrics to export")
            return ""
        
        if format not in self.export_formats:
            self.logger.warning(f"Unsupported export format: {format}")
            format = "json"
        
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.join(self.config["general"]["data_dir"], "exports")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"training_metrics_{timestamp}"
        
        # Export based on format
        if format == "json":
            filepath = os.path.join(output_dir, f"{filename}.json")
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        elif format == "csv":
            filepath = os.path.join(output_dir, f"{filename}.csv")
            
            # Flatten metrics for CSV
            flattened_metrics = []
            for metric in metrics:
                flat_metric = self._flatten_dict(metric)
                flattened_metrics.append(flat_metric)
            
            # Write to CSV
            if flattened_metrics:
                fieldnames = flattened_metrics[0].keys()
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(flattened_metrics)
        
        elif format == "pickle":
            filepath = os.path.join(output_dir, f"{filename}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(metrics, f)
        
        elif format == "pandas":
            filepath = os.path.join(output_dir, f"{filename}.parquet")
            
            # Convert to pandas DataFrame
            try:
                df = pd.DataFrame(metrics)
                df.to_parquet(filepath)
            except Exception as e:
                self.logger.error(f"Error exporting to pandas DataFrame: {e}")
                return ""
        
        self.logger.info(f"Exported {len(metrics)} training metrics to {filepath}")
        return filepath
    
    def generate_summary_report(self, session_stats: Dict[str, Any], 
                              output_dir: Optional[str] = None) -> str:
        """
        Generate a summary report from session statistics.
        
        Args:
            session_stats (Dict[str, Any]): Session statistics.
            output_dir (Optional[str], optional): Output directory. Defaults to None.
        
        Returns:
            str: Path to the report file.
        """
        if not session_stats:
            self.logger.warning("No session statistics to generate report")
            return ""
        
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.join(self.config["general"]["data_dir"], "reports")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        session_id = session_stats.get("session_id", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"report_{session_id}_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # Generate report
        with open(filepath, 'w') as f:
            # Session information
            start_time = datetime.fromtimestamp(session_stats.get("start_time", 0)).strftime("%Y-%m-%d %H:%M:%S")
            end_time = datetime.fromtimestamp(session_stats.get("end_time", 0)).strftime("%Y-%m-%d %H:%M:%S")
            duration = session_stats.get("end_time", 0) - session_stats.get("start_time", 0)
            
            f.write("=" * 80 + "\n")
            f.write(f"Session Report: {session_id}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Start Time: {start_time}\n")
            f.write(f"End Time: {end_time}\n")
            f.write(f"Duration: {duration:.2f} seconds\n")
            f.write(f"Hands Played: {session_stats.get('hands_played', 0)}\n\n")
            
            # Player statistics
            f.write("-" * 80 + "\n")
            f.write("Player Statistics\n")
            f.write("-" * 80 + "\n\n")
            
            player_stats = session_stats.get("player_stats", {})
            for player_id, stats in player_stats.items():
                f.write(f"Player {player_id}:\n")
                f.write(f"  Hands Played: {stats.get('hands_played', 0)}\n")
                f.write(f"  Hands Won: {stats.get('hands_won', 0)}\n")
                
                win_rate = stats.get("hands_won", 0) / stats.get("hands_played", 1) * 100
                f.write(f"  Win Rate: {win_rate:.2f}%\n")
                
                f.write(f"  Profit: {stats.get('profit', 0):.2f}\n")
                
                vpip = stats.get("vpip", 0) / stats.get("hands_played", 1) * 100
                pfr = stats.get("pfr", 0) / stats.get("hands_played", 1) * 100
                f.write(f"  VPIP: {vpip:.2f}%\n")
                f.write(f"  PFR: {pfr:.2f}%\n")
                
                f.write("  Actions:\n")
                actions = stats.get("actions", {})
                for action_type, count in actions.items():
                    f.write(f"    {action_type}: {count}\n")
                
                f.write("\n")
            
            # Game statistics
            f.write("-" * 80 + "\n")
            f.write("Game Statistics\n")
            f.write("-" * 80 + "\n\n")
            
            game_stats = session_stats.get("game_stats", {})
            for game_type, stats in game_stats.items():
                f.write(f"Game Type: {game_type}\n")
                f.write(f"  Hands Played: {stats.get('hands_played', 0)}\n")
                f.write(f"  Average Pot Size: {stats.get('avg_pot_size', 0):.2f}\n")
                f.write(f"  Average Winners per Hand: {stats.get('avg_winners', 0):.2f}\n")
                
                f.write("  Stages Reached:\n")
                stages = stats.get("stages_reached", {})
                for stage, count in stages.items():
                    pct = count / stats.get("hands_played", 1) * 100
                    f.write(f"    {stage}: {count} ({pct:.2f}%)\n")
                
                f.write("\n")
            
            # Tournament statistics (if available)
            if "tournaments" in session_stats:
                f.write("-" * 80 + "\n")
                f.write("Tournament Statistics\n")
                f.write("-" * 80 + "\n\n")
                
                tournament_stats = session_stats["tournaments"]
                f.write(f"Tournaments Played: {tournament_stats.get('count', 0)}\n")
                f.write(f"Average Duration: {tournament_stats.get('avg_duration', 0):.2f} seconds\n\n")
                
                f.write("Player Rankings:\n")
                rankings = tournament_stats.get("player_rankings", {})
                for player_id, stats in rankings.items():
                    f.write(f"  Player {player_id}:\n")
                    f.write(f"    Tournaments Played: {stats.get('tournaments', 0)}\n")
                    f.write(f"    Average Rank: {stats.get('avg_rank', 0):.2f}\n")
                    f.write(f"    Wins: {stats.get('wins', 0)}\n")
                    f.write(f"    Profit: {stats.get('profit', 0):.2f}\n\n")
        
        self.logger.info(f"Generated summary report at {filepath}")
        return filepath
    
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

def get_exporter() -> DataExporter:
    """
    Get the singleton data exporter instance.
    
    Returns:
        DataExporter: Data exporter instance.
    """
    global _instance
    if _instance is None:
        _instance = DataExporter()
    return _instance