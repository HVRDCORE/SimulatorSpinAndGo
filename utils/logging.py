"""
Logging utilities for poker simulations.

This module provides logging functionality for the poker simulation framework,
helping to track game states, agent actions, and training progress.
"""

import logging
import os
import sys
import time
from typing import Dict, List, Any, Optional, Union
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from pokersim.game.state import GameState, Action, Stage


# Configure default logger
logger = logging.getLogger("pokersim")
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging for the pokersim framework.
    
    Args:
        log_file (Optional[str], optional): Path to the log file. Defaults to None.
        level (int, optional): Logging level. Defaults to logging.INFO.
        
    Returns:
        logging.Logger: The configured logger.
    """
    logger.setLevel(level)
    
    # If log file is provided, add file handler
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class GameLogger:
    """
    Logger for poker games.
    
    This class logs game states, actions, and outcomes to help with debugging
    and analysis of poker games.
    
    Attributes:
        log_file (Optional[str]): Path to the log file.
        level (int): Logging level.
        logger (logging.Logger): The logger instance.
        game_history (List[Dict]): History of game states and actions.
    """
    
    def __init__(self, log_file: Optional[str] = None, level: int = logging.INFO):
        """
        Initialize a game logger.
        
        Args:
            log_file (Optional[str], optional): Path to the log file. Defaults to None.
            level (int, optional): Logging level. Defaults to logging.INFO.
        """
        self.log_file = log_file
        self.level = level
        self.logger = setup_logging(log_file, level)
        self.game_history = []
    
    def log_game_state(self, game_state: GameState, message: Optional[str] = None) -> None:
        """
        Log a game state.
        
        Args:
            game_state (GameState): The game state to log.
            message (Optional[str], optional): Additional message. Defaults to None.
        """
        if message:
            self.logger.info(message)
        
        # Create a serializable representation of the game state
        state_dict = {
            'stage': str(game_state.stage),
            'pot': game_state.pot,
            'community_cards': [str(card) for card in game_state.community_cards],
            'players': []
        }
        
        for i in range(game_state.num_players):
            player_dict = {
                'id': i,
                'stack': game_state.stacks[i],
                'bet': game_state.current_bets[i],
                'active': game_state.active[i],
                'hole_cards': [str(card) for card in game_state.hole_cards[i]] if game_state.hole_cards[i] else []
            }
            state_dict['players'].append(player_dict)
        
        self.game_history.append(state_dict)
        
        # Log detailed state
        self.logger.debug(f"Game state: {json.dumps(state_dict, indent=2)}")
    
    def log_action(self, player_id: int, action: Action) -> None:
        """
        Log a player action.
        
        Args:
            player_id (int): The ID of the player taking the action.
            action (Action): The action taken.
        """
        action_dict = {
            'player_id': player_id,
            'action_type': str(action.action_type),
            'amount': action.amount if hasattr(action, 'amount') else 0
        }
        
        # Add to history
        if self.game_history:
            if 'actions' not in self.game_history[-1]:
                self.game_history[-1]['actions'] = []
            self.game_history[-1]['actions'].append(action_dict)
        
        self.logger.info(f"Player {player_id} takes action: {action}")
    
    def log_game_outcome(self, game_state: GameState) -> None:
        """
        Log the outcome of a game.
        
        Args:
            game_state (GameState): The final game state.
        """
        payouts = game_state.get_payouts()
        
        outcome_dict = {
            'payouts': payouts,
            'winners': [i for i, payout in enumerate(payouts) if payout > 0]
        }
        
        # Add to history
        self.game_history.append(outcome_dict)
        
        winners_str = ", ".join([f"Player {i}" for i in outcome_dict['winners']])
        payouts_str = ", ".join([f"Player {i}: ${p}" for i, p in enumerate(payouts) if p > 0])
        
        self.logger.info(f"Game outcome - Winners: {winners_str}, Payouts: {payouts_str}")
    
    def save_history(self, file_path: str) -> None:
        """
        Save the game history to a JSON file.
        
        Args:
            file_path (str): Path to the output file.
        """
        with open(file_path, 'w') as f:
            json.dump(self.game_history, f, indent=2)
        
        self.logger.info(f"Game history saved to {file_path}")
    
    def clear_history(self) -> None:
        """Clear the game history."""
        self.game_history = []
        self.logger.debug("Game history cleared")


class TrainingLogger:
    """
    Logger for training machine learning models.
    
    This class logs training progress, metrics, and results to help with debugging
    and analysis of model training.
    
    Attributes:
        log_dir (str): Directory for log files and plots.
        log_file (Optional[str]): Path to the log file.
        level (int): Logging level.
        logger (logging.Logger): The logger instance.
        metrics (Dict[str, List]): Training metrics history.
    """
    
    def __init__(self, log_dir: str, log_file: Optional[str] = None, level: int = logging.INFO):
        """
        Initialize a training logger.
        
        Args:
            log_dir (str): Directory for log files and plots.
            log_file (Optional[str], optional): Path to the log file. Defaults to None.
            level (int, optional): Logging level. Defaults to logging.INFO.
        """
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up log file
        if log_file is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        
        self.log_file = log_file
        self.level = level
        self.logger = setup_logging(log_file, level)
        
        # Initialize metrics dictionary
        self.metrics = {}
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Log training hyperparameters.
        
        Args:
            hyperparams (Dict[str, Any]): Hyperparameters dictionary.
        """
        self.logger.info(f"Training hyperparameters: {json.dumps(hyperparams, indent=2, default=str)}")
        
        # Save hyperparameters to a file
        hyperparams_file = os.path.join(self.log_dir, "hyperparameters.json")
        with open(hyperparams_file, 'w') as f:
            json.dump(hyperparams, f, indent=2, default=str)
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log training metrics for a step.
        
        Args:
            metrics (Dict[str, float]): Metrics dictionary.
            step (int): Training step or iteration.
        """
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metrics_str}")
        
        # Add to metrics history
        for k, v in metrics.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append(v)
    
    def log_evaluation(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log evaluation metrics.
        
        Args:
            metrics (Dict[str, float]): Evaluation metrics dictionary.
            step (int): Training step or iteration.
        """
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Evaluation at step {step}: {metrics_str}")
        
        # Add to metrics history with 'eval_' prefix
        for k, v in metrics.items():
            eval_key = f"eval_{k}"
            if eval_key not in self.metrics:
                self.metrics[eval_key] = []
            self.metrics[eval_key].append(v)
    
    def save_metrics(self) -> None:
        """Save metrics to a JSON file."""
        metrics_file = os.path.join(self.log_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to {metrics_file}")
    
    def plot_metrics(self, metrics_to_plot: Optional[List[str]] = None) -> None:
        """
        Plot training metrics.
        
        Args:
            metrics_to_plot (Optional[List[str]], optional): List of metrics to plot. Defaults to None (all metrics).
        """
        if not self.metrics:
            self.logger.warning("No metrics to plot")
            return
        
        if metrics_to_plot is None:
            metrics_to_plot = list(self.metrics.keys())
        
        for metric in metrics_to_plot:
            if metric not in self.metrics:
                self.logger.warning(f"Metric '{metric}' not found in metrics history")
                continue
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics[metric])
            plt.title(f"{metric} over time")
            plt.xlabel("Steps")
            plt.ylabel(metric)
            plt.grid(True)
            
            plot_file = os.path.join(self.log_dir, f"{metric}_plot.png")
            plt.savefig(plot_file)
            plt.close()
            
            self.logger.info(f"Plot for {metric} saved to {plot_file}")
    
    def log_model_summary(self, model_summary: str) -> None:
        """
        Log a model summary.
        
        Args:
            model_summary (str): The model summary string.
        """
        self.logger.info(f"Model summary:\n{model_summary}")
    
    def log_best_model(self, metric: str, value: float, step: int) -> bool:
        """
        Log when a new best model is found.
        
        Args:
            metric (str): The metric name.
            value (float): The metric value.
            step (int): The training step.
            
        Returns:
            bool: True if this is the best model so far, False otherwise.
        """
        metric_key = f"best_{metric}"
        
        # Check if this is the best model so far
        if metric_key not in self.metrics or value > self.metrics[metric_key][-1]:
            if metric_key not in self.metrics:
                self.metrics[metric_key] = []
            
            self.metrics[metric_key].append(value)
            self.logger.info(f"New best model at step {step} with {metric} = {value:.4f}")
            return True
        
        return False
