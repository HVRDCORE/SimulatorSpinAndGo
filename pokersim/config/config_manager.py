"""
Configuration manager for the poker simulator.

This module provides a centralized configuration system that allows loading
configuration from multiple sources (files, environment variables, command-line)
and supports hierarchical configuration with defaults.
"""

import os
import json
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not installed. YAML configuration files will not be supported.")
import argparse
from typing import Dict, Any, Optional, List, Union
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Configuration manager for the poker simulator.
    
    This class handles loading and managing configuration from multiple sources,
    with support for hierarchical configuration and defaults.
    
    Attributes:
        config (Dict[str, Any]): Current configuration.
        config_paths (List[str]): Paths to search for configuration files.
    """
    
    def __init__(self, config_paths: Optional[List[str]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_paths (Optional[List[str]], optional): Paths to search for configuration files.
                                                       Defaults to None.
        """
        self.config = {}
        
        # Default configuration search paths
        if config_paths is None:
            self.config_paths = ["./config", "~/.pokersim"]
            if os.access("/etc/pokersim", os.R_OK):
                self.config_paths.append("/etc/pokersim")
            else:
                logger.warning("Skipping /etc/pokersim due to insufficient permissions")
        else:
            self.config_paths = config_paths
        
        # Add path from environment variable if specified
        env_config_path = os.environ.get("POKERSIM_CONFIG_PATH")
        if env_config_path:
            self.config_paths.insert(0, env_config_path)
        
        # Load default configuration
        self._load_defaults()
    
    def _load_defaults(self):
        """Load default configuration values."""
        # General simulation settings
        self.config["general"] = {
            "random_seed": None,  # Random seed for reproducibility (None = random)
            "log_level": "INFO",  # Logging level
            "log_file": None,     # Log file path (None = stdout)
            "data_dir": "./data"  # Directory for storing simulation data
        }
        
        # Game rules settings
        self.config["game"] = {
            "variant": "texas_holdem",  # Poker variant
            "stakes": {
                "small_blind": 1,
                "big_blind": 2,
                "ante": 0
            },
            "stack_size": 200,  # Starting stack size
            "table_size": 6     # Number of players
        }
        
        # Tournament settings
        self.config["tournament"] = {
            "enabled": False,        # Whether to run in tournament mode
            "format": "spin_and_go",  # Tournament format
            "buy_in": 10,            # Tournament buy-in
            "multipliers": [2, 6, 25, 120, 240, 1200],  # Prize multipliers
            "multiplier_weights": [75, 15, 5, 4, 0.9, 0.1],  # Probability weights
            "level_duration": 180,   # Duration of blind levels in seconds
            "starting_chips": 500    # Starting chips in tournaments
        }
        
        # Agent settings
        self.config["agents"] = {
            "default_agent": "rule_based",  # Default agent type
            "agent_types": ["random", "call", "rule_based", "advanced", "mcts", "cfr", "deep_cfr", "ppo", "nfsp"],
            "evaluation_episodes": 1000,    # Number of episodes for evaluation
            "save_dir": "./saved_models"    # Directory for saving trained models
        }
        
        # Training settings
        self.config["training"] = {
            "algorithm": "deep_cfr",    # Default training algorithm
            "batch_size": 128,          # Batch size for training
            "learning_rate": 0.0001,    # Learning rate
            "num_iterations": 10000,    # Number of training iterations
            "eval_frequency": 500,      # Evaluation frequency
            "checkpoint_frequency": 1000,  # Checkpoint saving frequency
            "use_gpu": True,            # Whether to use GPU acceleration
            "distributed": False,       # Whether to use distributed training
            "num_workers": 1            # Number of workers for distributed training
        }
        
        # Model architecture settings
        self.config["model"] = {
            "value_network": {
                "input_dim": 128,
                "hidden_layers": [256, 256],
                "output_dim": 1
            },
            "policy_network": {
                "input_dim": 128,
                "hidden_layers": [256, 256],
                "output_dim": 5
            }
        }
        
        # Data collection and logging
        self.config["data"] = {
            "save_hands": False,      # Whether to save hand histories
            "save_format": "json",    # Format for saving hand histories
            "collect_stats": True,    # Whether to collect statistics
            "export_interval": 100    # Interval for exporting data
        }

        os.makedirs(self.config["general"]["data_dir"], exist_ok=True)
        os.makedirs(self.config["agents"]["save_dir"], exist_ok=True)
        self._setup_logging()
        logger.info("Configuration loaded successfully")

    def load_from_file(self, filepath: str) -> bool:
        """
        Load configuration from a file.
        
        Args:
            filepath (str): Path to the configuration file.
        
        Returns:
            bool: Whether the file was successfully loaded.
        """
        try:
            path = Path(filepath).expanduser().resolve()
            
            if not path.exists():
                logger.warning(f"Configuration file not found: {filepath}")
                return False
            
            # Determine file format based on extension
            ext = path.suffix.lower()

            if ext in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    logger.error("YAML support is not available. Install PyYAML.")
                    return False
                with open(path, 'r') as f:
                    config_data = yaml.safe_load(f)
            elif ext in ['.json']:
                with open(path, 'r') as f:
                    config_data = json.load(f)
            else:
                logger.error(f"Unsupported configuration file format: {ext}")
                return False
            
            # Update configuration
            self._update_config(config_data)
            logger.info(f"Loaded configuration from {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading configuration from {filepath}: {e}")
            return False

    def load_from_env(self, prefix: str = "POKERSIM_"):
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                parts = config_key.split("__")
                try:
                    # Try JSON parsing
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    # Handle common string values
                    if value.lower() == "true":
                        parsed_value = True
                    elif value.lower() == "false":
                        parsed_value = False
                    elif value.isdigit():
                        parsed_value = int(value)
                    elif value.replace(".", "", 1).isdigit():
                        parsed_value = float(value)
                    else:
                        parsed_value = value
                self._set_config_value(parts, parsed_value)
        
        logger.info(f"Loaded configuration from environment variables with prefix {prefix}")
    
    def load_from_args(self, args: Optional[List[str]] = None):
        """
        Load configuration from command-line arguments.
        
        Args:
            args (Optional[List[str]], optional): Command-line arguments. Defaults to None.
        """
        parser = argparse.ArgumentParser(description="Poker Simulator Configuration")
        
        # Add arguments for commonly used configuration options
        parser.add_argument("--config", "-c", help="Path to configuration file")
        parser.add_argument("--log-level", help="Logging level")
        parser.add_argument("--data-dir", help="Data directory")
        parser.add_argument("--variant", help="Poker variant")
        parser.add_argument("--stack-size", type=int, help="Starting stack size")
        parser.add_argument("--table-size", type=int, help="Table size (number of players)")
        parser.add_argument("--agent", help="Agent type to use")
        parser.add_argument("--tournament", action="store_true", help="Run in tournament mode")
        parser.add_argument("--use-gpu", action="store_true", help="Use GPU acceleration")
        parser.add_argument("--random-seed", type=int, help="Random seed for reproducibility")
        
        # Parse arguments
        parsed_args = parser.parse_args(args)
        
        # Load configuration file if specified
        if parsed_args.config:
            self.load_from_file(parsed_args.config)
        
        # Update configuration with command-line arguments
        if parsed_args.log_level:
            self.config["general"]["log_level"] = parsed_args.log_level
        
        if parsed_args.data_dir:
            self.config["general"]["data_dir"] = parsed_args.data_dir
        
        if parsed_args.variant:
            self.config["game"]["variant"] = parsed_args.variant
        
        if parsed_args.stack_size:
            self.config["game"]["stack_size"] = parsed_args.stack_size
        
        if parsed_args.table_size:
            self.config["game"]["table_size"] = parsed_args.table_size
        
        if parsed_args.agent:
            self.config["agents"]["default_agent"] = parsed_args.agent
        
        if parsed_args.tournament:
            self.config["tournament"]["enabled"] = True
        
        if parsed_args.use_gpu:
            self.config["training"]["use_gpu"] = True
        
        if parsed_args.random_seed is not None:
            self.config["general"]["random_seed"] = parsed_args.random_seed
        
        logger.info("Loaded configuration from command-line arguments")
    
    def load(self):
        """
        Load configuration from all sources.
        
        This method loads configuration from default files, environment variables,
        and command-line arguments, in that order (later sources override earlier ones).
        """
        # Try to load from default configuration files
        for path in self.config_paths:
            expanded_path = os.path.expanduser(path)
            
            # Try YAML
            yaml_path = os.path.join(expanded_path, "config.yaml")
            if os.path.exists(yaml_path):
                self.load_from_file(yaml_path)
            
            # Try JSON
            json_path = os.path.join(expanded_path, "config.json")
            if os.path.exists(json_path):
                self.load_from_file(json_path)
        
        # Load from environment variables
        self.load_from_env()
        
        # Load from command-line arguments
        self.load_from_args()
        
        # Set up logging based on configuration
        self._setup_logging()
        
        logger.info("Configuration loaded successfully")
    
    def _setup_logging(self):
        """Set up logging based on configuration."""
        log_level = getattr(logging, self.config["general"]["log_level"].upper(), logging.INFO)
        log_file = self.config["general"]["log_file"]
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Only reconfigure if no handlers are set or explicitly needed
        if not root_logger.handlers:
            root_logger.setLevel(log_level)
            handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
    
    def _update_config(self, config_data: Dict[str, Any]):
        """
        Update configuration with new data.
        
        Args:
            config_data (Dict[str, Any]): New configuration data.
        """
        def update_dict(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    # Recursively update nested dictionaries
                    update_dict(target[key], value)
                else:
                    # Replace or add value
                    target[key] = value
        
        update_dict(self.config, config_data)
    
    def _set_config_value(self, keys: List[str], value: Any):
        """
        Set a configuration value at the specified path.
        
        Args:
            keys (List[str]): Path to the configuration value.
            value (Any): Value to set.
        """
        # Start from the root of the configuration
        current = self.config
        
        # Navigate to the parent of the leaf value
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the leaf value
        current[keys[-1]] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by key path.
        
        Args:
            key_path (str): Path to the configuration value, using dots for hierarchy.
            default (Any, optional): Default value if key not found. Defaults to None.
        
        Returns:
            Any: Configuration value.
        """
        keys = key_path.split('.')
        current = self.config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set(self, key_path: str, value: Any):
        """
        Set a configuration value by key path.
        
        Args:
            key_path (str): Path to the configuration value, using dots for hierarchy.
            value (Any): Value to set.
        """
        keys = key_path.split('.')
        self._set_config_value(keys, value)
    
    def save_to_file(self, filepath: str, format: str = "yaml"):
        """
        Save current configuration to a file.
        
        Args:
            filepath (str): Path to the output file.
            format (str, optional): Output format ('yaml' or 'json'). Defaults to "yaml".
        
        Returns:
            bool: Whether the file was successfully saved.
        """
        try:
            path = Path(filepath).expanduser().resolve()
            os.makedirs(path.parent, exist_ok=True)
            
            if format.lower() == "yaml":
                with open(path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif format.lower() == "json":
                with open(path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                logger.error(f"Unsupported output format: {format}")
                return False
            
            logger.info(f"Saved configuration to {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving configuration to {filepath}: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        return self.config.copy()


# Singleton instance
_instance = None

def get_config() -> ConfigManager:
    """
    Get the singleton configuration manager instance.
    
    Returns:
        ConfigManager: Configuration manager instance.
    """
    global _instance
    if _instance is None:
        _instance = ConfigManager()
        _instance.load()
    return _instance