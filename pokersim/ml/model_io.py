"""
Model I/O utilities for the poker simulator.

This module provides utilities for saving and loading machine learning models,
along with their metadata, for use in poker agents. It supports both PyTorch
and TensorFlow models with a unified interface.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, Union, Tuple, List
import pickle

# Conditional imports for PyTorch and TensorFlow
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from pokersim.config.config_manager import get_config

# Configure logging
logger = logging.getLogger("pokersim.ml.model_io")


class ModelIO:
    """
    Model I/O manager for saving and loading ML models.
    
    This class provides methods for saving and loading machine learning models
    for poker agents, along with their metadata. It supports both PyTorch and
    TensorFlow models with a unified interface.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings.
        save_dir (str): Directory for saving models.
        supported_frameworks (List[str]): List of supported ML frameworks.
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize the model I/O manager.
        
        Args:
            save_dir (Optional[str], optional): Directory for saving models. Defaults to None.
        """
        # Get configuration
        config = get_config()
        self.config = config.to_dict()
        
        # Set save directory
        if save_dir is None:
            self.save_dir = self.config["agents"]["save_dir"]
        else:
            self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Check available frameworks
        self.supported_frameworks = []
        if TORCH_AVAILABLE:
            self.supported_frameworks.append("pytorch")
        if TF_AVAILABLE:
            self.supported_frameworks.append("tensorflow")
        
        if not self.supported_frameworks:
            logger.warning("No supported machine learning frameworks found")
    
    def save_model(self, model: Any, model_name: str, framework: str, 
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a machine learning model and its metadata.
        
        Args:
            model (Any): Model to save (PyTorch or TensorFlow model).
            model_name (str): Name of the model.
            framework (str): Framework used by the model ('pytorch' or 'tensorflow').
            metadata (Optional[Dict[str, Any]], optional): Model metadata. Defaults to None.
        
        Returns:
            str: Path to the saved model.
        """
        if framework.lower() not in self.supported_frameworks:
            logger.error(f"Unsupported framework: {framework}")
            return ""
        
        # Create model directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_dir = os.path.join(self.save_dir, f"{model_name}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model based on framework
        if framework.lower() == "pytorch" and TORCH_AVAILABLE:
            model_path = self._save_pytorch_model(model, model_dir)
        elif framework.lower() == "tensorflow" and TF_AVAILABLE:
            model_path = self._save_tensorflow_model(model, model_dir)
        else:
            logger.error(f"Framework {framework} is not available")
            return ""
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "model_name": model_name,
            "framework": framework,
            "timestamp": timestamp,
            "save_path": model_path,
        })
        
        # Save metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {framework} model {model_name} to {model_dir}")
        return model_dir
    
    def load_model(self, model_path: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a machine learning model and its metadata.
        
        Args:
            model_path (str): Path to the model directory or file.
        
        Returns:
            Tuple[Any, Dict[str, Any]]: Tuple of (model, metadata).
        """
        # Determine if path is a directory or file
        if os.path.isdir(model_path):
            # It's a directory, look for metadata
            metadata_path = os.path.join(model_path, "metadata.json")
            if not os.path.exists(metadata_path):
                logger.error(f"No metadata found in {model_path}")
                return None, {}
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Determine framework and load model
            framework = metadata.get("framework", "").lower()
            if framework == "pytorch" and TORCH_AVAILABLE:
                model = self._load_pytorch_model(model_path, metadata)
            elif framework == "tensorflow" and TF_AVAILABLE:
                model = self._load_tensorflow_model(model_path, metadata)
            else:
                logger.error(f"Unsupported or unavailable framework: {framework}")
                return None, metadata
        
        else:
            # It's a file, try to determine the type
            _, ext = os.path.splitext(model_path)
            
            if ext == ".pt" and TORCH_AVAILABLE:
                # PyTorch model
                model = self._load_pytorch_model(model_path)
                metadata = {"framework": "pytorch", "save_path": model_path}
            
            elif (ext == ".h5" or ext == ".keras") and TF_AVAILABLE:
                # TensorFlow model
                model = self._load_tensorflow_model(model_path)
                metadata = {"framework": "tensorflow", "save_path": model_path}
            
            elif ext == ".pkl":
                # Pickle file, could be anything
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    metadata = {"framework": "pickle", "save_path": model_path}
                except Exception as e:
                    logger.error(f"Error loading pickle file: {e}")
                    return None, {}
            
            else:
                logger.error(f"Unsupported model file: {model_path}")
                return None, {}
        
        logger.info(f"Loaded model from {model_path}")
        return model, metadata
    
    def _save_pytorch_model(self, model: Any, model_dir: str) -> str:
        """
        Save a PyTorch model.
        
        Args:
            model (Any): PyTorch model to save.
            model_dir (str): Directory to save the model in.
        
        Returns:
            str: Path to the saved model.
        """
        model_path = os.path.join(model_dir, "model.pt")
        torch.save(model.state_dict(), model_path)
        
        # Also save model architecture if available
        if hasattr(model, "get_config"):
            config_path = os.path.join(model_dir, "architecture.json")
            with open(config_path, 'w') as f:
                json.dump(model.get_config(), f, indent=2)
        
        return model_path
    
    def _save_tensorflow_model(self, model: Any, model_dir: str) -> str:
        """
        Save a TensorFlow model.
        
        Args:
            model (Any): TensorFlow model to save.
            model_dir (str): Directory to save the model in.
        
        Returns:
            str: Path to the saved model.
        """
        model_path = os.path.join(model_dir, "model.keras")
        model.save(model_path)
        return model_path

    def _load_pytorch_model(self, model_path: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        model_file = os.path.join(model_path, "model.pt") if os.path.isdir(model_path) else model_path
        if not os.path.exists(model_file):
            logger.error(f"No PyTorch model found in {model_path}")
            return None

        state_dict = torch.load(model_file, map_location=torch.device('cpu'))

        if metadata and "model_class" in metadata:
            from importlib import import_module
            try:
                model_class = getattr(import_module("pokersim.ml.models"), metadata["model_class"])
                model = model_class(**metadata.get("model_args", {}))
                model.load_state_dict(state_dict)
                return model
            except (ImportError, KeyError, AttributeError) as e:
                logger.error(f"Failed to load model class: {e}")
                return state_dict

        return state_dict
    
    def _load_tensorflow_model(self, model_path: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """
        Load a TensorFlow model.
        
        Args:
            model_path (str): Path to the model file or directory.
            metadata (Optional[Dict[str, Any]], optional): Model metadata. Defaults to None.
        
        Returns:
            Any: Loaded TensorFlow model.
        """
        if os.path.isdir(model_path):
            # Look for model file in directory
            model_file = os.path.join(model_path, "model.keras")
            if not os.path.exists(model_file):
                # Try older format
                model_file = os.path.join(model_path, "model.h5")
                if not os.path.exists(model_file):
                    logger.error(f"No TensorFlow model found in {model_path}")
                    return None
        else:
            model_file = model_path
        
        # Load the model
        model = tf.keras.models.load_model(model_file)
        return model
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all saved models.
        
        Returns:
            List[Dict[str, Any]]: List of model metadata.
        """
        models = []
        
        # Check if save directory exists
        if not os.path.exists(self.save_dir):
            logger.warning(f"Save directory {self.save_dir} does not exist")
            return models
        
        # Iterate through subdirectories
        for model_dir in os.listdir(self.save_dir):
            dir_path = os.path.join(self.save_dir, model_dir)
            
            if not os.path.isdir(dir_path):
                continue
            
            # Look for metadata
            metadata_path = os.path.join(dir_path, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Add directory path
                    metadata["directory"] = dir_path
                    models.append(metadata)
                
                except Exception as e:
                    logger.warning(f"Error loading metadata from {metadata_path}: {e}")
        
        return models
    
    def get_latest_model(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Get the latest version of a model by name.
        
        Args:
            model_name (str): Name of the model.
        
        Returns:
            Tuple[Any, Dict[str, Any]]: Tuple of (model, metadata).
        """
        models = self.list_models()
        
        # Filter by model name
        matching_models = [m for m in models if m.get("model_name") == model_name]
        
        if not matching_models:
            logger.warning(f"No models found with name {model_name}")
            return None, {}
        
        # Sort by timestamp
        matching_models.sort(key=lambda m: m.get("timestamp", ""), reverse=True)
        
        # Load the latest model
        latest_model = matching_models[0]
        model_dir = latest_model["directory"]
        
        return self.load_model(model_dir)


# Singleton instance
_instance = None

def get_model_io() -> ModelIO:
    """
    Get the singleton model I/O instance.
    
    Returns:
        ModelIO: Model I/O instance.
    """
    global _instance
    if _instance is None:
        _instance = ModelIO()
    return _instance