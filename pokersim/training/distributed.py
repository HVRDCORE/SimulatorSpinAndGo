"""
Distributed training utilities for the poker simulator.

This module provides utilities for distributed training of poker agents,
including multi-GPU and multi-machine training using PyTorch and TensorFlow.
"""

import os
import logging
import json
import time
import socket
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import multiprocessing
from functools import partial

# Conditional imports for distributed training libraries
try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from pokersim.config.config_manager import get_config
from pokersim.utils.gpu_optimization import get_gpu_manager
from pokersim.ml.model_io import get_model_io

# Configure logging
logger = logging.getLogger("pokersim.training.distributed")


class DistributedTrainer:
    """
    Distributed trainer for poker agents.
    
    This class provides methods for training poker agents in a distributed
    manner, using multiple GPUs or machines.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings.
        world_size (int): Number of training processes.
        framework (str): Deep learning framework being used.
        gpu_manager: GPU resource manager.
        model_io: Model I/O manager.
    """
    
    def __init__(self, world_size: Optional[int] = None, framework: str = "auto"):
        """
        Initialize the distributed trainer.
        
        Args:
            world_size (Optional[int], optional): Number of processes. Defaults to None.
            framework (str, optional): Deep learning framework to use. Defaults to "auto".
        """
        # Get configuration
        config = get_config()
        self.config = config.to_dict()
        
        # Get GPU manager and model I/O
        self.gpu_manager = get_gpu_manager()
        self.model_io = get_model_io()
        
        # Set world size (number of processes)
        if world_size is None:
            self.world_size = self.config["training"]["num_workers"]
        else:
            self.world_size = world_size
        
        # Set framework
        if framework == "auto":
            self.framework = self.gpu_manager.framework
        else:
            self.framework = framework
        
        # Check if distributed training is possible
        self.can_distribute = self._check_distributed_availability()
        
        if not self.can_distribute:
            logger.warning("Distributed training not available")
    
    def _check_distributed_availability(self) -> bool:
        """
        Check if distributed training is available.
        
        Returns:
            bool: Whether distributed training is available.
        """
        # Check for supported frameworks
        framework_available = (
            (self.framework == "pytorch" and TORCH_AVAILABLE) or
            (self.framework == "tensorflow" and TF_AVAILABLE)
        )
        if not framework_available:
            logger.warning(f"Framework {self.framework} not available")
            return False
        
        # Check for GPUs
        if not self.gpu_manager.use_gpu:
            logger.warning("GPU acceleration disabled")
            return self.world_size > 1  # Can still distribute across CPUs
        
        # Check for sufficient GPU devices
        num_devices = len(self.gpu_manager.available_devices)
        if num_devices < self.world_size:
            logger.warning(f"Requested {self.world_size} processes but only {num_devices} GPUs available")
            self.world_size = max(1, num_devices)
        
        return self.world_size > 1
    
    def train_distributed(self, training_fn: Callable, *args, **kwargs) -> Any:
        """
        Run distributed training.
        
        Args:
            training_fn (Callable): Training function to run distributed.
            *args: Arguments to pass to the training function.
            **kwargs: Keyword arguments to pass to the training function.
        
        Returns:
            Any: Results from the training function.
        """
        if not self.can_distribute:
            logger.info("Running in single-process mode")
            return training_fn(*args, **kwargs)
        
        # Set up distributed training based on framework
        if self.framework == "pytorch" and TORCH_AVAILABLE:
            return self._train_distributed_pytorch(training_fn, *args, **kwargs)
        elif self.framework == "tensorflow" and TF_AVAILABLE:
            return self._train_distributed_tensorflow(training_fn, *args, **kwargs)
        else:
            logger.warning(f"Unsupported framework for distributed training: {self.framework}")
            return training_fn(*args, **kwargs)
    
    def _train_distributed_pytorch(self, training_fn: Callable, *args, **kwargs) -> Any:
        """
        Run distributed training with PyTorch.
        
        Args:
            training_fn (Callable): Training function to run distributed.
            *args: Arguments to pass to the training function.
            **kwargs: Keyword arguments to pass to the training function.
        
        Returns:
            Any: Results from the training function.
        """
        # Initialize distributed process group
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        
        # Prepare arguments for each process
        wrapped_fn = partial(self._pytorch_worker_fn, training_fn, *args, **kwargs)
        
        # Start processes
        if self.world_size > 1:
            mp.spawn(wrapped_fn, nprocs=self.world_size, join=True)
            results = {"distributed": True, "world_size": self.world_size}
        else:
            results = wrapped_fn(0)
        
        return results
    
    def _pytorch_worker_fn(self, training_fn: Callable, rank: int, *args, **kwargs) -> Any:
        """
        Worker function for PyTorch distributed training.
        
        Args:
            training_fn (Callable): Training function.
            rank (int): Process rank.
            *args: Arguments to pass to the training function.
            **kwargs: Keyword arguments to pass to the training function.
        
        Returns:
            Any: Results from the training function.
        """
        # Initialize distributed process group
        dist.init_process_group(
            backend="nccl" if self.gpu_manager.use_gpu else "gloo",
            init_method=f"env://",
            world_size=self.world_size,
            rank=rank
        )
        
        # Set device for this process
        if self.gpu_manager.use_gpu:
            torch.cuda.set_device(rank)
        
        # Update kwargs with process information
        kwargs["rank"] = rank
        kwargs["world_size"] = self.world_size
        kwargs["device"] = torch.device(f"cuda:{rank}" if self.gpu_manager.use_gpu else "cpu")
        
        # Run the training function
        results = training_fn(*args, **kwargs)
        
        # Clean up
        dist.destroy_process_group()
        
        return results
    
    def _train_distributed_tensorflow(self, training_fn: Callable, *args, **kwargs) -> Any:
        """
        Run distributed training with TensorFlow.
        
        Args:
            training_fn (Callable): Training function to run distributed.
            *args: Arguments to pass to the training function.
            **kwargs: Keyword arguments to pass to the training function.
        
        Returns:
            Any: Results from the training function.
        """
        # Set up TensorFlow distribution strategy
        if self.gpu_manager.use_gpu and len(self.gpu_manager.available_devices) > 0:
            # Multi-GPU strategy
            strategy = tf.distribute.MirroredStrategy()
        else:
            # CPU strategy
            strategy = tf.distribute.get_strategy()
        
        # Add strategy to kwargs
        kwargs["strategy"] = strategy
        kwargs["world_size"] = self.world_size
        
        # Run the training function within the strategy scope
        with strategy.scope():
            results = training_fn(*args, **kwargs)
        
        return results
    
    def gather_results(self, local_results: Any, rank: int, world_size: int) -> List[Any]:
        """
        Gather results from all processes.
        
        Args:
            local_results (Any): Local results from this process.
            rank (int): Process rank.
            world_size (int): Total number of processes.
        
        Returns:
            List[Any]: List of results from all processes.
        """
        if not self.can_distribute or world_size <= 1:
            return [local_results]
        
        # Gather results based on framework
        if self.framework == "pytorch" and TORCH_AVAILABLE:
            return self._gather_results_pytorch(local_results, rank, world_size)
        elif self.framework == "tensorflow" and TF_AVAILABLE:
            return self._gather_results_tensorflow(local_results, rank, world_size)
        else:
            logger.warning(f"Unsupported framework for result gathering: {self.framework}")
            return [local_results]
    
    def _gather_results_pytorch(self, local_results: Any, rank: int, world_size: int) -> List[Any]:
        """
        Gather results from all processes using PyTorch.
        
        Args:
            local_results (Any): Local results from this process.
            rank (int): Process rank.
            world_size (int): Total number of processes.
        
        Returns:
            List[Any]: List of results from all processes.
        """
        # Convert results to JSON string for transmission
        local_json = json.dumps(local_results)
        local_tensor = torch.tensor(bytearray(local_json.encode()), dtype=torch.uint8)
        
        # Get size of each result tensor
        size_tensor = torch.tensor([local_tensor.numel()], dtype=torch.long)
        size_list = [torch.tensor([0], dtype=torch.long) for _ in range(world_size)]
        
        # Gather sizes
        dist.all_gather(size_list, size_tensor)
        
        # Create tensors for gathering results
        max_size = max(size.item() for size in size_list)
        if local_tensor.numel() < max_size:
            padding = torch.zeros(max_size - local_tensor.numel(), dtype=torch.uint8)
            local_tensor = torch.cat((local_tensor, padding))
        
        tensor_list = [torch.zeros(max_size, dtype=torch.uint8) for _ in range(world_size)]
        
        # Gather results
        dist.all_gather(tensor_list, local_tensor)
        
        # Decode results
        results = []
        for tensor, size in zip(tensor_list, size_list):
            bytes_data = bytes(tensor[:size.item()].tolist())
            json_str = bytes_data.decode()
            result = json.loads(json_str)
            results.append(result)
        
        return results
    
    def _gather_results_tensorflow(self, local_results: Any, rank: int, world_size: int) -> List[Any]:
        """
        Gather results from all processes using TensorFlow.
        
        Args:
            local_results (Any): Local results from this process.
            rank (int): Process rank.
            world_size (int): Total number of processes.
        
        Returns:
            List[Any]: List of results from all processes.
        """
        # TensorFlow's distribution strategy handles this differently
        # For simplicity, we just return the local results
        # In a real implementation, this would use TensorFlow's mechanisms
        return [local_results]
    
    def reduce_results(self, all_results: List[Any]) -> Any:
        """
        Reduce results from all processes to a single result.
        
        Args:
            all_results (List[Any]): Results from all processes.
        
        Returns:
            Any: Combined result.
        """
        if not all_results:
            return None
        
        # This is a simplified implementation
        # In a real-world scenario, this would depend on the specific format of results
        
        # If results are dictionaries, combine them
        if all(isinstance(r, dict) for r in all_results):
            combined = {}
            for result in all_results:
                for key, value in result.items():
                    if key in combined:
                        # If value is numeric, sum it
                        if isinstance(value, (int, float)) and isinstance(combined[key], (int, float)):
                            combined[key] += value
                        # If value is a list, extend it
                        elif isinstance(value, list) and isinstance(combined[key], list):
                            combined[key].extend(value)
                        # Otherwise, keep the value from the first result
                    else:
                        combined[key] = value
            return combined
        
        # If results are lists, concatenate them
        elif all(isinstance(r, list) for r in all_results):
            combined = []
            for result in all_results:
                combined.extend(result)
            return combined
        
        # If results are numeric, sum them
        elif all(isinstance(r, (int, float)) for r in all_results):
            return sum(all_results)
        
        # Otherwise, return the results as a list
        return all_results


# Utility functions
def find_free_port() -> int:
    """
    Find a free port for distributed training.
    
    Returns:
        int: Free port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# Singleton instance
_instance = None

def get_distributed_trainer() -> DistributedTrainer:
    """
    Get the singleton distributed trainer instance.
    
    Returns:
        DistributedTrainer: Distributed trainer instance.
    """
    global _instance
    if _instance is None:
        _instance = DistributedTrainer()
    return _instance