"""
GPU optimization utilities for the poker simulator using PyTorch.

This module provides utilities for optimizing poker simulations with GPU acceleration
using PyTorch. It includes functionality for managing GPU resources, optimizing tensors,
and executing functions on the GPU.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Callable
import numpy as np
import torch

# Configure logging to output to both console and file for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log', mode='w')
    ]
)
logger = logging.getLogger("pokersim.utils.gpu_optimization")

def optimize_tensor_for_gpu(tensor: Any) -> Any:
    """
    Optimize a tensor for GPU execution if available.

    Args:
        tensor (Any): Input tensor (PyTorch tensor or numpy array).

    Returns:
        Any: Tensor optimized for GPU if possible, else unchanged.
    """
    if isinstance(tensor, torch.Tensor):
        if torch.cuda.is_available():
            logger.debug(f"Moving tensor to GPU: {tensor.shape}")
            return tensor.to('cuda')
        return tensor
    elif isinstance(tensor, np.ndarray):
        if torch.cuda.is_available():
            logger.debug(f"Converting numpy array to GPU tensor: {tensor.shape}")
            return torch.from_numpy(tensor).to('cuda')
        return torch.from_numpy(tensor)
    else:
        logger.warning(f"Cannot optimize tensor of type {type(tensor)}")
        return tensor

class GPUManager:
    def __init__(self, use_gpu: bool = True):
        """
        Initialize GPUManager for PyTorch GPU management.

        Args:
            use_gpu (bool): Whether to attempt using GPU if available.
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.gpu_available = self.use_gpu

        if self.use_gpu:
            try:
                logger.info(f"Using PyTorch with GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            except Exception as e:
                logger.error(f"Failed to initialize GPU: {e}")
                self.use_gpu = False
                self.device = torch.device("cpu")
                self.gpu_available = False
        else:
            logger.info("Using PyTorch with CPU")

    def to_device(self, data: Any) -> Any:
        """
        Move data to the configured device (GPU or CPU).

        Args:
            data (Any): Input data (PyTorch tensor or numpy array).

        Returns:
            Any: Data moved to the appropriate device.
        """
        if not self.use_gpu:
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data)
            return data

        if isinstance(data, torch.Tensor):
            logger.debug(f"Moving tensor to {self.device}")
            return data.to(self.device)
        elif isinstance(data, np.ndarray):
            logger.debug(f"Converting numpy array to {self.device} tensor")
            return torch.from_numpy(data).to(self.device)
        else:
            logger.warning(f"Cannot move type {type(data)} to {self.device}")
            return data

    def from_device(self, data: Any) -> np.ndarray:
        """
        Convert data from device to numpy array.

        Args:
            data (Any): Input data (PyTorch tensor or numpy array).

        Returns:
            np.ndarray: Data as a numpy array.
        """
        if isinstance(data, torch.Tensor):
            logger.debug("Converting tensor to numpy")
            return data.cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            logger.warning(f"Cannot convert type {type(data)} to numpy")
            return np.array(data)

    def memory_stats(self) -> Dict[str, float]:
        """
        Get GPU memory statistics.

        Returns:
            Dict[str, float]: Dictionary with memory stats in MB.
        """
        stats = {}
        if self.gpu_available:
            stats["allocated"] = torch.cuda.memory_allocated() / 1024**2
            stats["max_allocated"] = torch.cuda.max_memory_allocated() / 1024**2
            stats["reserved"] = torch.cuda.memory_reserved() / 1024**2
        return stats

    def synchronize(self) -> None:
        """
        Synchronize GPU operations.
        """
        if self.gpu_available:
            torch.cuda.synchronize()

    def parallel_map(self, fn: Callable, data: List[Any], batch_size: int = 100) -> List[Any]:
        """
        Apply a function to data in parallel batches.

        Args:
            fn (Callable): Function to apply to each batch.
            data (List[Any]): List of data to process.
            batch_size (int): Size of each batch.

        Returns:
            List[Any]: Processed results.
        """
        results = []
        total_batches = (len(data) + batch_size - 1) // batch_size
        for i in range(0, len(data), batch_size):
            batch_idx = i // batch_size + 1
            logger.info(f"Processing batch {batch_idx}/{total_batches}")
            batch = data[i:i + batch_size]
            batch_results = fn(self.to_device(batch))
            results.extend(self.from_device(batch_results) if isinstance(batch_results, torch.Tensor) else batch_results)
        return results

    def get_supported_tensor_type(self) -> str:
        """
        Get the supported tensor type.

        Returns:
            str: Tensor type.
        """
        return "torch.Tensor"

    def get_framework_info(self) -> Dict[str, Any]:
        """
        Get information about the framework and device.

        Returns:
            Dict[str, Any]: Framework and device information.
        """
        return {
            "framework": "pytorch",
            "device": str(self.device),
            "gpu_available": self.gpu_available,
        }

# Global GPUManager instance to avoid multiple initializations
_global_gpu_manager = None

def get_gpu_manager(use_gpu: bool = True) -> GPUManager:
    """
    Get the global GPUManager instance.

    Args:
        use_gpu (bool): Whether to attempt using GPU if available.

    Returns:
        GPUManager: Singleton GPUManager instance.
    """
    global _global_gpu_manager
    if _global_gpu_manager is None:
        _global_gpu_manager = GPUManager(use_gpu)
    return _global_gpu_manager

def execute_on_gpu(func: Callable) -> Callable:
    """
    Decorator to execute a function with GPU-optimized arguments.

    Args:
        func (Callable): Function to decorate.

    Returns:
        Callable: Wrapped function with GPU optimization.
    """
    def wrapper(*args, **kwargs) -> Any:
        manager = get_gpu_manager()
        args = [manager.to_device(arg) if isinstance(arg, (torch.Tensor, np.ndarray)) else arg for arg in args]
        kwargs = {k: manager.to_device(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v for k, v in kwargs.items()}
        result = func(*args, **kwargs)
        return manager.from_device(result) if isinstance(result, torch.Tensor) else result
    return wrapper

def time_execution(func: Callable) -> Callable:
    """
    Decorator to measure execution time of a function.

    Args:
        func (Callable): Function to decorate.

    Returns:
        Callable: Wrapped function with timing.
    """
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"Execution of {func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper