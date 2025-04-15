"""
GPU optimization utilities for the poker simulator.

This module provides utilities for optimizing poker simulations with GPU acceleration,
including CUDA support through PyTorch and TensorFlow, as well as integration with
Numba for accelerated CPU computation when GPU is not available.
"""

import os
import time
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np

# PyTorch-based GPU utilities
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# TensorFlow-based GPU utilities
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

# Numba-based optimization utilities
try:
    import numba
    from numba import cuda
    HAS_NUMBA = True
    HAS_CUDA = numba.cuda.is_available()
except ImportError:
    HAS_NUMBA = False
    HAS_CUDA = False


class GPUManager:
    """
    Manager for GPU acceleration in poker simulations.
    
    This class provides a unified interface for GPU acceleration using either
    PyTorch or TensorFlow, with fallback to CPU computation using Numba when
    GPU is not available.
    
    Attributes:
        use_gpu (bool): Whether GPU acceleration is enabled.
        framework (str): Deep learning framework being used ('pytorch', 'tensorflow', or 'none').
        device: Device to run computations (specific to framework).
    """
    
    def __init__(self, use_gpu: bool = True, framework: str = "auto"):
        """
        Initialize the GPU manager.
        
        Args:
            use_gpu (bool, optional): Whether to use GPU acceleration if available. Defaults to True.
            framework (str, optional): Deep learning framework to use. Defaults to "auto".
        """
        self.use_gpu = use_gpu
        
        # Check GPU availability for each framework
        self.torch_gpu_available = HAS_TORCH and torch.cuda.is_available() if use_gpu else False
        self.tf_gpu_available = HAS_TF and len(tf.config.list_physical_devices('GPU')) > 0 if use_gpu else False
        self.numba_gpu_available = HAS_NUMBA and HAS_CUDA if use_gpu else False
        
        # Select framework
        if framework == "auto":
            if self.torch_gpu_available:
                self.framework = "pytorch"
            elif self.tf_gpu_available:
                self.framework = "tensorflow"
            elif HAS_TORCH:
                self.framework = "pytorch"
                self.use_gpu = False
            elif HAS_TF:
                self.framework = "tensorflow"
                self.use_gpu = False
            else:
                self.framework = "none"
                self.use_gpu = False
        else:
            self.framework = framework
            
            # Check if the requested framework is available
            if framework == "pytorch" and not HAS_TORCH:
                logging.warning("PyTorch requested but not available, falling back to CPU")
                self.use_gpu = False
            elif framework == "tensorflow" and not HAS_TF:
                logging.warning("TensorFlow requested but not available, falling back to CPU")
                self.use_gpu = False
        
        # Initialize device based on framework
        if self.framework == "pytorch":
            self.device = torch.device("cuda" if self.torch_gpu_available else "cpu")
        elif self.framework == "tensorflow":
            if self.tf_gpu_available:
                self.device = "/GPU:0"
                # Configure TensorFlow to use memory growth
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    try:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                    except RuntimeError as e:
                        logging.error(f"GPU memory growth configuration error: {e}")
            else:
                self.device = "/CPU:0"
        else:
            self.device = "cpu"
        
        # Log initialization
        self._log_gpu_info()
    
    def _log_gpu_info(self) -> None:
        """Log information about the GPU configuration."""
        if self.use_gpu:
            if self.framework == "pytorch" and self.torch_gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
                logging.info(f"Using PyTorch with GPU acceleration: {gpu_name} (Count: {gpu_count})")
            elif self.framework == "tensorflow" and self.tf_gpu_available:
                gpus = tf.config.list_physical_devices('GPU')
                logging.info(f"Using TensorFlow with GPU acceleration: {len(gpus)} GPUs")
            elif self.numba_gpu_available:
                logging.info(f"Using Numba with CUDA acceleration")
            else:
                logging.info(f"No GPU acceleration available")
        else:
            logging.info(f"GPU acceleration disabled, using {self.framework} on CPU")
    
    def to_device(self, data: Any) -> Any:
        """
        Move data to the appropriate device (GPU or CPU).
        
        Args:
            data: Data to move (tensor, array, or similar).
            
        Returns:
            Data on the target device.
        """
        if not self.use_gpu:
            return data
        
        if self.framework == "pytorch" and self.torch_gpu_available:
            if isinstance(data, torch.Tensor):
                return data.to(self.device)
            elif isinstance(data, np.ndarray):
                return torch.tensor(data, device=self.device)
            else:
                return data  # Can't move to device
        
        elif self.framework == "tensorflow" and self.tf_gpu_available:
            if isinstance(data, tf.Tensor):
                with tf.device(self.device):
                    return tf.identity(data)
            elif isinstance(data, np.ndarray):
                with tf.device(self.device):
                    return tf.convert_to_tensor(data)
            else:
                return data  # Can't move to device
        
        return data  # Default case
    
    def from_device(self, data: Any) -> Any:
        """
        Move data from device to CPU (for numpy operations).
        
        Args:
            data: Data to move (tensor, array, or similar).
            
        Returns:
            Data on CPU as numpy array if possible.
        """
        if self.framework == "pytorch" and isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        
        elif self.framework == "tensorflow" and isinstance(data, tf.Tensor):
            return data.numpy()
        
        elif isinstance(data, np.ndarray):
            return data
        
        return data  # Default case
    
    def memory_stats(self) -> Dict[str, float]:
        """
        Get memory statistics for the GPU.
        
        Returns:
            Dict[str, float]: Memory statistics.
        """
        stats = {}
        
        if not self.use_gpu:
            return stats
        
        if self.framework == "pytorch" and self.torch_gpu_available:
            stats['memory_allocated'] = torch.cuda.memory_allocated(0)
            stats['memory_reserved'] = torch.cuda.memory_reserved(0)
            stats['memory_total'] = torch.cuda.get_device_properties(0).total_memory
            stats['utilization'] = stats['memory_allocated'] / stats['memory_total']
        
        elif self.framework == "tensorflow" and self.tf_gpu_available:
            # TensorFlow doesn't provide direct GPU memory statistics
            # We can only report device availability
            stats['gpu_available'] = True
        
        return stats
    
    def synchronize(self) -> None:
        """Synchronize the device (wait for pending operations to complete)."""
        if not self.use_gpu:
            return
        
        if self.framework == "pytorch" and self.torch_gpu_available:
            torch.cuda.synchronize()
    
    def parallel_map(self, func: Any, data_list: List[Any], 
                   batch_size: int = 32) -> List[Any]:
        """
        Apply a function to a list of data in parallel using GPU acceleration.
        
        Args:
            func: Function to apply.
            data_list (List[Any]): List of data items.
            batch_size (int, optional): Batch size for processing. Defaults to 32.
            
        Returns:
            List[Any]: Results from applying the function to each data item.
        """
        results = []
        
        if not self.use_gpu:
            # CPU fallback - still batch for efficiency
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                batch_results = [func(item) for item in batch]
                results.extend(batch_results)
            return results
        
        if self.framework == "pytorch" and self.torch_gpu_available:
            # PyTorch GPU implementation
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                
                # Convert batch to tensors if they aren't already
                batch_tensors = []
                for item in batch:
                    if isinstance(item, torch.Tensor):
                        batch_tensors.append(item.to(self.device))
                    elif isinstance(item, np.ndarray):
                        batch_tensors.append(torch.tensor(item, device=self.device))
                    else:
                        batch_tensors.append(item)  # Can't convert
                
                # Apply function to each item
                batch_results = [func(item) for item in batch_tensors]
                
                # Convert results back to CPU if they're tensors
                processed_results = []
                for res in batch_results:
                    if isinstance(res, torch.Tensor):
                        processed_results.append(res.detach().cpu().numpy())
                    else:
                        processed_results.append(res)
                
                results.extend(processed_results)
        
        elif self.framework == "tensorflow" and self.tf_gpu_available:
            # TensorFlow GPU implementation
            with tf.device(self.device):
                for i in range(0, len(data_list), batch_size):
                    batch = data_list[i:i + batch_size]
                    
                    # Convert batch to tensors if they aren't already
                    batch_tensors = []
                    for item in batch:
                        if isinstance(item, tf.Tensor):
                            batch_tensors.append(item)
                        elif isinstance(item, np.ndarray):
                            batch_tensors.append(tf.convert_to_tensor(item))
                        else:
                            batch_tensors.append(item)  # Can't convert
                    
                    # Apply function to each item
                    batch_results = [func(item) for item in batch_tensors]
                    
                    # Convert results back to CPU if they're tensors
                    processed_results = []
                    for res in batch_results:
                        if isinstance(res, tf.Tensor):
                            processed_results.append(res.numpy())
                        else:
                            processed_results.append(res)
                    
                    results.extend(processed_results)
        
        return results
    
    def get_supported_tensor_type(self):
        """
        Get the appropriate tensor type for the selected framework.
        
        Returns:
            The tensor class for the framework.
        """
        if self.framework == "pytorch":
            return torch.Tensor
        elif self.framework == "tensorflow":
            return tf.Tensor
        else:
            return np.ndarray
    
    def get_framework_info(self) -> Dict[str, Any]:
        """
        Get information about the framework and device.
        
        Returns:
            Dict[str, Any]: Framework and device information.
        """
        info = {
            'framework': self.framework,
            'use_gpu': self.use_gpu,
            'device': str(self.device)
        }
        
        if self.framework == "pytorch":
            info['torch_version'] = torch.__version__
            if self.torch_gpu_available:
                info['gpu_name'] = torch.cuda.get_device_name(0)
                info['gpu_count'] = torch.cuda.device_count()
                info['cuda_version'] = torch.version.cuda
        
        elif self.framework == "tensorflow":
            info['tensorflow_version'] = tf.__version__
            if self.tf_gpu_available:
                gpus = tf.config.list_physical_devices('GPU')
                info['gpu_count'] = len(gpus)
        
        return info


# Global instance for usage throughout the package
_GPU_MANAGER = None


def get_gpu_manager() -> GPUManager:
    """
    Get the global GPU manager instance.
    
    Returns:
        GPUManager: The global GPU manager instance.
    """
    global _GPU_MANAGER
    if _GPU_MANAGER is None:
        _GPU_MANAGER = GPUManager()
    return _GPU_MANAGER


def execute_on_gpu(func):
    """
    Decorator to execute a function on GPU if available.
    
    Args:
        func: Function to be executed on GPU.
        
    Returns:
        Wrapped function that will execute on GPU if available.
    """
    gpu_manager = get_gpu_manager()
    
    def wrapper(*args, **kwargs):
        # Move tensor arguments to GPU if possible
        gpu_args = []
        for arg in args:
            gpu_args.append(gpu_manager.to_device(arg))
        
        gpu_kwargs = {}
        for key, value in kwargs.items():
            gpu_kwargs[key] = gpu_manager.to_device(value)
        
        # Execute function
        result = func(*gpu_args, **gpu_kwargs)
        
        # Move result back to CPU
        return gpu_manager.from_device(result)
    
    return wrapper


def time_execution(func):
    """
    Decorator to time the execution of a function.
    
    Args:
        func: Function to be timed.
        
    Returns:
        Wrapped function that will be timed.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.debug(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    
    return wrapper


# Numba optimizations if available
if HAS_NUMBA:
    def optimize_with_numba(func):
        """
        Decorator to optimize a function with Numba.
        
        Args:
            func: Function to be optimized.
            
        Returns:
            Numba-optimized function.
        """
        return numba.jit(nopython=True)(func)
    
    def optimize_with_cuda(func):
        """
        Decorator to optimize a function with CUDA via Numba.
        
        Args:
            func: Function to be optimized.
            
        Returns:
            CUDA-optimized function if CUDA is available, otherwise the original function.
        """
        gpu_manager = get_gpu_manager()
        
        if HAS_CUDA and gpu_manager.use_gpu:
            return cuda.jit(func)
        else:
            return func
else:
    # Dummy decorators if Numba is not available
    def optimize_with_numba(func):
        """Dummy decorator when Numba is not available."""
        return func
    
    def optimize_with_cuda(func):
        """Dummy decorator when CUDA is not available."""
        return func