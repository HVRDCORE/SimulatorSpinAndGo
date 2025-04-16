"""
GPU optimization utilities for the poker simulator.

This module provides utilities for optimizing poker simulations with GPU acceleration,
including CUDA support through PyTorch and TensorFlow, as well as integration with
Numba for accelerated CPU computation when GPU is not available.
"""

import os
import time
import logging
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import numba
    from numba import cuda
    HAS_NUMBA = True
    HAS_CUDA = cuda.is_available()
except ImportError:
    HAS_NUMBA = False
    HAS_CUDA = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pokersim.utils.gpu_optimization")

def optimize_tensor_for_gpu(tensor: Any) -> Any:
    """
    Optimize a tensor for GPU execution if available.

    Args:
        tensor (Any): Input tensor (PyTorch or TensorFlow tensor).

    Returns:
        Any: Tensor optimized for GPU if possible, else unchanged.
    """
    if HAS_TORCH and isinstance(tensor, torch.Tensor):
        if torch.cuda.is_available():
            return tensor.to('cuda')
        return tensor
    elif HAS_TF and isinstance(tensor, tf.Tensor):
        return tensor
    else:
        logger.warning(f"Cannot optimize tensor of type {type(tensor)}")
        return tensor

class GPUManager:
    def __init__(self, use_gpu: bool = True, framework: Optional[str] = None):
        self.use_gpu = use_gpu and (HAS_TORCH or HAS_TF)
        self.framework = framework.lower() if framework else "none"
        self.torch_gpu_available = False
        self.tf_gpu_available = False
        self.device = "cpu"

        if not self.use_gpu:
            logger.info("GPU disabled by configuration")
            return

        if self.framework == "none":
            if HAS_TORCH:
                self.framework = "pytorch"
            elif HAS_TF:
                self.framework = "tensorflow"

        if self.framework == "pytorch" and HAS_TORCH:
            self.torch_gpu_available = torch.cuda.is_available()
            if self.torch_gpu_available:
                self.device = torch.device("cuda")
                logger.info(f"Using PyTorch with GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                logger.info("Using PyTorch with CPU")
        elif self.framework == "tensorflow" and HAS_TF:
            gpus = tf.config.list_physical_devices('GPU')
            self.tf_gpu_available = len(gpus) > 0
            if self.tf_gpu_available:
                self.device = "/GPU:0"
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Using TensorFlow with GPU: {gpus}")
                except RuntimeError as e:
                    logger.error(f"GPU memory growth configuration error: {e}")
            else:
                self.device = "/CPU:0"
                logger.info("Using TensorFlow with CPU")
        else:
            logger.warning("No supported framework selected or available")

    def to_device(self, data: Any) -> Any:
        if not self.use_gpu:
            return data
        if self.framework == "pytorch" and self.torch_gpu_available:
            if isinstance(data, torch.Tensor):
                return data.to(self.device)
            elif isinstance(data, np.ndarray):
                return torch.tensor(data, device=self.device)
            else:
                logger.warning(f"Cannot move type {type(data)} to PyTorch device")
                return data
        elif self.framework == "tensorflow" and self.tf_gpu_available:
            if isinstance(data, tf.Tensor):
                return data
            elif isinstance(data, np.ndarray):
                return tf.convert_to_tensor(data)
            else:
                logger.warning(f"Cannot move type {type(data)} to TensorFlow device")
                return data
        return data

    def from_device(self, data: Any) -> np.ndarray:
        if self.framework == "pytorch" and isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif self.framework == "tensorflow" and isinstance(data, tf.Tensor):
            return data.numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            logger.warning(f"Cannot convert type {type(data)} to numpy")
            return np.array(data)

    def memory_stats(self) -> Dict[str, float]:
        stats = {}
        if self.framework == "pytorch" and self.torch_gpu_available:
            stats["allocated"] = torch.cuda.memory_allocated() / 1024**2
            stats["max_allocated"] = torch.cuda.max_memory_allocated() / 1024**2
        return stats

    def synchronize(self) -> None:
        if self.framework == "pytorch" and self.torch_gpu_available:
            torch.cuda.synchronize()
        elif self.framework == "tensorflow" and self.tf_gpu_available:
            tf.config.experimental.synchronize()

    def parallel_map(self, fn: Callable, data: List[Any], batch_size: int = 1000) -> List[Any]:
        results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_results = fn(batch)
            results.extend(batch_results)
        return results

    def get_supported_tensor_type(self) -> str:
        if self.framework == "pytorch" and HAS_TORCH:
            return "torch.Tensor"
        elif self.framework == "tensorflow" and HAS_TF:
            return "tf.Tensor"
        return "numpy.ndarray"

    def get_framework_info(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "device": str(self.device),
            "torch_available": HAS_TORCH,
            "tensorflow_available": HAS_TF,
            "gpu_available": self.torch_gpu_available or self.tf_gpu_available,
        }

def get_gpu_manager(use_gpu: bool = True, framework: Optional[str] = None) -> GPUManager:
    return GPUManager(use_gpu, framework)

def execute_on_gpu(func: Callable) -> Callable:
    def wrapper(*args, **kwargs) -> Any:
        manager = get_gpu_manager()
        args = [manager.to_device(arg) if isinstance(arg, (torch.Tensor, tf.Tensor, np.ndarray)) else arg for arg in args]
        kwargs = {k: manager.to_device(v) if isinstance(v, (torch.Tensor, tf.Tensor, np.ndarray)) else v for k, v in kwargs.items()}
        result = func(*args, **kwargs)
        return manager.from_device(result) if isinstance(result, (torch.Tensor, tf.Tensor)) else result
    return wrapper

def time_execution(func: Callable) -> Callable:
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"Execution of {func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper

def optimize_with_numba(func: Callable) -> Callable:
    if HAS_NUMBA:
        return numba.jit(nopython=True)(func)
    return func

def optimize_with_cuda(func: Callable) -> Callable:
    if HAS_NUMBA and HAS_CUDA:
        return cuda.jit(func)
    return func