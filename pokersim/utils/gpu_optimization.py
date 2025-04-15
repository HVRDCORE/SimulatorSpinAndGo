"""
GPU Optimization utilities for poker simulation and training.

This module provides utilities for optimizing performance on GPUs,
including data transfer, memory management, and parallel processing 
for faster poker hand evaluation and neural network training.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_available_devices() -> List[torch.device]:
    """
    Get a list of available devices (CPU and GPUs).
    
    Returns:
        List[torch.device]: List of available devices.
    """
    devices = [torch.device("cpu")]
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} CUDA-capable devices")
        
        for i in range(num_gpus):
            device = torch.device(f"cuda:{i}")
            devices.append(device)
            
            # Log device properties
            properties = torch.cuda.get_device_properties(i)
            logger.info(f"Device {i}: {properties.name}")
            logger.info(f"  Memory: {properties.total_memory / 1e9:.2f} GB")
            logger.info(f"  CUDA Capability: {properties.major}.{properties.minor}")
    else:
        logger.info("No CUDA-capable devices found, using CPU only")
    
    return devices


def select_best_device() -> torch.device:
    """
    Select the best available device for training.
    
    Returns:
        torch.device: The best available device.
    """
    if not torch.cuda.is_available():
        logger.info("CUDA not available, using CPU")
        return torch.device("cpu")
    
    # Find GPU with the most free memory
    best_device = 0
    max_free_memory = 0
    
    for i in range(torch.cuda.device_count()):
        # Get memory info
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        
        # Get current memory usage
        reserved = torch.cuda.memory_reserved(i)
        allocated = torch.cuda.memory_allocated(i)
        free = reserved - allocated
        
        logger.info(f"GPU {i}: {free / 1e9:.2f} GB free")
        
        if free > max_free_memory:
            max_free_memory = free
            best_device = i
    
    logger.info(f"Selected GPU {best_device} as best device")
    return torch.device(f"cuda:{best_device}")


def optimize_model_for_device(model: nn.Module, device: torch.device) -> nn.Module:
    """
    Optimize a model for a specific device.
    
    Args:
        model (nn.Module): The PyTorch model to optimize.
        device (torch.device): Target device.
    
    Returns:
        nn.Module: Optimized model.
    """
    # Move model to device
    model = model.to(device)
    
    # Apply optimizations based on device
    if device.type == "cuda":
        # Enable cuDNN autotuner if available
        torch.backends.cudnn.benchmark = True
        
        # Try to convert model to TorchScript for better performance
        try:
            # Create a sample input for tracing
            sample_shape = next(model.parameters()).shape[1]
            sample_input = torch.zeros(1, sample_shape, device=device)
            
            # Trace the model
            traced_model = torch.jit.trace(model, sample_input)
            logger.info("Successfully converted model to TorchScript")
            return traced_model
        except Exception as e:
            logger.warning(f"Could not convert model to TorchScript: {e}")
            logger.warning("Using original model instead")
    
    return model


def optimize_tensor_for_gpu(tensor: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Optimize a numpy array for GPU processing.
    
    Args:
        tensor (np.ndarray): Numpy array.
        device (torch.device): Target device.
    
    Returns:
        torch.Tensor: Tensor optimized for the device.
    """
    # Convert numpy array to torch tensor
    tensor = torch.from_numpy(tensor)
    
    # Move to device
    tensor = tensor.to(device)
    
    # Optimize memory layout for the device
    if device.type == "cuda":
        tensor = tensor.contiguous()
    
    return tensor


def batch_process(func, items: List[Any], batch_size: int = 512, 
                device: Optional[torch.device] = None) -> List[Any]:
    """
    Process a list of items in batches for GPU efficiency.
    
    Args:
        func: Function to apply to each batch.
        items (List[Any]): Items to process.
        batch_size (int, optional): Batch size. Defaults to 512.
        device (Optional[torch.device], optional): Device to use. Defaults to None.
    
    Returns:
        List[Any]: Processed items.
    """
    if device is None:
        device = select_best_device()
    
    results = []
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Process batch
        batch_results = func(batch, device)
        results.extend(batch_results)
    
    return results


class TensorMemoryPool:
    """
    Memory pool for reusing GPU tensors.
    
    This class helps avoid frequent GPU memory allocations and deallocations
    by reusing tensor memory.
    
    Attributes:
        pools (Dict): Dictionary of tensor pools, keyed by shape and dtype.
    """
    
    def __init__(self):
        """Initialize a tensor memory pool."""
        self.pools = {}
    
    def get(self, shape: Tuple[int, ...], dtype: torch.dtype, 
          device: torch.device) -> torch.Tensor:
        """
        Get a tensor of the specified shape, dtype, and device.
        
        Args:
            shape (Tuple[int, ...]): Shape of the tensor.
            dtype (torch.dtype): Data type of the tensor.
            device (torch.device): Device for the tensor.
        
        Returns:
            torch.Tensor: A tensor from the pool or a new tensor.
        """
        key = (shape, dtype, str(device))
        
        if key in self.pools and self.pools[key]:
            # Reuse a tensor from the pool
            tensor = self.pools[key].pop()
            tensor.zero_()
            return tensor
        else:
            # Create a new tensor
            return torch.zeros(shape, dtype=dtype, device=device)
    
    def put(self, tensor: torch.Tensor):
        """
        Return a tensor to the pool.
        
        Args:
            tensor (torch.Tensor): Tensor to return to the pool.
        """
        key = (tensor.shape, tensor.dtype, str(tensor.device))
        
        if key not in self.pools:
            self.pools[key] = []
        
        # Add tensor to the pool
        self.pools[key].append(tensor)
    
    def clear(self):
        """Clear all pools."""
        self.pools = {}


# Create a global memory pool
memory_pool = TensorMemoryPool()


class GPUBatchProcessor:
    """
    Batch processor for GPU-optimized poker hand evaluations.
    
    This class manages efficient batch processing of poker hand evaluations
    on the GPU, including data transfer and memory management.
    
    Attributes:
        device (torch.device): Device to use for processing.
        batch_size (int): Size of batches for processing.
    """
    
    def __init__(self, device: Optional[torch.device] = None, batch_size: int = 1024):
        """
        Initialize a GPU batch processor.
        
        Args:
            device (Optional[torch.device], optional): Device to use. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 1024.
        """
        self.device = device if device is not None else select_best_device()
        self.batch_size = batch_size
        
        logger.info(f"Initialized GPU batch processor on {self.device}")
        logger.info(f"Batch size: {self.batch_size}")
    
    def process_hands(self, hole_cards: List[Tuple[int, int]], 
                    community_cards: List[int]) -> List[int]:
        """
        Evaluate multiple poker hands in a GPU-optimized manner.
        
        Args:
            hole_cards (List[Tuple[int, int]]): List of hole card pairs.
            community_cards (List[int]): Community cards.
        
        Returns:
            List[int]: Hand strength scores.
        """
        # Prepare data for GPU
        num_hands = len(hole_cards)
        
        # Organize into batches
        results = []
        
        for i in range(0, num_hands, self.batch_size):
            batch_size = min(self.batch_size, num_hands - i)
            batch_hole_cards = hole_cards[i:i + batch_size]
            
            # Process batch
            batch_results = self._process_hand_batch(batch_hole_cards, community_cards)
            results.extend(batch_results)
        
        return results
    
    def _process_hand_batch(self, hole_cards: List[Tuple[int, int]], 
                          community_cards: List[int]) -> List[int]:
        """
        Process a batch of poker hands on the GPU.
        
        Args:
            hole_cards (List[Tuple[int, int]]): List of hole card pairs.
            community_cards (List[int]): Community cards.
        
        Returns:
            List[int]: Hand strength scores.
        """
        # This is a placeholder for GPU-optimized hand evaluation logic
        # In a real implementation, you would use a CUDA kernel or PyTorch operations
        # to evaluate hands in parallel on the GPU
        
        # For now, just return dummy scores
        return [i for i in range(len(hole_cards))]
    
    def simulate_equity(self, hole_cards: List[Tuple[int, int]], 
                      community_cards: List[int], num_players: int, 
                      num_simulations: int = 1000) -> List[float]:
        """
        Simulate equity for multiple hands using GPU acceleration.
        
        Args:
            hole_cards (List[Tuple[int, int]]): List of hole card pairs.
            community_cards (List[int]): Community cards.
            num_players (int): Number of players.
            num_simulations (int, optional): Number of simulations. Defaults to 1000.
        
        Returns:
            List[float]: Equity for each hand.
        """
        # Placeholder for GPU-accelerated equity simulation
        return [0.5 for _ in range(len(hole_cards))]


def batch_inference(model: nn.Module, inputs: List[np.ndarray], 
                  device: Optional[torch.device] = None, 
                  batch_size: int = 64) -> List[np.ndarray]:
    """
    Perform batch inference with a PyTorch model.
    
    Args:
        model (nn.Module): PyTorch model.
        inputs (List[np.ndarray]): List of input arrays.
        device (Optional[torch.device], optional): Device to use. Defaults to None.
        batch_size (int, optional): Batch size. Defaults to 64.
    
    Returns:
        List[np.ndarray]: Model outputs.
    """
    if device is None:
        device = select_best_device()
    
    model = model.to(device)
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            # Get batch
            batch_inputs = inputs[i:i + batch_size]
            
            # Convert to tensors
            batch_tensors = [torch.from_numpy(x).float().to(device) for x in batch_inputs]
            
            # Stack tensors
            if all(x.shape == batch_tensors[0].shape for x in batch_tensors):
                stacked_batch = torch.stack(batch_tensors)
            else:
                # If inputs have different shapes, process them individually
                batch_outputs = []
                for tensor in batch_tensors:
                    output = model(tensor.unsqueeze(0)).cpu().numpy()
                    batch_outputs.append(output.squeeze(0))
                results.extend(batch_outputs)
                continue
            
            # Process batch
            outputs = model(stacked_batch)
            
            # Convert back to numpy and add to results
            cpu_outputs = outputs.cpu().numpy()
            results.extend([output for output in cpu_outputs])
    
    return results


class GPUProfiler:
    """
    Profiler for GPU operations.
    
    This class provides utilities for profiling GPU operations
    to help optimize performance.
    
    Attributes:
        enabled (bool): Whether profiling is enabled.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize a GPU profiler.
        
        Args:
            enabled (bool, optional): Whether profiling is enabled. Defaults to True.
        """
        self.enabled = enabled
        
        # Check if PyTorch profiler is available
        self.has_profiler = hasattr(torch.profiler, 'profile')
        
        if self.enabled and not self.has_profiler:
            logger.warning("PyTorch profiler not available. Using basic profiling.")
    
    def profile(self, func, *args, **kwargs):
        """
        Profile a function.
        
        Args:
            func: Function to profile.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
        
        Returns:
            Any: Function result.
        """
        if not self.enabled:
            return func(*args, **kwargs)
        
        if self.has_profiler:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                result = func(*args, **kwargs)
            
            logger.info("GPU Profiling Results:")
            logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            
            return result
        else:
            # Basic profiling
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            # Record start event
            start.record()
            
            # Run function
            result = func(*args, **kwargs)
            
            # Record end event
            end.record()
            
            # Wait for events to complete
            torch.cuda.synchronize()
            
            # Calculate time
            elapsed_time = start.elapsed_time(end)
            logger.info(f"Function execution time: {elapsed_time:.2f} ms")
            
            return result