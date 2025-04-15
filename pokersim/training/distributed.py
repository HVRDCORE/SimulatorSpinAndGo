"""
Distributed training utilities for poker AI.

This module provides tools for distributing poker AI training across multiple
processes, machines, and GPUs using PyTorch's distributed training capabilities.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
import time
import logging
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """
    Initialize the distributed environment.
    
    Args:
        rank (int): Process rank.
        world_size (int): Total number of processes.
        backend (str, optional): PyTorch distributed backend. Defaults to "nccl".
    """
    # Set environment variables for PyTorch distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    # Set device for this process
    torch.cuda.set_device(rank)
    
    logger.info(f"Initialized process {rank}/{world_size}")


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device(local_rank: int) -> torch.device:
    """
    Get the appropriate device for the current process.
    
    Args:
        local_rank (int): Local process rank.
    
    Returns:
        torch.device: Device for the current process.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    else:
        return torch.device("cpu")


def prepare_model_for_distributed(model: nn.Module, local_rank: int) -> nn.Module:
    """
    Prepare a model for distributed training.
    
    Args:
        model (nn.Module): PyTorch model.
        local_rank (int): Local process rank.
    
    Returns:
        nn.Module: Model wrapped for distributed training.
    """
    # Move model to the appropriate device
    device = get_device(local_rank)
    model = model.to(device)
    
    # Wrap model with DDP
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    return ddp_model


def distributed_trainer(rank: int, world_size: int, training_fn: Callable, args: Tuple):
    """
    Distributed trainer function that runs on each process.
    
    Args:
        rank (int): Process rank.
        world_size (int): Total number of processes.
        training_fn (Callable): Training function to run.
        args (Tuple): Arguments to pass to the training function.
    """
    try:
        # Initialize distributed environment
        setup_distributed(rank, world_size)
        
        # Run training function
        training_fn(rank, world_size, *args)
    except Exception as e:
        logger.error(f"Error in process {rank}: {e}")
        raise e
    finally:
        # Clean up
        cleanup_distributed()


def launch_distributed_training(training_fn: Callable, args: Tuple, world_size: int):
    """
    Launch distributed training across multiple processes.
    
    Args:
        training_fn (Callable): Training function to run.
        args (Tuple): Arguments to pass to the training function.
        world_size (int): Number of processes to launch.
    """
    if world_size > 1 and torch.cuda.is_available():
        # Launch multiple processes
        mp.spawn(
            distributed_trainer,
            args=(world_size, training_fn, args),
            nprocs=world_size,
            join=True
        )
    else:
        # Run in a single process
        logger.info("Running in a single process (no distributed training)")
        training_fn(0, 1, *args)


def all_reduce_dict(input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Perform all-reduce operation on a dictionary of tensors.
    
    Args:
        input_dict (Dict[str, torch.Tensor]): Dictionary of tensors.
    
    Returns:
        Dict[str, torch.Tensor]: Dictionary with reduced tensors.
    """
    if not dist.is_initialized():
        return input_dict
    
    # Make sure all tensors are on CUDA
    for k, v in input_dict.items():
        if not v.is_cuda:
            input_dict[k] = v.cuda()
    
    # Create a list of tensors and keys
    keys = list(input_dict.keys())
    tensors = [input_dict[k] for k in keys]
    
    # All-reduce
    dist.all_reduce_coalesced(tensors, dist.ReduceOp.SUM)
    
    # Divide by world size
    world_size = dist.get_world_size()
    reduced_dict = {k: t / world_size for k, t in zip(keys, tensors)}
    
    return reduced_dict


def broadcast_model_parameters(model: nn.Module, src: int = 0):
    """
    Broadcast model parameters from one process to all others.
    
    Args:
        model (nn.Module): PyTorch model.
        src (int, optional): Source process. Defaults to 0.
    """
    if not dist.is_initialized():
        return
    
    for param in model.parameters():
        dist.broadcast(param.data, src=src)


class DistributedReplayBuffer:
    """
    Distributed replay buffer for reinforcement learning.
    
    This buffer distributes experiences across multiple processes and
    synchronizes sampling to ensure all processes have the same data.
    
    Attributes:
        capacity (int): Maximum size of the buffer.
        batch_size (int): Batch size for sampling.
        rank (int): Process rank.
        world_size (int): Total number of processes.
        buffer (List): List of experiences.
    """
    
    def __init__(self, capacity: int, batch_size: int, 
                rank: int, world_size: int):
        """
        Initialize a distributed replay buffer.
        
        Args:
            capacity (int): Maximum size of the buffer.
            batch_size (int): Batch size for sampling.
            rank (int): Process rank.
            world_size (int): Total number of processes.
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        
        # Each process stores a portion of the buffer
        self.local_capacity = capacity // world_size
        self.buffer = []
        
        logger.info(f"Process {rank}: Initialized distributed buffer with local capacity {self.local_capacity}")
    
    def push(self, experience: Tuple):
        """
        Add an experience to the buffer.
        
        Args:
            experience (Tuple): Experience tuple.
        """
        # Determine if this process should store the experience
        if len(self.buffer) < self.local_capacity:
            self.buffer.append(experience)
        else:
            # Replace a random experience
            idx = random.randint(0, self.local_capacity - 1)
            self.buffer[idx] = experience
    
    def sample(self) -> List:
        """
        Sample a batch of experiences from all processes.
        
        Returns:
            List: A batch of experiences.
        """
        # Each process samples locally
        local_batch_size = self.batch_size // self.world_size
        local_samples = random.sample(self.buffer, min(local_batch_size, len(self.buffer)))
        
        # Convert to tensors
        local_samples_tensor = self._experiences_to_tensor(local_samples)
        
        # Gather samples from all processes
        all_samples = self._gather_samples(local_samples_tensor)
        
        # Convert back to experience tuples
        return self._tensor_to_experiences(all_samples)
    
    def _experiences_to_tensor(self, experiences: List[Tuple]) -> torch.Tensor:
        """
        Convert experiences to a tensor for communication.
        
        Args:
            experiences (List[Tuple]): List of experience tuples.
        
        Returns:
            torch.Tensor: Tensor representation of experiences.
        """
        # This is a simplified implementation - in practice, you would need to
        # handle different types of experiences and ensure proper serialization
        
        # For now, just return a dummy tensor
        return torch.zeros(len(experiences), 10).cuda()
    
    def _tensor_to_experiences(self, tensor: torch.Tensor) -> List[Tuple]:
        """
        Convert a tensor back to experiences.
        
        Args:
            tensor (torch.Tensor): Tensor representation of experiences.
        
        Returns:
            List[Tuple]: List of experience tuples.
        """
        # This is a simplified implementation - in practice, you would need to
        # deserialize the tensor back to experiences
        
        # For now, just return dummy experiences
        return [(None, None, 0.0, None, False) for _ in range(tensor.shape[0])]
    
    def _gather_samples(self, local_samples: torch.Tensor) -> torch.Tensor:
        """
        Gather samples from all processes.
        
        Args:
            local_samples (torch.Tensor): Local samples tensor.
        
        Returns:
            torch.Tensor: Tensor with samples from all processes.
        """
        if not dist.is_initialized():
            return local_samples
        
        # Get shape information
        local_size = torch.tensor([local_samples.shape[0]], device=local_samples.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(self.world_size)]
        
        # Gather sizes from all processes
        dist.all_gather(all_sizes, local_size)
        all_sizes = [size.item() for size in all_sizes]
        
        # Prepare output tensor
        total_size = sum(all_sizes)
        output_shape = list(local_samples.shape)
        output_shape[0] = total_size
        all_samples = torch.zeros(output_shape, device=local_samples.device)
        
        # Gather samples from all processes
        if sum(all_sizes) > 0:
            # Perform gather operation
            start_idx = 0
            for i, size in enumerate(all_sizes):
                if size > 0:
                    end_idx = start_idx + size
                    all_samples_i = all_samples[start_idx:end_idx]
                    
                    if i == self.rank:
                        all_samples_i.copy_(local_samples)
                    else:
                        dist.broadcast(all_samples_i, i)
                    
                    start_idx = end_idx
        
        return all_samples


class DistributedLearner:
    """
    Distributed learner for poker AI training.
    
    This class handles distributed training of poker AI models
    across multiple processes and machines.
    
    Attributes:
        rank (int): Process rank.
        world_size (int): Total number of processes.
        model (nn.Module): Model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        replay_buffer (DistributedReplayBuffer): Distributed replay buffer.
    """
    
    def __init__(self, model: nn.Module, rank: int, world_size: int, 
                replay_buffer_size: int = 100000, batch_size: int = 128,
                learning_rate: float = 0.001):
        """
        Initialize a distributed learner.
        
        Args:
            model (nn.Module): Model to train.
            rank (int): Process rank.
            world_size (int): Total number of processes.
            replay_buffer_size (int, optional): Replay buffer size. Defaults to 100000.
            batch_size (int, optional): Batch size. Defaults to 128.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
        """
        self.rank = rank
        self.world_size = world_size
        
        # Prepare model for distributed training
        self.device = get_device(rank)
        self.model = prepare_model_for_distributed(model, rank)
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Create replay buffer
        self.replay_buffer = DistributedReplayBuffer(
            capacity=replay_buffer_size,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size
        )
        
        # Metrics
        self.train_steps = 0
        self.losses = []
    
    def add_experience(self, experience: Tuple):
        """
        Add an experience to the replay buffer.
        
        Args:
            experience (Tuple): Experience tuple.
        """
        self.replay_buffer.push(experience)
    
    def train_step(self) -> float:
        """
        Perform a distributed training step.
        
        Returns:
            float: Training loss.
        """
        # Check if we have enough samples
        if len(self.replay_buffer.buffer) < self.replay_buffer.batch_size // self.world_size:
            return 0.0
        
        # Sample from replay buffer
        batch = self.replay_buffer.sample()
        
        # Process batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones, dtype=np.float32)).unsqueeze(1).to(self.device)
        
        # Forward pass
        predictions = self.model(states)
        
        # Compute loss
        loss = nn.MSELoss()(predictions, rewards)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Synchronize gradients
        self.sync_gradients()
        
        # Optimizer step
        self.optimizer.step()
        
        # Track metrics
        self.train_steps += 1
        self.losses.append(loss.item())
        
        return loss.item()
    
    def sync_gradients(self):
        """Synchronize gradients across processes."""
        if not dist.is_initialized():
            return
        
        # All-reduce gradients
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.data.div_(self.world_size)
    
    def sync_model(self):
        """Synchronize model parameters across processes."""
        if not dist.is_initialized():
            return
        
        # Broadcast model parameters from process 0
        broadcast_model_parameters(self.model.module, src=0)
    
    def save_checkpoint(self, filepath: str):
        """
        Save a checkpoint of the distributed training.
        
        Args:
            filepath (str): Path to save the checkpoint.
        """
        # Only save from rank 0
        if self.rank == 0:
            checkpoint = {
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_steps': self.train_steps,
                'losses': self.losses
            }
            torch.save(checkpoint, filepath)
            logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load a checkpoint for distributed training.
        
        Args:
            filepath (str): Path to the checkpoint.
        """
        if not os.path.exists(filepath):
            logger.warning(f"Checkpoint {filepath} does not exist")
            return
        
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model parameters
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load metrics
        self.train_steps = checkpoint['train_steps']
        self.losses = checkpoint['losses']
        
        # Synchronize model parameters
        self.sync_model()
        
        logger.info(f"Loaded checkpoint from {filepath}")


class DistributedTrainingConfig:
    """
    Configuration for distributed training.
    
    Attributes:
        world_size (int): Number of processes to use.
        backend (str): Distributed backend to use.
        checkpoint_interval (int): Interval for saving checkpoints.
        sync_interval (int): Interval for synchronizing models.
        num_episodes (int): Number of episodes to train for.
        eval_interval (int): Interval for evaluation.
    """
    
    def __init__(self, world_size: int = 1, backend: str = "nccl",
                checkpoint_interval: int = 1000, sync_interval: int = 10,
                num_episodes: int = 10000, eval_interval: int = 100):
        """
        Initialize distributed training configuration.
        
        Args:
            world_size (int, optional): Number of processes. Defaults to 1.
            backend (str, optional): Distributed backend. Defaults to "nccl".
            checkpoint_interval (int, optional): Checkpoint interval. Defaults to 1000.
            sync_interval (int, optional): Sync interval. Defaults to 10.
            num_episodes (int, optional): Number of episodes. Defaults to 10000.
            eval_interval (int, optional): Evaluation interval. Defaults to 100.
        """
        self.world_size = world_size
        self.backend = backend
        self.checkpoint_interval = checkpoint_interval
        self.sync_interval = sync_interval
        self.num_episodes = num_episodes
        self.eval_interval = eval_interval
        
        # Adjust world size based on available GPUs
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if self.world_size > num_gpus:
                logger.warning(f"Requested {self.world_size} processes but only {num_gpus} GPUs available")
                self.world_size = num_gpus
        else:
            if self.world_size > 1:
                logger.warning("No GPUs available, setting world_size to 1")
                self.world_size = 1
                self.backend = "gloo"


def train_distributed(model_fn: Callable, training_fn: Callable, config: DistributedTrainingConfig,
                    checkpoint_dir: str = "./checkpoints", *args, **kwargs):
    """
    Train a model using distributed training.
    
    Args:
        model_fn (Callable): Function that creates the model.
        training_fn (Callable): Function that performs training.
        config (DistributedTrainingConfig): Training configuration.
        checkpoint_dir (str, optional): Directory for checkpoints. Defaults to "./checkpoints".
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Wrap training function
    def wrapped_training_fn(rank: int, world_size: int, *training_args):
        # Create model
        model = model_fn()
        
        # Create learner
        learner = DistributedLearner(
            model=model,
            rank=rank,
            world_size=world_size,
            replay_buffer_size=kwargs.get('replay_buffer_size', 100000),
            batch_size=kwargs.get('batch_size', 128),
            learning_rate=kwargs.get('learning_rate', 0.001)
        )
        
        # Load checkpoint if available
        checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        if os.path.exists(checkpoint_path):
            learner.load_checkpoint(checkpoint_path)
        
        # Run training function
        training_fn(learner, config, *training_args)
    
    # Launch distributed training
    launch_distributed_training(wrapped_training_fn, args, config.world_size)