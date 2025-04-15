"""
Example script demonstrating distributed training for poker AI.

This script shows how to set up distributed training across multiple
processes and GPUs to accelerate the training of poker AI agents.
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Any

from pokersim.game.spingo import SpinGoGame
from pokersim.ml.models import create_policy_network
from pokersim.utils.gpu_optimization import get_available_devices, select_best_device
from pokersim.training.distributed import (
    DistributedTrainingConfig,
    train_distributed,
    DistributedLearner
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_model():
    """
    Create a policy network model for distributed training.
    
    Returns:
        nn.Module: Policy network model.
    """
    # Create policy network
    input_dim = 128  # State representation size
    hidden_dim = 256
    action_dim = 5   # Number of possible actions
    
    model = create_policy_network(
        input_size=input_dim,
        hidden_layers=[hidden_dim, hidden_dim],
        action_space_size=action_dim
    )
    
    return model


def train_poker_agent(learner: DistributedLearner, config: DistributedTrainingConfig, 
                     num_games_per_process: int = 1000):
    """
    Train a poker agent using distributed training.
    
    Args:
        learner (DistributedLearner): Distributed learner.
        config (DistributedTrainingConfig): Training configuration.
        num_games_per_process (int, optional): Number of games per process. Defaults to 1000.
    """
    # Create game environment
    game = SpinGoGame(num_players=3)
    
    # Track metrics
    total_games = 0
    start_time = time.time()
    
    # Training loop
    while total_games < num_games_per_process:
        # Start a new tournament
        game_state = game.start_new_hand()
        
        # Play until tournament is over
        while not game.is_tournament_over():
            # Play a hand
            while not game_state.is_terminal():
                # Get current player
                current_player = game_state.current_player
                
                # If it's our agent's turn (player 0)
                if current_player == 0:
                    # Get state representation
                    state = get_state_representation(game_state)
                    
                    # Choose action using the model
                    legal_actions = game_state.get_legal_actions()
                    action = choose_action(learner, state, legal_actions)
                    
                    # Execute action
                    next_state = game_state.apply_action(action)
                    
                    # Get reward (0 for non-terminal states)
                    reward = 0.0
                    if next_state.is_terminal():
                        reward = next_state.get_utility(0)
                    
                    # Store experience
                    next_state_rep = get_state_representation(next_state)
                    experience = (state, action, reward, next_state_rep, next_state.is_terminal())
                    learner.add_experience(experience)
                    
                    # Train model
                    loss = learner.train_step()
                    
                    # Synchronize model periodically
                    if total_games % config.sync_interval == 0:
                        learner.sync_model()
                    
                    # Game state update
                    game_state = next_state
                else:
                    # Other player's turn - use a simple rule-based policy
                    legal_actions = game_state.get_legal_actions()
                    action = legal_actions[0]  # Simple policy: always take first action
                    
                    # Execute action
                    game_state = game_state.apply_action(action)
            
            # Hand is finished, update game state
            game.update_stacks_after_hand()
            
            # Start a new hand if tournament is not over
            if not game.is_tournament_over():
                game_state = game.start_new_hand()
        
        # Tournament is over
        total_games += 1
        
        # Save checkpoint periodically
        if total_games % config.checkpoint_interval == 0 and learner.rank == 0:
            checkpoint_path = os.path.join("./checkpoints", f"checkpoint_{total_games}.pt")
            learner.save_checkpoint(checkpoint_path)
            
            # Also save as latest checkpoint
            latest_path = os.path.join("./checkpoints", "latest_checkpoint.pt")
            learner.save_checkpoint(latest_path)
        
        # Log progress
        if total_games % 10 == 0 and learner.rank == 0:
            elapsed_time = time.time() - start_time
            games_per_second = total_games / elapsed_time
            logger.info(f"Process {learner.rank}: Completed {total_games}/{num_games_per_process} games")
            logger.info(f"Training speed: {games_per_second:.2f} games/second")
            
            if learner.losses:
                avg_loss = sum(learner.losses[-100:]) / min(len(learner.losses), 100)
                logger.info(f"Average loss (last 100): {avg_loss:.6f}")
    
    # Final checkpoint
    if learner.rank == 0:
        final_path = os.path.join("./checkpoints", "final_checkpoint.pt")
        learner.save_checkpoint(final_path)
        logger.info(f"Training complete. Final checkpoint saved to {final_path}")


def get_state_representation(game_state: Any) -> np.ndarray:
    """
    Get a state representation from a game state.
    
    Args:
        game_state (Any): Game state.
    
    Returns:
        np.ndarray: State representation.
    """
    # This is a placeholder - in practice, you would extract meaningful
    # features from the game state
    
    # For example:
    # - Player's cards
    # - Community cards
    # - Pot size
    # - Stack sizes
    # - Betting history
    
    # Placeholder representation
    return np.random.random(128)  # 128-dimensional state representation


def choose_action(learner: DistributedLearner, state: np.ndarray, 
                legal_actions: List[Any]) -> Any:
    """
    Choose an action using the model.
    
    Args:
        learner (DistributedLearner): Distributed learner.
        state (np.ndarray): State representation.
        legal_actions (List[Any]): List of legal actions.
    
    Returns:
        Any: Chosen action.
    """
    # Convert state to tensor
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(learner.device)
    
    # Forward pass through the model
    with torch.no_grad():
        logits = learner.model(state_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    
    # Mask illegal actions
    action_mask = np.zeros(probs.shape)
    for i, action in enumerate(legal_actions):
        # This is a simplified mapping - you would need a proper mapping
        # based on your action space
        action_idx = i % len(action_mask)
        action_mask[action_idx] = 1
    
    masked_probs = probs * action_mask
    
    # Normalize probabilities
    if np.sum(masked_probs) > 0:
        masked_probs = masked_probs / np.sum(masked_probs)
    else:
        # Fallback to uniform distribution
        masked_probs = action_mask / np.sum(action_mask)
    
    # Choose action
    action_idx = np.random.choice(len(masked_probs), p=masked_probs)
    
    # Find corresponding action
    for i, action in enumerate(legal_actions):
        if i % len(action_mask) == action_idx:
            return action
    
    # Fallback to first legal action
    return legal_actions[0]


def main():
    """Main function for distributed training example."""
    parser = argparse.ArgumentParser(description="Distributed Training Example")
    parser.add_argument("--num_processes", type=int, default=1,
                      help="Number of processes for distributed training")
    parser.add_argument("--num_games", type=int, default=1000,
                      help="Number of games per process")
    parser.add_argument("--checkpoint_interval", type=int, default=100,
                      help="Interval for saving checkpoints")
    parser.add_argument("--backend", type=str, default="nccl",
                      choices=["nccl", "gloo"],
                      help="PyTorch distributed backend")
    args = parser.parse_args()
    
    # Create configuration
    config = DistributedTrainingConfig(
        world_size=args.num_processes,
        backend=args.backend,
        checkpoint_interval=args.checkpoint_interval,
        sync_interval=10,
        num_episodes=args.num_games,
        eval_interval=100
    )
    
    # Check available devices
    devices = get_available_devices()
    logger.info(f"Available devices: {devices}")
    
    # Create checkpoint directory
    os.makedirs("./checkpoints", exist_ok=True)
    
    # Launch distributed training
    train_distributed(
        model_fn=create_model,
        training_fn=train_poker_agent,
        config=config,
        checkpoint_dir="./checkpoints",
        num_games_per_process=args.num_games,
        replay_buffer_size=10000,
        batch_size=64,
        learning_rate=0.001
    )


if __name__ == "__main__":
    main()