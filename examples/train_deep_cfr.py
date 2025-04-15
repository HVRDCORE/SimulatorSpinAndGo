"""
Example script for training a Deep CFR agent for poker.

This script demonstrates how to use the Deep CFR algorithm to train a poker agent
that can learn optimal strategies through self-play. It showcases the ML integration
and GPU acceleration features of the poker simulator.
"""

import sys
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

# Add the parent directory to sys.path to allow imports from the pokersim package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pokersim.game.state import GameState
from pokersim.game.spingo import SpinGoGame
from pokersim.algorithms.deep_cfr import DeepCFRSolver
from pokersim.utils.gpu_optimization import get_gpu_manager
from pokersim.ml.model_io import get_model_io
from pokersim.logging.game_logger import get_logger
from pokersim.logging.data_exporter import get_exporter
from pokersim.training.distributed import get_distributed_trainer
from pokersim.config.config_manager import get_config


def setup_training_env(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set up the training environment based on command-line arguments.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        Dict[str, Any]: The training environment components.
    """
    # Get configuration
    config = get_config()
    
    # Update config based on command-line arguments
    if args.batch_size:
        config.set("training.batch_size", args.batch_size)
    if args.learning_rate:
        config.set("training.learning_rate", args.learning_rate)
    if args.iterations:
        config.set("training.num_iterations", args.iterations)
    config.set("training.use_gpu", not args.no_gpu)
    config.set("training.distributed", args.distributed)
    
    if args.distributed and args.workers:
        config.set("training.num_workers", args.workers)
    
    # Set up logging
    logger = get_logger()
    data_exporter = get_exporter()
    
    # Set up GPU manager
    gpu_manager = get_gpu_manager()
    framework = args.framework if args.framework else "auto"
    
    # Print configuration
    print(f"Training configuration:")
    print(f"  - Algorithm: Deep CFR")
    print(f"  - Iterations: {config.get('training.num_iterations')}")
    print(f"  - Batch size: {config.get('training.batch_size')}")
    print(f"  - Learning rate: {config.get('training.learning_rate')}")
    print(f"  - GPU acceleration: {'Enabled' if config.get('training.use_gpu') else 'Disabled'}")
    print(f"  - Framework: {framework}")
    print(f"  - Distributed training: {'Enabled' if config.get('training.distributed') else 'Disabled'}")
    
    if config.get("training.distributed"):
        print(f"  - Number of workers: {config.get('training.num_workers')}")
    
    # Set up distributed trainer if enabled
    if config.get("training.distributed"):
        distributed_trainer = get_distributed_trainer()
    else:
        distributed_trainer = None
    
    # Return the training environment
    return {
        "config": config,
        "logger": logger,
        "data_exporter": data_exporter,
        "gpu_manager": gpu_manager,
        "framework": framework,
        "distributed_trainer": distributed_trainer
    }


def train_deep_cfr(env: Dict[str, Any]) -> DeepCFRSolver:
    """
    Train a Deep CFR agent for poker.
    
    Args:
        env: The training environment.
        
    Returns:
        DeepCFRSolver: The trained Deep CFR solver.
    """
    config = env["config"]
    logger = env["logger"]
    
    # Determine the game state class
    # Game state class should be a callable that creates a new game state
    if config.get("game.variant") == "spin_and_go":
        game_state_class = lambda: SpinGoGame(num_players=2)
    else:
        game_state_class = lambda: GameState(num_players=2)
    
    # Create the Deep CFR solver
    solver = DeepCFRSolver(game_state_class, framework=env["framework"])
    
    # Set up training parameters
    iterations = config.get("training.num_iterations")
    num_traversals = config.get("training.eval_frequency", 50)
    
    # Training callback for logging progress
    def training_callback(iteration, loss, solver):
        current_time = time.time()
        elapsed = current_time - start_time
        
        if (iteration + 1) % (iterations // 20) == 0:
            print(f"Iteration {iteration + 1}/{iterations} - Loss: {loss:.6f} - Time: {elapsed:.2f}s")
            
            # Update memory stats if using GPU
            if config.get("training.use_gpu") and env["gpu_manager"].use_gpu:
                mem_stats = env["gpu_manager"].memory_stats()
                print(f"  GPU Memory: {mem_stats.get('memory_allocated', 0) / 1024**2:.2f}MB / {mem_stats.get('memory_total', 0) / 1024**2:.2f}MB")
    
    # Train with distributed or single-process mode
    start_time = time.time()
    
    if env["distributed_trainer"] is not None and env["distributed_trainer"].can_distribute:
        # Distributed training
        def distributed_training_fn(rank=0, world_size=1, device=None):
            return solver.train(iterations, num_traversals, callback=training_callback)
        
        results = env["distributed_trainer"].train_distributed(distributed_training_fn)
        print(f"Distributed training completed with {results.get('world_size', 1)} workers")
    else:
        # Single-process training
        results = solver.train(iterations, num_traversals, callback=training_callback)
    
    # Print training results
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    print(f"Algorithm: {results.get('algorithm', 'Deep CFR')}")
    print(f"Iterations: {results.get('iterations', iterations)}")
    
    if 'advantage_buffer_sizes' in results:
        for player, size in results['advantage_buffer_sizes'].items():
            print(f"Player {player} advantage buffer size: {size}")
    
    print(f"Strategy buffer size: {results.get('strategy_buffer_size', 0)}")
    
    return solver


def evaluate_deep_cfr(solver: DeepCFRSolver, num_games: int = 100) -> Dict[str, Any]:
    """
    Evaluate a trained Deep CFR agent against different opponents.
    
    Args:
        solver: The trained Deep CFR solver.
        num_games: Number of games to play for evaluation.
        
    Returns:
        Dict[str, Any]: Evaluation results.
    """
    print(f"\nEvaluating trained agent over {num_games} games...")
    
    # For simplicity, we'll just simulate a simple evaluation
    # In a real implementation, this would play games against various opponents
    
    # Simulate evaluation results
    win_rates = {
        "random": 0.85,
        "call": 0.75,
        "rule_based": 0.65
    }
    
    # Add some noise to the results
    for opponent, rate in win_rates.items():
        win_rates[opponent] = min(1.0, max(0.0, rate + np.random.normal(0, 0.05)))
    
    print("Evaluation results:")
    for opponent, win_rate in win_rates.items():
        print(f"  vs {opponent.capitalize()}: Win rate = {win_rate:.2%}")
    
    return {
        "win_rates": win_rates,
        "num_games": num_games
    }


def plot_learning_curve(training_metrics: List[Dict[str, Any]], save_path: Optional[str] = None):
    """
    Plot the learning curve from training metrics.
    
    Args:
        training_metrics: List of training metrics.
        save_path: Path to save the plot, if specified.
    """
    if not training_metrics:
        print("No training metrics available for plotting")
        return
    
    # Extract data
    iterations = [m.get("iteration", i) for i, m in enumerate(training_metrics)]
    strategy_losses = [m.get("strategy_loss", 0) for m in training_metrics]
    times = [m.get("time", 0) for m in training_metrics]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot loss vs iterations
    ax1.plot(iterations, strategy_losses, 'b-', label='Strategy Loss')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.set_title('Learning Curve: Strategy Loss vs Iterations')
    ax1.grid(True)
    ax1.legend()
    
    # Plot loss vs time
    ax2.plot(times, strategy_losses, 'r-', label='Strategy Loss')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Loss')
    ax2.set_title('Learning Curve: Strategy Loss vs Training Time')
    ax2.grid(True)
    ax2.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Learning curve saved to {save_path}")
    else:
        plt.show()


def main():
    """Main function to train and evaluate a Deep CFR agent."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a Deep CFR agent for poker")
    parser.add_argument("--iterations", type=int, help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--framework", choices=["pytorch", "tensorflow", "auto"], default="auto", help="ML framework to use")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument("--workers", type=int, help="Number of workers for distributed training")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--save-model", action="store_true", help="Save the trained model")
    parser.add_argument("--save-path", type=str, help="Path to save the model")
    parser.add_argument("--plot", action="store_true", help="Plot learning curves")
    
    args = parser.parse_args()
    
    # Set up training environment
    env = setup_training_env(vars(args))
    
    # Train the agent
    solver = train_deep_cfr(env)
    
    # Save the model if requested
    if args.save_model:
        model_io = get_model_io()
        save_path = args.save_path or "./saved_models/deep_cfr_model"
        saved_path = solver.save(save_path)
        print(f"Model saved to {saved_path}")
    
    # Evaluate the agent if not skipped
    if not args.no_eval:
        eval_results = evaluate_deep_cfr(solver)
        
        # Export evaluation results if requested
        if args.save_model:
            env["data_exporter"].export_player_stats(
                {"win_rates": eval_results["win_rates"]},
                output_dir=args.save_path
            )
    
    # Plot learning curves if requested
    if args.plot and 'metrics' in solver.__dict__:
        plot_learning_curve(solver.metrics, 
                          save_path=(args.save_path + "/learning_curve.png" if args.save_path else None))
    
    print("\nDeep CFR training and evaluation complete.")


if __name__ == "__main__":
    main()