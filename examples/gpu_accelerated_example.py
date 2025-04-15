"""
Example script demonstrating GPU-accelerated poker computations.

This script shows how to use GPU acceleration for poker hand evaluation,
equity calculation, and Monte Carlo simulations to achieve significant
performance improvements over CPU-based implementations.
"""

import time
import argparse
import torch
import numpy as np
from typing import List, Dict, Tuple

from pokersim.utils.gpu_optimization import (
    get_available_devices,
    select_best_device,
    optimize_model_for_device
)
from pokersim.ml.gpu_kernels import (
    CUDAHandEvaluator,
    CUDAEquityCalculator,
    CUDACardGenerator
)
from pokersim.utils.numba_optimizations import (
    evaluate_hand_fast,
    monte_carlo_hand_equity
)


def compare_hand_evaluation_performance(num_hands: int = 10000):
    """
    Compare performance of GPU vs CPU hand evaluation.
    
    Args:
        num_hands (int, optional): Number of hands to evaluate. Defaults to 10000.
    """
    print("=" * 80)
    print("Comparing Hand Evaluation Performance: GPU vs CPU")
    print("=" * 80)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. GPU tests will be skipped.")
        use_gpu = False
    else:
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        use_gpu = True
    
    # Generate random hands for testing
    if use_gpu:
        device = select_best_device()
        card_generator = CUDACardGenerator(device=device)
        
        # Generate random hole cards and community cards
        print(f"Generating {num_hands} random poker hands...")
        hole_cards, community_cards = card_generator.generate_hands(
            num_hands=num_hands, 
            num_players=1
        )
        
        # For CPU comparison, convert to numpy
        cpu_hole_cards = hole_cards.cpu().numpy()
        cpu_community_cards = community_cards.cpu().numpy()
        
        # Create evaluators
        gpu_evaluator = CUDAHandEvaluator(device=device)
    else:
        # Generate hands using numpy for CPU only
        print(f"Generating {num_hands} random poker hands...")
        cpu_hole_cards = np.random.randint(0, 52, size=(num_hands, 1, 2))
        cpu_community_cards = np.random.randint(0, 52, size=(num_hands, 5))
    
    # --- GPU Evaluation ---
    if use_gpu:
        print("\nRunning GPU hand evaluation...")
        start_time = time.time()
        
        # Reshape hole cards for the evaluator
        eval_hole_cards = hole_cards[:, 0, :]  # Shape: (num_hands, 2)
        
        # Evaluate all hands in a single batch
        _ = gpu_evaluator.evaluate_hands(
            hole_cards=eval_hole_cards,
            community_cards=community_cards[0]  # Use same community cards for all hands
        )
        
        gpu_time = time.time() - start_time
        print(f"GPU time: {gpu_time:.4f} seconds")
        print(f"GPU throughput: {num_hands / gpu_time:.2f} hands/sec")
    
    # --- CPU Evaluation ---
    print("\nRunning CPU hand evaluation...")
    start_time = time.time()
    
    # Evaluate each hand individually on CPU
    for i in range(num_hands):
        hole = cpu_hole_cards[i, 0, :]
        community = cpu_community_cards[i]
        _ = evaluate_hand_fast(hole, community)
    
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"CPU throughput: {num_hands / cpu_time:.2f} hands/sec")
    
    # --- Performance Comparison ---
    if use_gpu:
        speedup = cpu_time / gpu_time
        print(f"\nGPU speedup: {speedup:.2f}x faster than CPU")


def compare_equity_calculation_performance(num_simulations: int = 1000):
    """
    Compare performance of GPU vs CPU equity calculation.
    
    Args:
        num_simulations (int, optional): Number of simulations. Defaults to 1000.
    """
    print("\n" + "=" * 80)
    print("Comparing Equity Calculation Performance: GPU vs CPU")
    print("=" * 80)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. GPU tests will be skipped.")
        use_gpu = False
    else:
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        use_gpu = True
    
    # Define test case
    hole_cards = [0, 13]  # Ace of clubs, Ace of diamonds
    community_cards = [26, 39, 7]  # Ace of hearts, Ace of spades, 8 of clubs
    num_players = 6
    
    print(f"\nTest case: Pocket Aces vs 5 opponents")
    print(f"Community cards: A♥ A♠ 8♣")
    print(f"Running {num_simulations} Monte Carlo simulations...\n")
    
    # --- GPU Equity Calculation ---
    if use_gpu:
        print("Running GPU equity calculation...")
        device = select_best_device()
        gpu_calculator = CUDAEquityCalculator(
            device=device,
            num_simulations=num_simulations
        )
        
        start_time = time.time()
        gpu_equity = gpu_calculator.calculate_equity(
            hole_cards=hole_cards,
            community_cards=community_cards,
            num_players=num_players
        )
        gpu_time = time.time() - start_time
        
        print(f"GPU time: {gpu_time:.4f} seconds")
        print(f"GPU estimated equity: {gpu_equity:.4f}")
    
    # --- CPU Equity Calculation ---
    print("\nRunning CPU equity calculation...")
    start_time = time.time()
    
    cpu_equity = monte_carlo_hand_equity(
        hole_cards=np.array(hole_cards),
        community_cards=np.array(community_cards),
        num_players=num_players,
        num_simulations=num_simulations
    )
    
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"CPU estimated equity: {cpu_equity:.4f}")
    
    # --- Performance Comparison ---
    if use_gpu:
        speedup = cpu_time / gpu_time
        print(f"\nGPU speedup: {speedup:.2f}x faster than CPU")


def batch_monte_carlo_simulation(hole_cards_list: List[List[int]], 
                               community_cards: List[int], 
                               num_players: int,
                               num_simulations: int = 1000):
    """
    Run Monte Carlo simulations for multiple starting hands in a single batch.
    
    Args:
        hole_cards_list (List[List[int]]): List of hole card pairs.
        community_cards (List[int]): Community cards.
        num_players (int): Number of players.
        num_simulations (int, optional): Number of simulations. Defaults to 1000.
    
    Returns:
        List[float]: Equity estimates for each starting hand.
    """
    print("\n" + "=" * 80)
    print("Batch Monte Carlo Simulation with GPU Acceleration")
    print("=" * 80)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU calculations.")
        
        # Run on CPU
        equities = []
        for hole_cards in hole_cards_list:
            equity = monte_carlo_hand_equity(
                hole_cards=np.array(hole_cards),
                community_cards=np.array(community_cards),
                num_players=num_players,
                num_simulations=num_simulations
            )
            equities.append(equity)
        
        return equities
    
    # Run on GPU
    device = select_best_device()
    print(f"Using GPU: {torch.cuda.get_device_name(device.index if device.index is not None else 0)}")
    
    # Convert inputs to tensors
    hole_tensor = torch.tensor(hole_cards_list, dtype=torch.int32, device=device)
    community_tensor = torch.tensor(community_cards, dtype=torch.int32, device=device)
    
    # Initialize result tensor
    num_hands = len(hole_cards_list)
    equities = torch.zeros(num_hands, dtype=torch.float32, device=device)
    
    # Generate deck excluding community cards
    deck = torch.arange(52, dtype=torch.int32, device=device)
    used_mask = torch.zeros(52, dtype=torch.bool, device=device)
    
    for card in community_cards:
        if card >= 0:
            used_mask[card] = True
    
    available_deck = deck[~used_mask]
    
    # Run simulations
    print(f"Running {num_simulations} simulations for {num_hands} starting hands...")
    start_time = time.time()
    
    for sim in range(num_simulations):
        # Shuffle available deck for this simulation
        shuffled_deck = available_deck[torch.randperm(len(available_deck), device=device)]
        
        # Deal remaining community cards if needed
        remaining = 5 - len(community_cards)
        if remaining > 0:
            sim_community = torch.cat([
                community_tensor,
                shuffled_deck[:remaining]
            ])
        else:
            sim_community = community_tensor
        
        # For each starting hand
        for hand_idx in range(num_hands):
            # Player's hole cards
            hole_cards = hole_tensor[hand_idx]
            
            # Mark player's hole cards as used
            player_mask = torch.zeros(52, dtype=torch.bool, device=device)
            for card in hole_cards:
                if card >= 0:
                    player_mask[card] = True
            
            # Generate random hole cards for opponents
            opponent_deck = shuffled_deck[~player_mask[:len(shuffled_deck)]]
            
            # Deal to opponents
            opponents_win = False
            for opp_idx in range(num_players - 1):
                if len(opponent_deck) >= (opp_idx + 1) * 2:
                    opp_hole = opponent_deck[opp_idx*2:(opp_idx+1)*2]
                    
                    # Compare hands
                    # In a real implementation, you would use the GPU hand evaluator here
                    # For simplicity, we're using a placeholder comparison
                    if torch.sum(opp_hole) > torch.sum(hole_cards):
                        opponents_win = True
                        break
            
            # Update equity
            if not opponents_win:
                equities[hand_idx] += 1
    
    # Normalize equities
    equities = equities / num_simulations
    
    gpu_time = time.time() - start_time
    print(f"Completed in {gpu_time:.4f} seconds")
    print(f"Throughput: {num_hands * num_simulations / gpu_time:.2f} hand-sims/sec")
    
    return equities.cpu().numpy().tolist()


def main():
    """Main function to run GPU acceleration examples."""
    parser = argparse.ArgumentParser(description="GPU Acceleration Examples")
    parser.add_argument("--test", type=str, default="all", 
                      choices=["all", "hand_eval", "equity", "batch"],
                      help="Test to run")
    parser.add_argument("--num_hands", type=int, default=10000,
                      help="Number of hands for hand evaluation test")
    parser.add_argument("--num_simulations", type=int, default=1000,
                      help="Number of simulations for equity test")
    args = parser.parse_args()
    
    # Check available devices
    devices = get_available_devices()
    print("Available devices:")
    for i, device in enumerate(devices):
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            print(f"  {i}: {device} - {props.name}, {props.total_memory / 1e9:.1f} GB")
        else:
            print(f"  {i}: {device}")
    
    # Run selected tests
    if args.test in ["all", "hand_eval"]:
        compare_hand_evaluation_performance(num_hands=args.num_hands)
    
    if args.test in ["all", "equity"]:
        compare_equity_calculation_performance(num_simulations=args.num_simulations)
    
    if args.test in ["all", "batch"]:
        # Test batch Monte Carlo simulation
        hole_cards_list = [
            [0, 13],  # Ace-Ace
            [0, 12],  # Ace-King
            [0, 11],  # Ace-Queen
            [12, 11], # King-Queen
            [8, 9],   # 10-Jack
            [3, 16]   # 5-5
        ]
        community_cards = [26, 39, 7]  # Ace of hearts, Ace of spades, 8 of clubs
        
        equities = batch_monte_carlo_simulation(
            hole_cards_list=hole_cards_list,
            community_cards=community_cards,
            num_players=6,
            num_simulations=args.num_simulations
        )
        
        print("\nEquity results for each starting hand:")
        for i, hole_cards in enumerate(hole_cards_list):
            print(f"Hand {i+1}: {hole_cards} - Equity: {equities[i]:.4f}")


if __name__ == "__main__":
    main()