#!/usr/bin/env python3
"""
Training script for ML-assisted Hamming decoder with soft-decision inputs (LLRs).

This script trains neural network models using Log-Likelihood Ratios (LLRs)
instead of hard-decision bits, providing the model with reliability information.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.ml_decoder import (
    DirectMappingDecoder,
    PostProcessingDecoder,
    HammingDataset,
    generate_training_data,
    train_model,
    save_model
)


def main():
    parser = argparse.ArgumentParser(
        description='Train ML-assisted Hamming decoder with soft-decision inputs (LLRs)'
    )
    parser.add_argument('--approach', type=str, choices=['direct', 'post'], 
                       default='direct', help='Approach: direct mapping or post-processing')
    parser.add_argument('--num_samples', type=int, default=100000,
                       help='Number of training samples')
    parser.add_argument('--ebno_min', type=float, default=-5.0,
                       help='Minimum Eb/N0 for training (dB)')
    parser.add_argument('--ebno_max', type=float, default=10.0,
                       help='Maximum Eb/N0 for training (dB)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', 
                       default=[64, 32], help='Hidden layer sizes')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Device to train on')
    parser.add_argument('--output', type=str, 
                       default='models/ml_decoder_direct_soft.pth',
                       help='Output model path')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Training ML Decoder with Soft-Decision Inputs (LLRs)")
    print(f"Approach: {args.approach}")
    print("=" * 60)
    print(f"Training samples: {args.num_samples:,}")
    print(f"Eb/N0 range: {args.ebno_min} to {args.ebno_max} dB")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Hidden layers: {args.hidden_sizes}")
    print(f"Device: {args.device}")
    print()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # Generate training data with soft inputs
    print("Generating training data with soft-decision inputs (LLRs)...")
    train_inputs, train_targets = generate_training_data(
        num_samples=int(args.num_samples * 0.8),
        ebno_range=(args.ebno_min, args.ebno_max),
        approach=args.approach,
        seed=args.seed,
        use_soft_input=True  # Use soft-decision inputs
    )
    
    # Generate validation data with soft inputs
    print("Generating validation data with soft-decision inputs (LLRs)...")
    val_inputs, val_targets = generate_training_data(
        num_samples=int(args.num_samples * 0.2),
        ebno_range=(args.ebno_min, args.ebno_max),
        approach=args.approach,
        seed=args.seed + 1000,
        use_soft_input=True  # Use soft-decision inputs
    )
    
    print(f"Training samples: {len(train_inputs):,}")
    print(f"Validation samples: {len(val_inputs):,}")
    print(f"Input range: [{train_inputs.min():.2f}, {train_inputs.max():.2f}] (LLRs)")
    print()
    
    # Create datasets and data loaders
    train_dataset = HammingDataset(train_inputs, train_targets)
    val_dataset = HammingDataset(val_inputs, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model with soft input support
    if args.approach == 'direct':
        model = DirectMappingDecoder(
            input_size=7,
            hidden_sizes=args.hidden_sizes,
            output_size=4,
            use_soft_input=True  # Enable soft input mode
        )
    else:
        # Post-processing doesn't typically use soft inputs
        # (it processes already-decoded hard bits)
        model = PostProcessingDecoder(
            input_size=7,
            hidden_sizes=args.hidden_sizes,
            output_size=4
        )
        print("Warning: Post-processing approach typically uses hard inputs.")
        print("Soft inputs may not be applicable for post-processing.")
    
    print(f"Model architecture:")
    print(model)
    print()
    
    # Train model
    print("Starting training...")
    train_losses, val_losses, val_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        verbose=True
    )
    
    print()
    print("Training complete!")
    print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print()
    
    # Save model
    save_model(model, args.output)
    
    # Save training history
    history_path = args.output.replace('.pth', '_history.npy')
    np.save(history_path, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'use_soft_input': True
    })
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()

