#!/usr/bin/env python3
"""
Training script for ML-assisted Hamming decoder.

This script trains neural network models for either:
1. Direct mapping: 7 received bits → 4 data bits
2. Post-processing: 7 bits from classical decoder → 4 corrected data bits
"""

import argparse
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
    parser = argparse.ArgumentParser(description='Train ML-assisted Hamming decoder')
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
                       default='models/ml_decoder_direct.pth',
                       help='Output model path')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Training ML Decoder: {args.approach} approach")
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
    
    # Generate training data
    print("Generating training data...")
    train_inputs, train_targets = generate_training_data(
        num_samples=int(args.num_samples * 0.8),
        ebno_range=(args.ebno_min, args.ebno_max),
        approach=args.approach,
        seed=args.seed
    )
    
    # Generate validation data
    print("Generating validation data...")
    val_inputs, val_targets = generate_training_data(
        num_samples=int(args.num_samples * 0.2),
        ebno_range=(args.ebno_min, args.ebno_max),
        approach=args.approach,
        seed=args.seed + 1000
    )
    
    print(f"Training samples: {len(train_inputs):,}")
    print(f"Validation samples: {len(val_inputs):,}")
    print()
    
    # Create datasets and data loaders
    train_dataset = HammingDataset(train_inputs, train_targets)
    val_dataset = HammingDataset(val_inputs, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    if args.approach == 'direct':
        model = DirectMappingDecoder(
            input_size=7,
            hidden_sizes=args.hidden_sizes,
            output_size=4
        )
    else:
        model = PostProcessingDecoder(
            input_size=7,
            hidden_sizes=args.hidden_sizes,
            output_size=4
        )
    
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
        'val_accuracies': val_accuracies
    })
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()

