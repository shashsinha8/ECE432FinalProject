#!/usr/bin/env python3
"""
Training script for ML-assisted Hamming decoder with Phase 3 improvements.

Phase 3 improvements include:
1. Weighted training data sampling (focus on error-prone Eb/N0 regions)
2. Codeword-level loss function (instead of bit-level BCE)
3. Data augmentation (add controlled noise to training samples)

This builds on Phase 2 (soft-decision inputs) with improved training strategy.
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
    save_model,
    augment_training_data
)


def main():
    parser = argparse.ArgumentParser(
        description='Train ML-assisted Hamming decoder with Phase 3 improvements'
    )
    parser.add_argument('--approach', type=str, choices=['direct', 'post'], 
                       default='direct', help='Approach: direct mapping or post-processing')
    parser.add_argument('--num_samples', type=int, default=100000,
                       help='Number of training samples')
    parser.add_argument('--ebno_min', type=float, default=-5.0,
                       help='Minimum Eb/N0 for training (dB)')
    parser.add_argument('--ebno_max', type=float, default=10.0,
                       help='Maximum Eb/N0 for training (dB)')
    parser.add_argument('--focus_ebno_min', type=float, default=0.0,
                       help='Minimum Eb/N0 for focused sampling (error-prone region)')
    parser.add_argument('--focus_ebno_max', type=float, default=5.0,
                       help='Maximum Eb/N0 for focused sampling (error-prone region)')
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
                       default='models/ml_decoder_phase3.pth',
                       help='Output model path')
    parser.add_argument('--use_codeword_loss', action='store_true',
                       help='Use codeword-level loss instead of bit-level BCE')
    parser.add_argument('--augment_ratio', type=float, default=0.1,
                       help='Fraction of training data to augment (0.0 to 1.0)')
    parser.add_argument('--augment_noise', type=float, default=0.1,
                       help='Noise level for data augmentation')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Training ML Decoder with Phase 3 Improvements")
    print("=" * 70)
    print(f"Approach: {args.approach}")
    print(f"Training samples: {args.num_samples:,}")
    print(f"Eb/N0 range: {args.ebno_min} to {args.ebno_max} dB")
    print(f"Focused region: {args.focus_ebno_min} to {args.focus_ebno_max} dB (70% of samples)")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Hidden layers: {args.hidden_sizes}")
    print(f"Device: {args.device}")
    print(f"Codeword loss: {args.use_codeword_loss}")
    print(f"Data augmentation: {args.augment_ratio * 100:.1f}% (noise={args.augment_noise})")
    print()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # Generate training data with weighted sampling and soft inputs
    print("Generating training data with weighted sampling (Phase 3)...")
    print(f"  - 70% samples from focused region ({args.focus_ebno_min}-{args.focus_ebno_max} dB)")
    print(f"  - 30% samples from full range ({args.ebno_min}-{args.ebno_max} dB)")
    print(f"  - Using soft-decision inputs (LLRs)")
    
    train_inputs, train_targets = generate_training_data(
        num_samples=int(args.num_samples * 0.8),
        ebno_range=(args.ebno_min, args.ebno_max),
        approach=args.approach,
        seed=args.seed,
        use_soft_input=True,  # Use soft-decision inputs (Phase 2)
        weighted_sampling=True,  # Phase 3: weighted sampling
        focus_ebno_range=(args.focus_ebno_min, args.focus_ebno_max)
    )
    
    # Apply data augmentation (Phase 3)
    if args.augment_ratio > 0:
        print(f"\nApplying data augmentation ({args.augment_ratio * 100:.1f}% of data)...")
        train_inputs, train_targets = augment_training_data(
            train_inputs, train_targets,
            noise_level=args.augment_noise,
            augmentation_ratio=args.augment_ratio,
            seed=args.seed + 2000
        )
        print(f"  Augmented training samples: {len(train_inputs):,}")
    
    # Generate validation data (uniform sampling, no augmentation)
    print("\nGenerating validation data...")
    val_inputs, val_targets = generate_training_data(
        num_samples=int(args.num_samples * 0.2),
        ebno_range=(args.ebno_min, args.ebno_max),
        approach=args.approach,
        seed=args.seed + 1000,
        use_soft_input=True,
        weighted_sampling=False  # Uniform for validation
    )
    
    print(f"\nTraining samples: {len(train_inputs):,}")
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
        model = PostProcessingDecoder(
            input_size=7,
            hidden_sizes=args.hidden_sizes,
            output_size=4
        )
        print("Warning: Post-processing approach typically uses hard inputs.")
    
    print(f"Model architecture:")
    print(model)
    print()
    
    # Train model with codeword loss if requested
    print("Starting training...")
    if args.use_codeword_loss:
        print("  Using codeword-level loss function (Phase 3)")
    else:
        print("  Using bit-level BCE loss")
    
    train_losses, val_losses, val_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        verbose=True,
        use_codeword_loss=args.use_codeword_loss  # Phase 3: codeword loss
    )
    
    print()
    print("=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Best validation accuracy: {max(val_accuracies):.4f} (epoch {np.argmax(val_accuracies) + 1})")
    print()
    
    # Save model
    save_model(model, args.output)
    
    # Save training history
    history_path = args.output.replace('.pth', '_history.npy')
    np.save(history_path, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'use_soft_input': True,
        'phase3_improvements': {
            'weighted_sampling': True,
            'focus_ebno_range': (args.focus_ebno_min, args.focus_ebno_max),
            'codeword_loss': args.use_codeword_loss,
            'augmentation_ratio': args.augment_ratio,
            'augmentation_noise': args.augment_noise
        }
    })
    print(f"Training history saved to {history_path}")
    print(f"\nModel saved to: {args.output}")


if __name__ == "__main__":
    main()

