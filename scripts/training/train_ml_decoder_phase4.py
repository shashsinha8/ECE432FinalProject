#!/usr/bin/env python3
"""
Training script for ML-assisted Hamming decoder with Phase 4 architecture improvements.

Phase 4 improvements include:
1. Deeper networks (more layers)
2. Wider networks (more neurons per layer)
3. Residual connections (skip connections)

This builds on Phase 3 (improved training strategy) with architecture enhancements.
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
    DeepDirectMappingDecoder,
    WideDirectMappingDecoder,
    ResidualDirectMappingDecoder,
    HammingDataset,
    generate_training_data,
    train_model,
    save_model,
    augment_training_data
)


def get_model(architecture, use_soft_input=True, **kwargs):
    """
    Get model based on architecture type.
    
    Parameters:
    -----------
    architecture : str
        Architecture type: 'standard', 'deep', 'wide', 'residual'
    use_soft_input : bool
        Whether to use soft inputs (LLRs)
    **kwargs : dict
        Additional arguments for model initialization
    
    Returns:
    --------
    model : nn.Module
        Neural network model
    """
    if architecture == 'standard':
        return DirectMappingDecoder(
            input_size=7,
            hidden_sizes=[64, 32],
            output_size=4,
            use_soft_input=use_soft_input
        )
    elif architecture == 'deep':
        return DeepDirectMappingDecoder(
            input_size=7,
            output_size=4,
            use_soft_input=use_soft_input
        )
    elif architecture == 'wide':
        return WideDirectMappingDecoder(
            input_size=7,
            output_size=4,
            use_soft_input=use_soft_input
        )
    elif architecture == 'residual':
        return ResidualDirectMappingDecoder(
            input_size=7,
            output_size=4,
            use_soft_input=use_soft_input
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def main():
    parser = argparse.ArgumentParser(
        description='Train ML-assisted Hamming decoder with Phase 4 architecture improvements'
    )
    parser.add_argument('--architecture', type=str, 
                       choices=['standard', 'deep', 'wide', 'residual'],
                       default='deep',
                       help='Architecture type: standard, deep, wide, or residual')
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
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Device to train on')
    parser.add_argument('--output', type=str, 
                       default='models/ml_decoder_phase4.pth',
                       help='Output model path')
    parser.add_argument('--use_codeword_loss', action='store_true',
                       help='Use codeword-level loss instead of bit-level BCE')
    parser.add_argument('--augment_ratio', type=float, default=0.1,
                       help='Fraction of training data to augment (0.0 to 1.0)')
    parser.add_argument('--augment_noise', type=float, default=0.1,
                       help='Noise level for data augmentation')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Training ML Decoder with Phase 4 Architecture Improvements")
    print("=" * 70)
    print(f"Architecture: {args.architecture}")
    print(f"Training samples: {args.num_samples:,}")
    print(f"Eb/N0 range: {args.ebno_min} to {args.ebno_max} dB")
    print(f"Focused region: {args.focus_ebno_min} to {args.focus_ebno_max} dB (70% of samples)")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    print(f"Codeword loss: {args.use_codeword_loss}")
    print(f"Data augmentation: {args.augment_ratio * 100:.1f}% (noise={args.augment_noise})")
    print()
    
    # Architecture descriptions
    arch_descriptions = {
        'standard': 'Standard: 7 → 64 → 32 → 4',
        'deep': 'Deep: 7 → 128 → 64 → 32 → 16 → 4',
        'wide': 'Wide: 7 → 256 → 128 → 4',
        'residual': 'Residual: 7 → 128 → 64 → 32 → 4 (with skip connections)'
    }
    print(f"Architecture details: {arch_descriptions[args.architecture]}")
    print()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # Generate training data with Phase 3 improvements
    print("Generating training data with Phase 3 improvements...")
    print(f"  - Weighted sampling (70% from {args.focus_ebno_min}-{args.focus_ebno_max} dB)")
    print(f"  - Soft-decision inputs (LLRs)")
    
    train_inputs, train_targets = generate_training_data(
        num_samples=int(args.num_samples * 0.8),
        ebno_range=(args.ebno_min, args.ebno_max),
        approach='direct',
        seed=args.seed,
        use_soft_input=True,  # Phase 2: soft-decision
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
    
    # Generate validation data
    print("\nGenerating validation data...")
    val_inputs, val_targets = generate_training_data(
        num_samples=int(args.num_samples * 0.2),
        ebno_range=(args.ebno_min, args.ebno_max),
        approach='direct',
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
    
    # Create model with selected architecture
    print(f"Creating {args.architecture} architecture model...")
    model = get_model(args.architecture, use_soft_input=True)
    
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Train model with Phase 3 improvements
    print("Starting training...")
    print("  Using Phase 2: Soft-decision inputs (LLRs)")
    print("  Using Phase 3: Weighted sampling + Data augmentation")
    if args.use_codeword_loss:
        print("  Using Phase 3: Codeword-level loss")
    print(f"  Using Phase 4: {args.architecture.capitalize()} architecture")
    
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
        'architecture': args.architecture,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'phase4_improvements': {
            'architecture': args.architecture,
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

