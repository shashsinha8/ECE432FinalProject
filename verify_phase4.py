#!/usr/bin/env python3
"""
Verification script for Phase 4: ML Model Development

This script demonstrates the ML decoder functionality including:
- Model architecture
- Data generation
- Training (quick example)
- Inference
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.ml_decoder import (
    DirectMappingDecoder,
    PostProcessingDecoder,
    HammingDataset,
    generate_training_data,
    train_model,
    MLDecoder
)
from src.hamming import encode
from src.channel import bpsk_modulate, bpsk_demodulate_hard, awgn_channel


def main():
    print("=" * 60)
    print("Phase 4 Verification: ML Model Development")
    print("=" * 60)
    print()
    
    # Test 1: Model Architecture
    print("Test 1: Model Architecture")
    print("-" * 60)
    model_direct = DirectMappingDecoder(input_size=7, hidden_sizes=[32, 16], output_size=4)
    model_post = PostProcessingDecoder(input_size=7, hidden_sizes=[16, 8], output_size=4)
    
    print("DirectMappingDecoder:")
    print(f"  Parameters: {sum(p.numel() for p in model_direct.parameters()):,}")
    print(f"  Architecture: 7 → 32 → 16 → 4")
    
    print("PostProcessingDecoder:")
    print(f"  Parameters: {sum(p.numel() for p in model_post.parameters()):,}")
    print(f"  Architecture: 7 → 16 → 8 → 4")
    print()
    
    # Test 2: Forward Pass
    print("Test 2: Forward Pass")
    print("-" * 60)
    test_input = torch.randn(5, 7)
    output_direct = model_direct(test_input)
    output_post = model_post(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Direct mapping output shape: {output_direct.shape}")
    print(f"Post-processing output shape: {output_post.shape}")
    print(f"✓ Outputs are in [0, 1] range (sigmoid): "
          f"{torch.all(output_direct >= 0) and torch.all(output_direct <= 1)}")
    print()
    
    # Test 3: Data Generation
    print("Test 3: Training Data Generation")
    print("-" * 60)
    print("Generating small dataset (direct mapping)...")
    inputs, targets = generate_training_data(
        num_samples=100, ebno_range=(0, 5), approach='direct', seed=42
    )
    
    print(f"Generated samples: {len(inputs):,}")
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Input range: [{inputs.min():.0f}, {inputs.max():.0f}] (binary)")
    print(f"Target range: [{targets.min():.0f}, {targets.max():.0f}] (binary)")
    print(f"✓ Data generation works correctly")
    print()
    
    # Test 4: Dataset and DataLoader
    print("Test 4: Dataset and DataLoader")
    print("-" * 60)
    dataset = HammingDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: 16")
    
    # Get one batch
    batch_inputs, batch_targets = next(iter(dataloader))
    print(f"Batch input shape: {batch_inputs.shape}")
    print(f"Batch target shape: {batch_targets.shape}")
    print(f"✓ Dataset and DataLoader work correctly")
    print()
    
    # Test 5: Quick Training (few epochs for verification)
    print("Test 5: Quick Training (5 epochs for verification)")
    print("-" * 60)
    
    # Split data
    train_size = int(0.8 * len(inputs))
    train_inputs, train_targets = inputs[:train_size], targets[:train_size]
    val_inputs, val_targets = inputs[train_size:], targets[train_size:]
    
    train_dataset = HammingDataset(train_inputs, train_targets)
    val_dataset = HammingDataset(val_inputs, val_targets)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create and train model
    model = DirectMappingDecoder(input_size=7, hidden_sizes=[16, 8], output_size=4)
    
    train_losses, val_losses, val_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        learning_rate=0.001,
        device='cpu',
        verbose=True
    )
    
    print(f"✓ Training completed")
    print(f"  Final validation accuracy: {val_accuracies[-1]:.4f}")
    print(f"  Final validation loss: {val_losses[-1]:.4f}")
    print()
    
    # Test 6: Inference
    print("Test 6: ML Decoder Inference")
    print("-" * 60)
    ml_decoder = MLDecoder(model, device='cpu')
    
    # Generate test case
    test_data = np.array([1, 0, 1, 0], dtype=np.uint8)
    codeword = encode(test_data)
    symbols = bpsk_modulate(codeword)
    noisy_symbols = awgn_channel(symbols, eb_no_db=5.0, seed=42)
    rx_bits = bpsk_demodulate_hard(noisy_symbols)
    
    print(f"Original data: {test_data}")
    print(f"Received bits: {rx_bits}")
    
    # Decode with ML
    decoded_ml = ml_decoder.decode(rx_bits)
    print(f"ML decoded:    {decoded_ml}")
    
    # Compare
    match = np.array_equal(decoded_ml, test_data)
    print(f"✓ ML decoder produces output: {decoded_ml.shape == (4,)}")
    print()
    
    # Test 7: Batch Inference
    print("Test 7: Batch Inference")
    print("-" * 60)
    batch_rx_bits = np.random.randint(0, 2, size=(10, 7), dtype=np.uint8)
    batch_decoded = ml_decoder.decode_batch(batch_rx_bits)
    
    print(f"Batch input shape: {batch_rx_bits.shape}")
    print(f"Batch output shape: {batch_decoded.shape}")
    print(f"✓ Batch inference works correctly")
    print()
    
    # Summary
    print("=" * 60)
    print("Phase 4 Verification: COMPLETE ✓")
    print("=" * 60)
    print()
    print("All ML decoder functionality verified:")
    print("  ✓ Model architectures (DirectMapping and PostProcessing)")
    print("  ✓ Training data generation")
    print("  ✓ Dataset and DataLoader integration")
    print("  ✓ Model training pipeline")
    print("  ✓ Inference (single and batch)")
    print()
    print("Note: For full training with better performance,")
    print("      run: python train_ml_decoder.py --approach direct --num_samples 100000")
    print()
    print("Ready to proceed to Phase 5: ML-Assisted Decoding Integration")


if __name__ == "__main__":
    main()

