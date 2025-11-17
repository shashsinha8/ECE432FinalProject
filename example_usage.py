#!/usr/bin/env python3
"""
Example Usage Script

This script demonstrates how to use the ML-assisted Hamming decoder
for various tasks.
"""

import numpy as np
from src.hamming import encode
from src.classical_decoder import ClassicalDecoder
from src.ml_decoder import DirectMappingDecoder, MLDecoder, load_model
from src.channel import bpsk_modulate, bpsk_demodulate_hard, awgn_channel
from src.evaluation import calculate_ber


def example_1_basic_encoding_decoding():
    """Example 1: Basic Hamming encoding and decoding."""
    print("=" * 60)
    print("Example 1: Basic Hamming Encoding/Decoding")
    print("=" * 60)
    
    # Original data
    data = np.array([1, 0, 1, 1], dtype=np.uint8)
    print(f"Original data: {data}")
    
    # Encode
    codeword = encode(data)
    print(f"Encoded codeword: {codeword}")
    
    # Decode (no errors)
    from src.hamming import decode
    decoded, _, _ = decode(codeword)
    print(f"Decoded data: {decoded}")
    print(f"Match: {np.array_equal(data, decoded)}")
    print()


def example_2_error_correction():
    """Example 2: Single-bit error correction."""
    print("=" * 60)
    print("Example 2: Single-Bit Error Correction")
    print("=" * 60)
    
    data = np.array([1, 0, 1, 0], dtype=np.uint8)
    codeword = encode(data)
    
    # Introduce error
    corrupted = codeword.copy()
    corrupted[2] = 1 - corrupted[2]
    print(f"Original codeword: {codeword}")
    print(f"Corrupted codeword: {corrupted} (error at position 2)")
    
    # Decode
    from src.hamming import decode
    decoded, corrected, error_pos = decode(corrupted)
    print(f"Corrected codeword: {corrected}")
    print(f"Decoded data: {decoded}")
    print(f"Error detected at position: {error_pos}")
    print(f"Recovered correctly: {np.array_equal(data, decoded)}")
    print()


def example_3_channel_simulation():
    """Example 3: Channel simulation."""
    print("=" * 60)
    print("Example 3: Channel Simulation")
    print("=" * 60)
    
    data = np.array([0, 1, 0, 1], dtype=np.uint8)
    codeword = encode(data)
    
    # Modulate
    symbols = bpsk_modulate(codeword)
    print(f"BPSK symbols: {symbols}")
    
    # Add noise
    eb_no_db = 5.0
    noisy_symbols = awgn_channel(symbols, eb_no_db, seed=42)
    print(f"Noisy symbols: {noisy_symbols}")
    
    # Demodulate
    rx_bits = bpsk_demodulate_hard(noisy_symbols)
    print(f"Received bits: {rx_bits}")
    
    # Decode
    decoder = ClassicalDecoder()
    decoded, _, _ = decoder.decode(rx_bits)
    print(f"Decoded data: {decoded}")
    
    ber, errors, total = calculate_ber(data, decoded)
    print(f"BER: {ber:.4f} ({errors}/{total} errors)")
    print()


def example_4_ml_decoder():
    """Example 4: Using ML decoder (requires trained model)."""
    print("=" * 60)
    print("Example 4: ML Decoder Usage")
    print("=" * 60)
    
    # Create or load model
    model = DirectMappingDecoder(input_size=7, hidden_sizes=[16, 8], output_size=4)
    ml_decoder = MLDecoder(model, device='cpu')
    
    # Generate test case
    data = np.array([1, 0, 1, 0], dtype=np.uint8)
    codeword = encode(data)
    symbols = bpsk_modulate(codeword)
    noisy_symbols = awgn_channel(symbols, eb_no_db=5.0, seed=42)
    rx_bits = bpsk_demodulate_hard(noisy_symbols)
    
    print(f"Original data: {data}")
    print(f"Received bits: {rx_bits}")
    
    # Decode with ML
    ml_decoded = ml_decoder.decode(rx_bits)
    print(f"ML decoded: {ml_decoded}")
    
    # Compare with classical
    classical_decoder = ClassicalDecoder()
    classical_decoded, _, _ = classical_decoder.decode(rx_bits)
    print(f"Classical decoded: {classical_decoded}")
    print()
    
    print("Note: For best results, use a trained model:")
    print("  model = load_model(model, 'models/ml_decoder_direct.pth')")
    print()


def example_5_batch_processing():
    """Example 5: Batch processing."""
    print("=" * 60)
    print("Example 5: Batch Processing")
    print("=" * 60)
    
    # Generate multiple codewords
    num_codewords = 10
    data_batch = []
    codewords = []
    
    for i in range(num_codewords):
        data = np.random.randint(0, 2, size=4, dtype=np.uint8)
        data_batch.append(data)
        codewords.append(encode(data))
    
    codewords = np.array(codewords)
    print(f"Generated {num_codewords} codewords")
    
    # Process with classical decoder
    decoder = ClassicalDecoder()
    decoded_batch, _, _ = decoder.decode_batch(codewords)
    
    # Check accuracy
    correct = 0
    for i in range(num_codewords):
        if np.array_equal(data_batch[i], decoded_batch[i]):
            correct += 1
    
    print(f"Correctly decoded: {correct}/{num_codewords}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ML-Assisted Hamming Decoder - Example Usage")
    print("=" * 60 + "\n")
    
    example_1_basic_encoding_decoding()
    example_2_error_correction()
    example_3_channel_simulation()
    example_4_ml_decoder()
    example_5_batch_processing()
    
    print("=" * 60)
    print("Examples Complete!")
    print("=" * 60)

