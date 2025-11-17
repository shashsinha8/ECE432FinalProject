"""
Evaluation Framework for Hamming Code Decoder Performance

This module provides functions to evaluate decoder performance by calculating
Bit Error Rate (BER) vs. Eb/N0 curves.
"""

import numpy as np
from src.hamming import encode
from src.classical_decoder import ClassicalDecoder
from src.channel import (
    bpsk_modulate,
    bpsk_demodulate_hard,
    awgn_channel,
    generate_ebno_range
)


def calculate_ber(original_bits, decoded_bits):
    """
    Calculate Bit Error Rate (BER).
    
    BER = (number of bit errors) / (total number of bits)
    
    Parameters:
    -----------
    original_bits : array-like
        Original transmitted bits
    decoded_bits : array-like
        Decoded bits (may contain errors)
    
    Returns:
    --------
    ber : float
        Bit Error Rate (0.0 to 1.0)
    num_errors : int
        Number of bit errors
    total_bits : int
        Total number of bits
    
    Examples:
    ---------
    >>> original = np.array([0, 1, 0, 1])
    >>> decoded = np.array([0, 1, 1, 1])  # 1 error
    >>> ber, errors, total = calculate_ber(original, decoded)
    >>> ber
    0.25
    >>> errors
    1
    """
    original_bits = np.array(original_bits)
    decoded_bits = np.array(decoded_bits)
    
    # Flatten if needed
    if original_bits.ndim > 1:
        original_bits = original_bits.flatten()
    if decoded_bits.ndim > 1:
        decoded_bits = decoded_bits.flatten()
    
    # Count errors
    num_errors = np.sum(original_bits != decoded_bits)
    total_bits = len(original_bits)
    
    # Calculate BER
    ber = num_errors / total_bits if total_bits > 0 else 0.0
    
    return ber, num_errors, total_bits


def simulate_classical_decoder(num_bits, eb_no_db, rate=4/7, seed=None):
    """
    Simulate transmission and decoding using classical Hamming decoder.
    
    Process:
    1. Generate random data bits
    2. Encode using Hamming(7,4)
    3. Modulate using BPSK
    4. Add AWGN noise
    5. Demodulate (hard decision)
    6. Decode using classical decoder
    7. Calculate BER
    
    Parameters:
    -----------
    num_bits : int
        Number of data bits to simulate (will be rounded to multiple of 4)
    eb_no_db : float
        Eb/N0 in dB
    rate : float, optional
        Code rate. Default is 4/7 for Hamming(7,4)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    ber : float
        Bit Error Rate
    num_errors : int
        Number of bit errors
    total_bits : int
        Total number of data bits
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Round to multiple of 4 (Hamming(7,4) encodes 4 bits at a time)
    num_data_bits = (num_bits // 4) * 4
    if num_data_bits == 0:
        num_data_bits = 4
    
    # Generate random data bits
    data_bits = np.random.randint(0, 2, size=num_data_bits, dtype=np.uint8)
    
    # Encode: 4 bits → 7 bits
    num_codewords = num_data_bits // 4
    codewords = []
    for i in range(num_codewords):
        data_chunk = data_bits[i*4:(i+1)*4]
        codeword = encode(data_chunk)
        codewords.append(codeword)
    codewords = np.array(codewords)  # Shape: (num_codewords, 7)
    
    # Flatten codewords for transmission
    tx_bits = codewords.flatten()  # Shape: (num_codewords * 7,)
    
    # Modulate
    tx_symbols = bpsk_modulate(tx_bits)
    
    # Add AWGN
    rx_symbols = awgn_channel(tx_symbols, eb_no_db, rate, seed=seed)
    
    # Demodulate (hard decision)
    rx_bits = bpsk_demodulate_hard(rx_symbols)
    
    # Reshape for decoding: (num_codewords, 7)
    rx_codewords = rx_bits.reshape(num_codewords, 7)
    
    # Decode using classical decoder
    decoder = ClassicalDecoder()
    decoded_data, _, _ = decoder.decode_batch(rx_codewords)
    
    # Flatten decoded data
    decoded_bits = decoded_data.flatten()
    
    # Calculate BER
    ber, num_errors, total_bits = calculate_ber(data_bits, decoded_bits)
    
    return ber, num_errors, total_bits


def evaluate_classical_decoder(ebno_range, num_bits_per_point=100000, rate=4/7, seed=None):
    """
    Evaluate classical decoder performance across a range of Eb/N0 values.
    
    Parameters:
    -----------
    ebno_range : array-like
        Array of Eb/N0 values in dB
    num_bits_per_point : int, optional
        Number of data bits to simulate per Eb/N0 point. Default is 100000.
    rate : float, optional
        Code rate. Default is 4/7 for Hamming(7,4)
    seed : int, optional
        Random seed for reproducibility (uses sequential seeds for each point)
    
    Returns:
    --------
    ber_results : numpy.ndarray
        BER values for each Eb/N0 point
    error_counts : numpy.ndarray
        Number of errors for each Eb/N0 point
    total_bits : numpy.ndarray
        Total bits simulated for each Eb/N0 point
    """
    ebno_range = np.array(ebno_range)
    num_points = len(ebno_range)
    
    ber_results = np.zeros(num_points)
    error_counts = np.zeros(num_points, dtype=int)
    total_bits_array = np.zeros(num_points, dtype=int)
    
    print(f"Evaluating classical decoder performance...")
    print(f"Eb/N0 range: {ebno_range[0]:.1f} to {ebno_range[-1]:.1f} dB")
    print(f"Bits per point: {num_bits_per_point:,}")
    print()
    
    for i, eb_no_db in enumerate(ebno_range):
        # Use sequential seeds for reproducibility
        point_seed = seed + i if seed is not None else None
        
        ber, num_errors, total = simulate_classical_decoder(
            num_bits_per_point, eb_no_db, rate, seed=point_seed
        )
        
        ber_results[i] = ber
        error_counts[i] = num_errors
        total_bits_array[i] = total
        
        # Progress indicator
        if (i + 1) % max(1, num_points // 10) == 0 or i == num_points - 1:
            print(f"  Eb/N0 = {eb_no_db:5.1f} dB: BER = {ber:.2e}, Errors = {num_errors}/{total}")
    
    print()
    return ber_results, error_counts, total_bits_array


def plot_ber_curve(ebno_range, ber_values, title="BER vs. Eb/N0", 
                   label="Classical Decoder", save_path=None, show_plot=True):
    """
    Plot BER vs. Eb/N0 curve.
    
    Parameters:
    -----------
    ebno_range : array-like
        Eb/N0 values in dB
    ber_values : array-like
        BER values (can be multiple curves)
    title : str, optional
        Plot title
    label : str or list, optional
        Label(s) for the curve(s)
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.
    show_plot : bool, optional
        Whether to display the plot. Default is True.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    # Handle multiple curves
    if isinstance(ber_values, list) or (isinstance(ber_values, np.ndarray) and ber_values.ndim > 1):
        # Multiple curves
        if isinstance(label, str):
            label = [f"{label} {i+1}" for i in range(len(ber_values))]
        for i, ber_curve in enumerate(ber_values):
            plt.semilogy(ebno_range, ber_curve, marker='o', label=label[i] if label else None)
    else:
        # Single curve
        plt.semilogy(ebno_range, ber_values, marker='o', label=label)
    
    plt.xlabel('Eb/N0 (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.ylim(bottom=1e-6, top=1.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def run_baseline_evaluation(ebno_start=-5, ebno_end=10, num_points=16, 
                           num_bits_per_point=100000, seed=42, save_plot=True):
    """
    Run complete baseline evaluation for classical Hamming decoder.
    
    Parameters:
    -----------
    ebno_start : float, optional
        Starting Eb/N0 in dB. Default is -5.
    ebno_end : float, optional
        Ending Eb/N0 in dB. Default is 10.
    num_points : int, optional
        Number of Eb/N0 points. Default is 16.
    num_bits_per_point : int, optional
        Number of data bits per point. Default is 100000.
    seed : int, optional
        Random seed. Default is 42.
    save_plot : bool, optional
        Whether to save the plot. Default is True.
    
    Returns:
    --------
    ebno_range : numpy.ndarray
        Eb/N0 values in dB
    ber_values : numpy.ndarray
        BER values
    error_counts : numpy.ndarray
        Error counts
    total_bits : numpy.ndarray
        Total bits simulated
    """
    print("=" * 60)
    print("Baseline Performance Evaluation: Classical Hamming Decoder")
    print("=" * 60)
    print()
    
    # Generate Eb/N0 range
    ebno_range = generate_ebno_range(ebno_start, ebno_end, num_points=num_points)
    
    # Evaluate decoder
    ber_values, error_counts, total_bits = evaluate_classical_decoder(
        ebno_range, num_bits_per_point, seed=seed
    )
    
    # Plot results
    plot_ber_curve(
        ebno_range, ber_values,
        title="Hamming(7,4) Classical Decoder - BER vs. Eb/N0",
        label="Classical Decoder",
        save_path="data/baseline_ber_curve.png" if save_plot else None,
        show_plot=False
    )
    
    # Print summary
    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Eb/N0 range: {ebno_start} to {ebno_end} dB")
    print(f"Number of points: {num_points}")
    print(f"Total bits simulated: {np.sum(total_bits):,}")
    print(f"Total errors: {np.sum(error_counts):,}")
    print()
    print("BER Results:")
    print("Eb/N0 (dB) | BER      | Errors")
    print("-" * 35)
    for i in range(len(ebno_range)):
        print(f"  {ebno_range[i]:5.1f}   | {ber_values[i]:.2e} | {error_counts[i]:,}")
    print()
    
    return ebno_range, ber_values, error_counts, total_bits

