"""
ML-Assisted Decoder Evaluation Framework

This module extends the evaluation framework to support ML-assisted decoding
and comparison with classical baseline.
"""

import numpy as np
from src.hamming import encode
from src.ml_decoder import MLDecoder, load_model, DirectMappingDecoder, PostProcessingDecoder
from src.channel import (
    bpsk_modulate,
    bpsk_demodulate_hard,
    awgn_channel,
    generate_ebno_range
)
from src.evaluation import calculate_ber, plot_ber_curve
import torch


def simulate_ml_decoder(num_bits, eb_no_db, ml_decoder, approach='direct',
                       rate=4/7, seed=None):
    """
    Simulate transmission and decoding using ML-assisted decoder.
    
    Parameters:
    -----------
    num_bits : int
        Number of data bits to simulate (rounded to multiple of 4)
    eb_no_db : float
        Eb/N0 in dB
    ml_decoder : MLDecoder
        Trained ML decoder instance
    approach : str
        'direct' for direct mapping, 'post' for post-processing
    rate : float
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
    
    from src.classical_decoder import ClassicalDecoder
    
    # Round to multiple of 4
    num_data_bits = (num_bits // 4) * 4
    if num_data_bits == 0:
        num_data_bits = 4
    
    # Generate random data bits
    data_bits = np.random.randint(0, 2, size=num_data_bits, dtype=np.uint8)
    
    # Encode
    num_codewords = num_data_bits // 4
    codewords = []
    for i in range(num_codewords):
        data_chunk = data_bits[i*4:(i+1)*4]
        codeword = encode(data_chunk)
        codewords.append(codeword)
    codewords = np.array(codewords)
    
    # Flatten for transmission
    tx_bits = codewords.flatten()
    
    # Modulate
    tx_symbols = bpsk_modulate(tx_bits)
    
    # Add AWGN
    rx_symbols = awgn_channel(tx_symbols, eb_no_db, rate, seed=seed)
    
    # Demodulate (hard decision)
    rx_bits = bpsk_demodulate_hard(rx_symbols)
    
    # Reshape for decoding
    rx_codewords = rx_bits.reshape(num_codewords, 7)
    
    # Decode using ML decoder
    if approach == 'direct':
        # Direct mapping: ML decoder takes received bits directly
        decoded_data = ml_decoder.decode_batch(rx_codewords)
    elif approach == 'post':
        # Post-processing: First use classical decoder, then ML
        classical_decoder = ClassicalDecoder()
        classical_output, _, _ = classical_decoder.decode_batch(rx_codewords)
        # ML decoder takes the corrected codeword (need to reconstruct)
        # Actually, for post-processing, we need the corrected codeword
        _, corrected_codewords, _ = classical_decoder.decode_batch(rx_codewords)
        decoded_data = ml_decoder.decode_batch(corrected_codewords)
    else:
        raise ValueError(f"Unknown approach: {approach}")
    
    # Flatten decoded data
    decoded_bits = decoded_data.flatten()
    
    # Calculate BER
    ber, num_errors, total_bits = calculate_ber(data_bits, decoded_bits)
    
    return ber, num_errors, total_bits


def evaluate_ml_decoder(ebno_range, ml_decoder, approach='direct',
                       num_bits_per_point=100000, rate=4/7, seed=None):
    """
    Evaluate ML-assisted decoder performance across a range of Eb/N0 values.
    
    Parameters:
    -----------
    ebno_range : array-like
        Array of Eb/N0 values in dB
    ml_decoder : MLDecoder
        Trained ML decoder instance
    approach : str
        'direct' for direct mapping, 'post' for post-processing
    num_bits_per_point : int
        Number of data bits to simulate per Eb/N0 point
    rate : float
        Code rate. Default is 4/7 for Hamming(7,4)
    seed : int, optional
        Random seed (uses sequential seeds for each point)
    
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
    
    print(f"Evaluating ML-assisted decoder performance ({approach} approach)...")
    print(f"Eb/N0 range: {ebno_range[0]:.1f} to {ebno_range[-1]:.1f} dB")
    print(f"Bits per point: {num_bits_per_point:,}")
    print()
    
    for i, eb_no_db in enumerate(ebno_range):
        point_seed = seed + i if seed is not None else None
        
        ber, num_errors, total = simulate_ml_decoder(
            num_bits_per_point, eb_no_db, ml_decoder, approach, rate, seed=point_seed
        )
        
        ber_results[i] = ber
        error_counts[i] = num_errors
        total_bits_array[i] = total
        
        # Progress indicator
        if (i + 1) % max(1, num_points // 10) == 0 or i == num_points - 1:
            print(f"  Eb/N0 = {eb_no_db:5.1f} dB: BER = {ber:.2e}, Errors = {num_errors}/{total}")
    
    print()
    return ber_results, error_counts, total_bits_array


def compare_decoders(ebno_range, classical_ber, ml_ber, ml_label="ML Decoder",
                    save_path=None, show_plot=True):
    """
    Compare classical and ML-assisted decoder performance.
    
    Parameters:
    -----------
    ebno_range : array-like
        Eb/N0 values in dB
    classical_ber : array-like
        BER values for classical decoder
    ml_ber : array-like
        BER values for ML decoder
    ml_label : str
        Label for ML decoder curve
    save_path : str, optional
        Path to save comparison plot
    show_plot : bool
        Whether to display the plot
    """
    plot_ber_curve(
        ebno_range,
        [classical_ber, ml_ber],
        title="Hamming(7,4) Decoder Comparison: Classical vs. ML-Assisted",
        label=["Classical Decoder", ml_label],
        save_path=save_path,
        show_plot=show_plot
    )


def run_ml_evaluation(model_path, approach='direct', ebno_start=-5, ebno_end=10,
                     num_points=16, num_bits_per_point=100000, seed=42,
                     device='cpu', save_plot=True):
    """
    Run complete ML decoder evaluation.
    
    Parameters:
    -----------
    model_path : str
        Path to trained ML model
    approach : str
        'direct' or 'post'
    ebno_start : float
        Starting Eb/N0 in dB
    ebno_end : float
        Ending Eb/N0 in dB
    num_points : int
        Number of Eb/N0 points
    num_bits_per_point : int
        Bits per point
    seed : int
        Random seed
    device : str
        Device for inference
    save_plot : bool
        Whether to save plot
    
    Returns:
    --------
    ebno_range : numpy.ndarray
        Eb/N0 values
    ml_ber : numpy.ndarray
        ML decoder BER values
    error_counts : numpy.ndarray
        Error counts
    total_bits : numpy.ndarray
        Total bits
    """
    print("=" * 60)
    print(f"ML-Assisted Decoder Evaluation ({approach} approach)")
    print("=" * 60)
    print()
    
    # Load model
    print(f"Loading model from {model_path}...")
    if approach == 'direct':
        model = DirectMappingDecoder(input_size=7, hidden_sizes=[64, 32], output_size=4)
    else:
        model = PostProcessingDecoder(input_size=7, hidden_sizes=[32, 16], output_size=4)
    
    model = load_model(model, model_path, device=device)
    ml_decoder = MLDecoder(model, device=device)
    print()
    
    # Generate Eb/N0 range
    ebno_range = generate_ebno_range(ebno_start, ebno_end, num_points=num_points)
    
    # Evaluate ML decoder
    ml_ber, error_counts, total_bits = evaluate_ml_decoder(
        ebno_range, ml_decoder, approach, num_bits_per_point, seed=seed
    )
    
    # Plot results
    plot_ber_curve(
        ebno_range, ml_ber,
        title=f"Hamming(7,4) ML-Assisted Decoder - BER vs. Eb/N0 ({approach})",
        label=f"ML Decoder ({approach})",
        save_path=f"data/ml_decoder_{approach}_ber_curve.png" if save_plot else None,
        show_plot=False
    )
    
    # Print summary
    print("=" * 60)
    print("ML Decoder Evaluation Summary")
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
        print(f"  {ebno_range[i]:5.1f}   | {ml_ber[i]:.2e} | {error_counts[i]:,}")
    print()
    
    return ebno_range, ml_ber, error_counts, total_bits

