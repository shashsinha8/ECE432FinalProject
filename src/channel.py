"""
Channel Simulation Module

This module implements BPSK (Binary Phase Shift Keying) modulation,
AWGN (Additive White Gaussian Noise) channel, and demodulation functions.
"""

import numpy as np


def bpsk_modulate(bits):
    """
    Modulate binary bits to BPSK symbols.
    
    BPSK mapping:
    - 0 → +1
    - 1 → -1
    
    Parameters:
    -----------
    bits : array-like
        Binary bits (0 or 1) to modulate
    
    Returns:
    --------
    symbols : numpy.ndarray
        BPSK symbols (+1 or -1)
    
    Examples:
    ---------
    >>> bpsk_modulate([0, 1, 0, 1])
    array([ 1., -1.,  1., -1.])
    """
    bits = np.array(bits, dtype=np.float64)
    # Map 0 → +1, 1 → -1
    symbols = 1 - 2 * bits
    return symbols


def bpsk_demodulate_hard(symbols):
    """
    Demodulate BPSK symbols to binary bits using hard decision.
    
    Hard decision: threshold at 0
    - symbol > 0 → 0
    - symbol ≤ 0 → 1
    
    Parameters:
    -----------
    symbols : array-like
        Received BPSK symbols (may be noisy)
    
    Returns:
    --------
    bits : numpy.ndarray
        Demodulated binary bits (0 or 1)
    
    Examples:
    ---------
    >>> bpsk_demodulate_hard([1.2, -0.8, 0.5, -1.1])
    array([0, 1, 0, 1])
    """
    symbols = np.array(symbols, dtype=np.float64)
    # Hard decision: positive → 0, negative/zero → 1
    bits = (symbols <= 0).astype(np.uint8)
    return bits


def bpsk_demodulate_soft(symbols):
    """
    Demodulate BPSK symbols to log-likelihood ratios (LLRs) for soft decoding.
    
    LLR = log(P(bit=0|symbol) / P(bit=1|symbol))
    For AWGN channel with unit energy: LLR = 2 * symbol / sigma^2
    
    For now, assuming unit noise variance, so LLR = 2 * symbol
    
    Parameters:
    -----------
    symbols : array-like
        Received BPSK symbols (may be noisy)
    
    Returns:
    --------
    llrs : numpy.ndarray
        Log-likelihood ratios (positive favors bit=0, negative favors bit=1)
    
    Examples:
    ---------
    >>> bpsk_demodulate_soft([1.2, -0.8])
    array([ 2.4, -1.6])
    """
    symbols = np.array(symbols, dtype=np.float64)
    # LLR = 2 * symbol / sigma^2
    # For unit variance: LLR = 2 * symbol
    # Note: In practice, this should use the actual noise variance
    llrs = 2 * symbols
    return llrs


def awgn_channel(symbols, eb_no_db, rate=4/7, seed=None):
    """
    Add AWGN noise to BPSK symbols.
    
    The noise variance is calculated from Eb/N0 (energy per bit to noise
    power spectral density ratio) in dB.
    
    For BPSK with unit energy symbols:
    - Es/N0 = Eb/N0 * rate (where rate = k/n for code rate)
    - sigma^2 = 1 / (2 * Es/N0) = 1 / (2 * 10^(Eb/N0_dB/10) * rate)
    
    Parameters:
    -----------
    symbols : array-like
        Transmitted BPSK symbols
    eb_no_db : float
        Energy per bit to noise power spectral density ratio in dB
    rate : float, optional
        Code rate (k/n). For Hamming(7,4), rate = 4/7. Default is 4/7.
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    noisy_symbols : numpy.ndarray
        Received symbols with AWGN noise added
    
    Examples:
    ---------
    >>> symbols = bpsk_modulate([0, 1, 0, 1])
    >>> noisy = awgn_channel(symbols, eb_no_db=10.0)
    >>> noisy.shape == symbols.shape
    True
    """
    if seed is not None:
        np.random.seed(seed)
    
    symbols = np.array(symbols, dtype=np.float64)
    
    # Convert Eb/N0 from dB to linear scale
    eb_no_linear = 10 ** (eb_no_db / 10.0)
    
    # Calculate Es/N0 (energy per symbol)
    # Es/N0 = Eb/N0 * rate
    es_no_linear = eb_no_linear * rate
    
    # Calculate noise variance
    # For BPSK: sigma^2 = 1 / (2 * Es/N0)
    noise_variance = 1.0 / (2.0 * es_no_linear)
    noise_std = np.sqrt(noise_variance)
    
    # Generate AWGN noise
    noise = np.random.normal(0.0, noise_std, size=symbols.shape)
    
    # Add noise to symbols
    noisy_symbols = symbols + noise
    
    return noisy_symbols


def ebno_to_noise_variance(eb_no_db, rate=4/7):
    """
    Convert Eb/N0 (dB) to noise variance for AWGN channel.
    
    Parameters:
    -----------
    eb_no_db : float
        Energy per bit to noise power spectral density ratio in dB
    rate : float, optional
        Code rate (k/n). For Hamming(7,4), rate = 4/7. Default is 4/7.
    
    Returns:
    --------
    noise_variance : float
        Noise variance (sigma^2)
    noise_std : float
        Noise standard deviation (sigma)
    
    Examples:
    ---------
    >>> var, std = ebno_to_noise_variance(10.0)
    >>> var > 0
    True
    """
    # Convert Eb/N0 from dB to linear scale
    eb_no_linear = 10 ** (eb_no_db / 10.0)
    
    # Calculate Es/N0
    es_no_linear = eb_no_linear * rate
    
    # Calculate noise variance
    noise_variance = 1.0 / (2.0 * es_no_linear)
    noise_std = np.sqrt(noise_variance)
    
    return noise_variance, noise_std


def generate_ebno_range(start_db, end_db, num_points=None, step_db=None):
    """
    Generate a range of Eb/N0 values in dB.
    
    Parameters:
    -----------
    start_db : float
        Starting Eb/N0 value in dB
    end_db : float
        Ending Eb/N0 value in dB
    num_points : int, optional
        Number of points (if specified, creates linear spacing)
    step_db : float, optional
        Step size in dB (if specified, creates uniform spacing)
    
    Returns:
    --------
    ebno_db : numpy.ndarray
        Array of Eb/N0 values in dB
    
    Examples:
    ---------
    >>> ebno = generate_ebno_range(0, 10, num_points=11)
    >>> len(ebno)
    11
    >>> ebno[0], ebno[-1]
    (0.0, 10.0)
    """
    if num_points is not None:
        ebno_db = np.linspace(start_db, end_db, num_points)
    elif step_db is not None:
        ebno_db = np.arange(start_db, end_db + step_db, step_db)
    else:
        # Default: 0.5 dB steps
        ebno_db = np.arange(start_db, end_db + 0.5, 0.5)
    
    return ebno_db


def simulate_transmission(bits, eb_no_db, rate=4/7, seed=None):
    """
    Simulate complete transmission: modulation → AWGN → demodulation.
    
    Parameters:
    -----------
    bits : array-like
        Binary bits to transmit
    eb_no_db : float
        Eb/N0 in dB
    rate : float, optional
        Code rate. Default is 4/7 for Hamming(7,4)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    transmitted_symbols : numpy.ndarray
        BPSK symbols before channel
    received_symbols : numpy.ndarray
        BPSK symbols after AWGN channel
    demodulated_bits_hard : numpy.ndarray
        Hard-decision demodulated bits
    demodulated_bits_soft : numpy.ndarray
        Soft-decision LLRs
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Modulate
    transmitted_symbols = bpsk_modulate(bits)
    
    # Add AWGN
    received_symbols = awgn_channel(transmitted_symbols, eb_no_db, rate)
    
    # Demodulate (hard and soft)
    demodulated_bits_hard = bpsk_demodulate_hard(received_symbols)
    llrs = bpsk_demodulate_soft(received_symbols)
    
    return transmitted_symbols, received_symbols, demodulated_bits_hard, llrs

