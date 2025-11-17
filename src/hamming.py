"""
Hamming(7,4) Code Encoder and Decoder

The Hamming(7,4) code encodes 4 data bits into 7 codeword bits with the ability
to detect and correct single-bit errors.

Codeword structure: [p1, p2, d1, p3, d2, d3, d4]
where p1, p2, p3 are parity bits and d1, d2, d3, d4 are data bits.
"""

import numpy as np


# Generator matrix for Hamming(7,4)
# Codeword structure: [p1, p2, d1, p3, d2, d3, d4]
# Parity equations:
#   p1 = d1 ⊕ d2 ⊕ d4
#   p2 = d1 ⊕ d3 ⊕ d4
#   p3 = d2 ⊕ d3 ⊕ d4
# Each row corresponds to one data bit [d1, d2, d3, d4]
# Columns are [p1, p2, d1, p3, d2, d3, d4]
G = np.array([
    [1, 1, 1, 0, 0, 0, 0],  # d1: p1=1, p2=1, d1=1
    [1, 0, 0, 1, 1, 0, 0],  # d2: p1=1, p3=1, d2=1
    [0, 1, 0, 1, 0, 1, 0],  # d3: p2=1, p3=1, d3=1
    [1, 1, 0, 1, 0, 0, 1]   # d4: p1=1, p2=1, p3=1, d4=1
], dtype=np.uint8)

# Parity-check matrix for Hamming(7,4)
# Codeword structure: [p1, p2, d1, p3, d2, d3, d4]
# Each row checks one parity equation
# Row 0: p1 ⊕ d1 ⊕ d2 ⊕ d4 = 0 (checks positions 0, 2, 4, 6)
# Row 1: p2 ⊕ d1 ⊕ d3 ⊕ d4 = 0 (checks positions 1, 2, 5, 6)
# Row 2: p3 ⊕ d2 ⊕ d3 ⊕ d4 = 0 (checks positions 3, 4, 5, 6)
H = np.array([
    [1, 0, 1, 0, 1, 0, 1],  # p1 check: p1, d1, d2, d4
    [0, 1, 1, 0, 0, 1, 1],  # p2 check: p2, d1, d3, d4
    [0, 0, 0, 1, 1, 1, 1]   # p3 check: p3, d2, d3, d4
], dtype=np.uint8)

# Syndrome to error position mapping
# Syndrome is a 3-bit value indicating which parity checks failed
# Codeword positions: [p1, p2, d1, p3, d2, d3, d4] = [0, 1, 2, 3, 4, 5, 6]
# Syndrome values map to error positions (1-indexed, 0 = no error)
SYNDROME_TO_POSITION = {
    0b000: 0,  # No error
    0b001: 4,  # Error in position 4 (p3, 0-indexed position 3)
    0b010: 2,  # Error in position 2 (p2, 0-indexed position 1)
    0b011: 6,  # Error in position 6 (d3, 0-indexed position 5)
    0b100: 1,  # Error in position 1 (p1, 0-indexed position 0)
    0b101: 5,  # Error in position 5 (d2, 0-indexed position 4)
    0b110: 3,  # Error in position 3 (d1, 0-indexed position 2)
    0b111: 7,  # Error in position 7 (d4, 0-indexed position 6)
}


def encode(data_bits):
    """
    Encode 4 data bits into a 7-bit Hamming codeword.
    
    Parameters:
    -----------
    data_bits : array-like of shape (4,) or int
        If array-like: 4 data bits [d1, d2, d3, d4]
        If int: integer representation (0-15) of 4-bit data
    
    Returns:
    --------
    codeword : numpy.ndarray of shape (7,)
        Encoded 7-bit codeword [p1, p2, d1, p3, d2, d3, d4]
    
    Examples:
    ---------
    >>> encode([1, 0, 1, 0])
    array([1, 1, 1, 0, 0, 1, 0], dtype=uint8)
    >>> encode(10)  # 10 = 1010 in binary
    array([1, 1, 1, 0, 0, 1, 0], dtype=uint8)
    """
    # Convert integer input to bit array
    if isinstance(data_bits, (int, np.integer)):
        if data_bits < 0 or data_bits > 15:
            raise ValueError("Data bits must be in range 0-15")
        data_bits = np.array([(data_bits >> i) & 1 for i in range(3, -1, -1)], dtype=np.uint8)
    else:
        data_bits = np.array(data_bits, dtype=np.uint8)
    
    if data_bits.shape != (4,):
        raise ValueError(f"Expected 4 data bits, got {data_bits.shape}")
    
    # Encode using generator matrix: codeword = data_bits @ G (mod 2)
    codeword = np.mod(np.dot(data_bits, G), 2).astype(np.uint8)
    
    return codeword


def compute_syndrome(received_word):
    """
    Compute the syndrome of a received 7-bit word.
    
    Parameters:
    -----------
    received_word : array-like of shape (7,)
        Received 7-bit word (may contain errors)
    
    Returns:
    --------
    syndrome : int
        3-bit syndrome value (0-7)
        - 0: No error detected
        - Non-zero: Indicates error position
    """
    received_word = np.array(received_word, dtype=np.uint8)
    if received_word.shape != (7,):
        raise ValueError(f"Expected 7-bit word, got {received_word.shape}")
    
    # Syndrome = H @ received_word (mod 2)
    syndrome_bits = np.mod(np.dot(H, received_word), 2)
    
    # Convert 3-bit syndrome to integer
    syndrome = int(syndrome_bits[0] * 4 + syndrome_bits[1] * 2 + syndrome_bits[2])
    
    return syndrome


def decode(received_word):
    """
    Decode a received 7-bit word to recover the original 4 data bits.
    Corrects single-bit errors using syndrome decoding.
    
    Parameters:
    -----------
    received_word : array-like of shape (7,)
        Received 7-bit word (may contain errors)
    
    Returns:
    --------
    data_bits : numpy.ndarray of shape (4,)
        Decoded 4 data bits [d1, d2, d3, d4]
    corrected_word : numpy.ndarray of shape (7,)
        Corrected codeword (if error was corrected)
    error_position : int
        Position of corrected error (0 = no error, 1-7 = error position)
    """
    received_word = np.array(received_word, dtype=np.uint8)
    if received_word.shape != (7,):
        raise ValueError(f"Expected 7-bit word, got {received_word.shape}")
    
    # Compute syndrome
    syndrome = compute_syndrome(received_word)
    
    # Get error position from syndrome
    error_position = SYNDROME_TO_POSITION[syndrome]
    
    # Correct error if present
    corrected_word = received_word.copy()
    if error_position > 0:
        # Flip the bit at error position (1-indexed, convert to 0-indexed)
        corrected_word[error_position - 1] = 1 - corrected_word[error_position - 1]
    
    # Extract data bits from corrected codeword
    # Codeword structure: [p1, p2, d1, p3, d2, d3, d4]
    # Data bits are at positions: 2, 4, 5, 6 (0-indexed)
    data_bits = np.array([
        corrected_word[2],  # d1
        corrected_word[4],  # d2
        corrected_word[5],  # d3
        corrected_word[6]   # d4
    ], dtype=np.uint8)
    
    return data_bits, corrected_word, error_position


def get_all_codewords():
    """
    Generate all 16 possible Hamming(7,4) codewords.
    
    Returns:
    --------
    codewords : numpy.ndarray of shape (16, 7)
        All 16 codewords, one per row
    """
    codewords = []
    for i in range(16):
        codeword = encode(i)
        codewords.append(codeword)
    return np.array(codewords)

