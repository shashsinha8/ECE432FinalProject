"""
Classical Syndrome-Based Decoder for Hamming(7,4) Code

This module provides the classical hard-decision decoder that uses syndrome
decoding to detect and correct single-bit errors.
"""

import numpy as np
from src.hamming import decode, compute_syndrome


class ClassicalDecoder:
    """
    Classical syndrome-based decoder for Hamming(7,4) code.
    
    This decoder performs hard-decision decoding:
    1. Receives 7-bit word (may contain errors)
    2. Computes syndrome
    3. Corrects single-bit error based on syndrome
    4. Extracts 4 data bits from corrected codeword
    """
    
    def __init__(self):
        """Initialize the classical decoder."""
        pass
    
    def decode(self, received_word):
        """
        Decode a received word using classical syndrome decoding.
        
        Parameters:
        -----------
        received_word : array-like of shape (7,)
            Received 7-bit word (hard decisions: 0 or 1)
        
        Returns:
        --------
        data_bits : numpy.ndarray of shape (4,)
            Decoded 4 data bits [d1, d2, d3, d4]
        corrected_word : numpy.ndarray of shape (7,)
            Corrected codeword
        error_corrected : bool
            True if an error was detected and corrected
        """
        # Convert to numpy array and ensure binary
        received_word = np.array(received_word, dtype=np.uint8)
        
        # Ensure binary values (0 or 1)
        received_word = (received_word > 0).astype(np.uint8)
        
        # Use the Hamming decoder
        data_bits, corrected_word, error_position = decode(received_word)
        
        error_corrected = (error_position > 0)
        
        return data_bits, corrected_word, error_corrected
    
    def decode_batch(self, received_words):
        """
        Decode a batch of received words.
        
        Parameters:
        -----------
        received_words : array-like of shape (N, 7)
            Batch of N received 7-bit words
        
        Returns:
        --------
        data_bits : numpy.ndarray of shape (N, 4)
            Decoded data bits for each word
        corrected_words : numpy.ndarray of shape (N, 7)
            Corrected codewords
        errors_corrected : numpy.ndarray of shape (N,)
            Boolean array indicating which words had errors corrected
        """
        received_words = np.array(received_words)
        if len(received_words.shape) == 1:
            received_words = received_words.reshape(1, -1)
        
        N = received_words.shape[0]
        data_bits = np.zeros((N, 4), dtype=np.uint8)
        corrected_words = np.zeros((N, 7), dtype=np.uint8)
        errors_corrected = np.zeros(N, dtype=bool)
        
        for i in range(N):
            data_bits[i], corrected_words[i], errors_corrected[i] = self.decode(received_words[i])
        
        return data_bits, corrected_words, errors_corrected

