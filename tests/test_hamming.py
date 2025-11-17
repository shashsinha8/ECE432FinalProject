"""
Unit tests for Hamming(7,4) encoder and decoder.
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

from src.hamming import encode, decode, compute_syndrome, get_all_codewords
from src.classical_decoder import ClassicalDecoder


class TestHammingEncoder:
    """Test cases for Hamming encoder."""
    
    def test_encode_array_input(self):
        """Test encoding with array input."""
        data = np.array([1, 0, 1, 0], dtype=np.uint8)
        codeword = encode(data)
        
        assert codeword.shape == (7,)
        assert codeword.dtype == np.uint8
        # Verify it's a valid codeword (syndrome should be 0)
        assert compute_syndrome(codeword) == 0
    
    def test_encode_integer_input(self):
        """Test encoding with integer input."""
        # Test all 16 possible 4-bit values
        for i in range(16):
            codeword = encode(i)
            assert codeword.shape == (7,)
            assert compute_syndrome(codeword) == 0
    
    def test_encode_all_codewords(self):
        """Test that all 16 codewords are valid."""
        codewords = get_all_codewords()
        assert codewords.shape == (16, 7)
        
        # All codewords should have zero syndrome
        for codeword in codewords:
            assert compute_syndrome(codeword) == 0
    
    def test_encode_invalid_input(self):
        """Test that invalid inputs raise errors."""
        # Too few bits
        with pytest.raises(ValueError):
            encode([1, 0, 1])
        
        # Too many bits
        with pytest.raises(ValueError):
            encode([1, 0, 1, 0, 1])
        
        # Invalid integer
        with pytest.raises(ValueError):
            encode(16)
        
        with pytest.raises(ValueError):
            encode(-1)
    
    def test_encode_decodability(self):
        """Test that encoded words can be decoded back to original data."""
        for i in range(16):
            data = np.array([(i >> j) & 1 for j in range(3, -1, -1)], dtype=np.uint8)
            codeword = encode(data)
            decoded_data, _, _ = decode(codeword)
            np.testing.assert_array_equal(decoded_data, data)


class TestHammingDecoder:
    """Test cases for Hamming decoder."""
    
    def test_decode_no_error(self):
        """Test decoding with no errors."""
        data = np.array([1, 0, 1, 1], dtype=np.uint8)
        codeword = encode(data)
        decoded_data, corrected_word, error_pos = decode(codeword)
        
        np.testing.assert_array_equal(decoded_data, data)
        np.testing.assert_array_equal(corrected_word, codeword)
        assert error_pos == 0
    
    def test_decode_single_bit_errors(self):
        """Test that all single-bit errors are corrected."""
        data = np.array([1, 0, 1, 0], dtype=np.uint8)
        codeword = encode(data)
        
        # Test error in each of the 7 positions
        for error_pos in range(7):
            # Create error
            corrupted = codeword.copy()
            corrupted[error_pos] = 1 - corrupted[error_pos]
            
            # Decode
            decoded_data, corrected_word, detected_error_pos = decode(corrupted)
            
            # Should recover original data
            np.testing.assert_array_equal(decoded_data, data)
            np.testing.assert_array_equal(corrected_word, codeword)
            # Error position detection (1-indexed)
            assert detected_error_pos == error_pos + 1
    
    def test_decode_all_single_errors(self):
        """Test decoding for all possible codewords with all single-bit errors."""
        for i in range(16):
            data = np.array([(i >> j) & 1 for j in range(3, -1, -1)], dtype=np.uint8)
            codeword = encode(data)
            
            # Test all 7 single-bit error positions
            for error_pos in range(7):
                corrupted = codeword.copy()
                corrupted[error_pos] = 1 - corrupted[error_pos]
                
                decoded_data, corrected_word, _ = decode(corrupted)
                np.testing.assert_array_equal(decoded_data, data)
                np.testing.assert_array_equal(corrected_word, codeword)
    
    def test_decode_double_bit_errors(self):
        """Test that double-bit errors are not correctly decoded."""
        data = np.array([1, 0, 1, 0], dtype=np.uint8)
        codeword = encode(data)
        
        # Create double-bit error
        corrupted = codeword.copy()
        corrupted[0] = 1 - corrupted[0]  # Error in position 1
        corrupted[1] = 1 - corrupted[1]  # Error in position 2
        
        decoded_data, corrected_word, error_pos = decode(corrupted)
        
        # Decoder will try to correct, but may not get original data
        # (This is expected - Hamming(7,4) can only correct single errors)
        # The decoder will still produce a result, but it may be incorrect
        assert error_pos > 0  # Will detect an error, but correction may be wrong
    
    def test_decode_invalid_input(self):
        """Test that invalid inputs raise errors."""
        with pytest.raises(ValueError):
            decode([1, 0, 1])  # Too short
        
        with pytest.raises(ValueError):
            decode([1, 0, 1, 0, 1, 0, 1, 0])  # Too long


class TestSyndrome:
    """Test cases for syndrome computation."""
    
    def test_syndrome_no_error(self):
        """Test syndrome for valid codewords (should be 0)."""
        for i in range(16):
            codeword = encode(i)
            syndrome = compute_syndrome(codeword)
            assert syndrome == 0
    
    def test_syndrome_single_errors(self):
        """Test syndrome for single-bit errors."""
        codeword = encode(5)  # 0101
        
        for error_pos in range(7):
            corrupted = codeword.copy()
            corrupted[error_pos] = 1 - corrupted[error_pos]
            syndrome = compute_syndrome(corrupted)
            # Syndrome should be non-zero for errors
            assert syndrome != 0


class TestClassicalDecoder:
    """Test cases for ClassicalDecoder class."""
    
    def test_decode_no_error(self):
        """Test decoding with no errors."""
        decoder = ClassicalDecoder()
        data = np.array([1, 0, 1, 1], dtype=np.uint8)
        codeword = encode(data)
        
        decoded_data, corrected_word, error_corrected = decoder.decode(codeword)
        
        np.testing.assert_array_equal(decoded_data, data)
        np.testing.assert_array_equal(corrected_word, codeword)
        assert error_corrected == False
    
    def test_decode_single_error(self):
        """Test decoding with single-bit error."""
        decoder = ClassicalDecoder()
        data = np.array([1, 0, 1, 0], dtype=np.uint8)
        codeword = encode(data)
        
        # Introduce error
        corrupted = codeword.copy()
        corrupted[2] = 1 - corrupted[2]
        
        decoded_data, corrected_word, error_corrected = decoder.decode(corrupted)
        
        np.testing.assert_array_equal(decoded_data, data)
        np.testing.assert_array_equal(corrected_word, codeword)
        assert error_corrected == True
    
    def test_decode_batch(self):
        """Test batch decoding."""
        decoder = ClassicalDecoder()
        
        # Create batch of codewords
        num_words = 10
        data_batch = []
        codewords = []
        
        for i in range(num_words):
            data = np.array([(i >> j) & 1 for j in range(3, -1, -1)], dtype=np.uint8)
            data_batch.append(data)
            codewords.append(encode(data))
        
        codewords = np.array(codewords)
        
        # Decode batch
        decoded_data, corrected_words, errors_corrected = decoder.decode_batch(codewords)
        
        assert decoded_data.shape == (num_words, 4)
        assert corrected_words.shape == (num_words, 7)
        assert errors_corrected.shape == (num_words,)
        
        # All should decode correctly with no errors
        for i in range(num_words):
            np.testing.assert_array_equal(decoded_data[i], data_batch[i])
            assert errors_corrected[i] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

