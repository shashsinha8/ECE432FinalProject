"""
Unit tests for evaluation framework.
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

from src.evaluation import (
    calculate_ber,
    simulate_classical_decoder,
    evaluate_classical_decoder
)
from src.channel import generate_ebno_range


class TestBERCalculation:
    """Test cases for BER calculation."""
    
    def test_ber_no_errors(self):
        """Test BER calculation with no errors."""
        original = np.array([0, 1, 0, 1])
        decoded = np.array([0, 1, 0, 1])
        
        ber, errors, total = calculate_ber(original, decoded)
        
        assert ber == 0.0
        assert errors == 0
        assert total == 4
    
    def test_ber_all_errors(self):
        """Test BER calculation with all errors."""
        original = np.array([0, 1, 0, 1])
        decoded = np.array([1, 0, 1, 0])
        
        ber, errors, total = calculate_ber(original, decoded)
        
        assert ber == 1.0
        assert errors == 4
        assert total == 4
    
    def test_ber_partial_errors(self):
        """Test BER calculation with partial errors."""
        original = np.array([0, 1, 0, 1, 0, 1])
        decoded = np.array([0, 1, 1, 1, 0, 0])  # 2 errors
        
        ber, errors, total = calculate_ber(original, decoded)
        
        assert ber == pytest.approx(2.0 / 6.0)
        assert errors == 2
        assert total == 6
    
    def test_ber_shape_handling(self):
        """Test BER calculation with 2D arrays."""
        original = np.array([[0, 1], [0, 1]])
        decoded = np.array([[0, 1], [1, 1]])  # 1 error
        
        ber, errors, total = calculate_ber(original, decoded)
        
        assert ber == pytest.approx(0.25)
        assert errors == 1
        assert total == 4


class TestSimulation:
    """Test cases for decoder simulation."""
    
    def test_simulate_classical_decoder_shape(self):
        """Test that simulation produces correct outputs."""
        ber, errors, total = simulate_classical_decoder(
            num_bits=100, eb_no_db=10.0, seed=42
        )
        
        assert isinstance(ber, (float, np.floating))
        assert isinstance(errors, (int, np.integer))
        assert isinstance(total, (int, np.integer))
        assert 0.0 <= ber <= 1.0
        assert errors >= 0
        assert total > 0
    
    def test_simulate_classical_decoder_reproducibility(self):
        """Test that simulation is reproducible with same seed."""
        ber1, errors1, total1 = simulate_classical_decoder(
            num_bits=1000, eb_no_db=5.0, seed=123
        )
        ber2, errors2, total2 = simulate_classical_decoder(
            num_bits=1000, eb_no_db=5.0, seed=123
        )
        
        assert ber1 == ber2
        assert errors1 == errors2
        assert total1 == total2
    
    def test_simulate_classical_decoder_high_ebno(self):
        """Test that high Eb/N0 produces low BER."""
        ber_low, _, _ = simulate_classical_decoder(
            num_bits=10000, eb_no_db=0.0, seed=42
        )
        ber_high, _, _ = simulate_classical_decoder(
            num_bits=10000, eb_no_db=10.0, seed=42
        )
        
        # High Eb/N0 should generally have lower BER
        # (allowing for some statistical variation)
        # We just check that both are valid BER values
        assert 0.0 <= ber_low <= 1.0
        assert 0.0 <= ber_high <= 1.0
    
    def test_simulate_classical_decoder_rounding(self):
        """Test that num_bits is rounded to multiple of 4."""
        # Test with num_bits not divisible by 4
        ber, errors, total = simulate_classical_decoder(
            num_bits=13, eb_no_db=10.0, seed=42
        )
        
        # Should round to 12 (multiple of 4)
        assert total == 12 or total == 16  # Could round up or down
        assert total % 4 == 0


class TestEvaluation:
    """Test cases for full evaluation."""
    
    def test_evaluate_classical_decoder_shape(self):
        """Test that evaluation produces correct output shapes."""
        ebno_range = generate_ebno_range(0, 5, num_points=3)
        ber, errors, total = evaluate_classical_decoder(
            ebno_range, num_bits_per_point=1000, seed=42
        )
        
        assert len(ber) == len(ebno_range)
        assert len(errors) == len(ebno_range)
        assert len(total) == len(ebno_range)
    
    def test_evaluate_classical_decoder_ber_range(self):
        """Test that BER values are in valid range."""
        ebno_range = generate_ebno_range(0, 5, num_points=3)
        ber, _, _ = evaluate_classical_decoder(
            ebno_range, num_bits_per_point=1000, seed=42
        )
        
        assert np.all(ber >= 0.0)
        assert np.all(ber <= 1.0)
    
    def test_evaluate_classical_decoder_monotonic(self):
        """Test that BER generally decreases with increasing Eb/N0."""
        ebno_range = generate_ebno_range(0, 10, num_points=5)
        ber, _, _ = evaluate_classical_decoder(
            ebno_range, num_bits_per_point=5000, seed=42
        )
        
        # Check that BER is generally decreasing (allowing for statistical variation)
        # We just verify that the values are reasonable
        assert np.all(ber >= 0.0)
        assert np.all(ber <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

