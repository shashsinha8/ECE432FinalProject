"""
Unit tests for ML decoder evaluation.
"""

import pytest
import numpy as np
import torch
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

from src.ml_evaluation import simulate_ml_decoder, evaluate_ml_decoder
from src.ml_decoder import MLDecoder, DirectMappingDecoder
from src.channel import generate_ebno_range


class TestMLSimulation:
    """Test cases for ML decoder simulation."""
    
    def test_simulate_ml_decoder_shape(self):
        """Test that ML simulation produces correct outputs."""
        model = DirectMappingDecoder(input_size=7, hidden_sizes=[16, 8], output_size=4)
        ml_decoder = MLDecoder(model, device='cpu')
        
        ber, errors, total = simulate_ml_decoder(
            num_bits=100, eb_no_db=5.0, ml_decoder=ml_decoder, 
            approach='direct', seed=42
        )
        
        assert isinstance(ber, (float, np.floating))
        assert isinstance(errors, (int, np.integer))
        assert isinstance(total, (int, np.integer))
        assert 0.0 <= ber <= 1.0
        assert errors >= 0
        assert total > 0
    
    def test_simulate_ml_decoder_reproducibility(self):
        """Test that ML simulation is reproducible."""
        model = DirectMappingDecoder(input_size=7, hidden_sizes=[16, 8], output_size=4)
        ml_decoder = MLDecoder(model, device='cpu')
        
        ber1, errors1, total1 = simulate_ml_decoder(
            num_bits=1000, eb_no_db=5.0, ml_decoder=ml_decoder,
            approach='direct', seed=123
        )
        ber2, errors2, total2 = simulate_ml_decoder(
            num_bits=1000, eb_no_db=5.0, ml_decoder=ml_decoder,
            approach='direct', seed=123
        )
        
        assert ber1 == ber2
        assert errors1 == errors2
        assert total1 == total2
    
    def test_simulate_ml_decoder_direct(self):
        """Test direct mapping approach."""
        model = DirectMappingDecoder(input_size=7, hidden_sizes=[16, 8], output_size=4)
        ml_decoder = MLDecoder(model, device='cpu')
        
        ber, errors, total = simulate_ml_decoder(
            num_bits=100, eb_no_db=10.0, ml_decoder=ml_decoder,
            approach='direct', seed=42
        )
        
        assert 0.0 <= ber <= 1.0
        assert total > 0


class TestMLEvaluation:
    """Test cases for ML decoder evaluation."""
    
    def test_evaluate_ml_decoder_shape(self):
        """Test that ML evaluation produces correct output shapes."""
        model = DirectMappingDecoder(input_size=7, hidden_sizes=[16, 8], output_size=4)
        ml_decoder = MLDecoder(model, device='cpu')
        
        ebno_range = generate_ebno_range(0, 5, num_points=3)
        ber, errors, total = evaluate_ml_decoder(
            ebno_range, ml_decoder, approach='direct',
            num_bits_per_point=1000, seed=42
        )
        
        assert len(ber) == len(ebno_range)
        assert len(errors) == len(ebno_range)
        assert len(total) == len(ebno_range)
    
    def test_evaluate_ml_decoder_ber_range(self):
        """Test that BER values are in valid range."""
        model = DirectMappingDecoder(input_size=7, hidden_sizes=[16, 8], output_size=4)
        ml_decoder = MLDecoder(model, device='cpu')
        
        ebno_range = generate_ebno_range(0, 5, num_points=3)
        ber, _, _ = evaluate_ml_decoder(
            ebno_range, ml_decoder, approach='direct',
            num_bits_per_point=1000, seed=42
        )
        
        assert np.all(ber >= 0.0)
        assert np.all(ber <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

