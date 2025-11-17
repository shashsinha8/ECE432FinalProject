"""
Unit tests for ML decoder.
"""

import pytest
import numpy as np
import torch
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

from src.ml_decoder import (
    DirectMappingDecoder,
    PostProcessingDecoder,
    HammingDataset,
    MLDecoder,
    generate_training_data
)


class TestMLModels:
    """Test cases for ML model architectures."""
    
    def test_direct_mapping_decoder_forward(self):
        """Test DirectMappingDecoder forward pass."""
        model = DirectMappingDecoder(input_size=7, hidden_sizes=[16, 8], output_size=4)
        model.eval()
        
        # Test with batch
        inputs = torch.randn(10, 7)
        outputs = model(inputs)
        
        assert outputs.shape == (10, 4)
        assert torch.all(outputs >= 0.0) and torch.all(outputs <= 1.0)  # Sigmoid output
    
    def test_post_processing_decoder_forward(self):
        """Test PostProcessingDecoder forward pass."""
        model = PostProcessingDecoder(input_size=7, hidden_sizes=[16, 8], output_size=4)
        model.eval()
        
        inputs = torch.randn(10, 7)
        outputs = model(inputs)
        
        assert outputs.shape == (10, 4)
        assert torch.all(outputs >= 0.0) and torch.all(outputs <= 1.0)
    
    def test_direct_mapping_decoder_single_sample(self):
        """Test DirectMappingDecoder with single sample."""
        model = DirectMappingDecoder()
        model.eval()
        
        inputs = torch.randn(1, 7)
        outputs = model(inputs)
        
        assert outputs.shape == (1, 4)


class TestDataset:
    """Test cases for HammingDataset."""
    
    def test_dataset_length(self):
        """Test dataset length."""
        inputs = np.random.rand(100, 7)
        targets = np.random.rand(100, 4)
        dataset = HammingDataset(inputs, targets)
        
        assert len(dataset) == 100
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        inputs = np.random.rand(10, 7)
        targets = np.random.rand(10, 4)
        dataset = HammingDataset(inputs, targets)
        
        item_input, item_target = dataset[0]
        
        assert item_input.shape == (7,)
        assert item_target.shape == (4,)
        assert isinstance(item_input, torch.Tensor)
        assert isinstance(item_target, torch.Tensor)


class TestDataGeneration:
    """Test cases for training data generation."""
    
    def test_generate_training_data_direct(self):
        """Test direct mapping data generation."""
        inputs, targets = generate_training_data(
            num_samples=100, ebno_range=(0, 5), approach='direct', seed=42
        )
        
        assert inputs.shape[0] == targets.shape[0]
        assert inputs.shape[1] == 7
        assert targets.shape[1] == 4
        assert np.all((inputs == 0) | (inputs == 1))  # Binary inputs
    
    def test_generate_training_data_post(self):
        """Test post-processing data generation."""
        inputs, targets = generate_training_data(
            num_samples=100, ebno_range=(0, 5), approach='post', seed=42
        )
        
        assert inputs.shape[0] == targets.shape[0]
        assert inputs.shape[1] == 7
        assert targets.shape[1] == 4
        assert np.all((inputs == 0) | (inputs == 1))  # Binary inputs
    
    def test_generate_training_data_reproducibility(self):
        """Test that data generation is reproducible."""
        inputs1, targets1 = generate_training_data(
            num_samples=50, ebno_range=(0, 5), approach='direct', seed=42
        )
        inputs2, targets2 = generate_training_data(
            num_samples=50, ebno_range=(0, 5), approach='direct', seed=42
        )
        
        np.testing.assert_array_equal(inputs1, inputs2)
        np.testing.assert_array_equal(targets1, targets2)


class TestMLDecoder:
    """Test cases for MLDecoder wrapper."""
    
    def test_ml_decoder_single_sample(self):
        """Test MLDecoder with single sample."""
        model = DirectMappingDecoder()
        decoder = MLDecoder(model)
        
        received_bits = np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        decoded = decoder.decode(received_bits)
        
        assert decoded.shape == (4,)
        assert np.all((decoded == 0) | (decoded == 1))
    
    def test_ml_decoder_batch(self):
        """Test MLDecoder with batch."""
        model = DirectMappingDecoder()
        decoder = MLDecoder(model)
        
        received_bits = np.random.randint(0, 2, size=(10, 7), dtype=np.uint8)
        decoded = decoder.decode(received_bits)
        
        assert decoded.shape == (10, 4)
        assert np.all((decoded == 0) | (decoded == 1))
    
    def test_ml_decoder_decode_batch(self):
        """Test decode_batch method."""
        model = DirectMappingDecoder()
        decoder = MLDecoder(model)
        
        received_bits = np.random.randint(0, 2, size=(10, 7), dtype=np.uint8)
        decoded = decoder.decode_batch(received_bits)
        
        assert decoded.shape == (10, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

