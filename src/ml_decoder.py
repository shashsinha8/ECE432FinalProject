"""
Machine Learning-Assisted Decoder for Hamming(7,4) Code

This module implements neural network-based decoders that can either:
1. Post-process classical decoder output to correct remaining errors
2. Directly map noisy received bits to original data bits
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


class HammingDataset(Dataset):
    """
    Dataset for training ML decoder.
    
    Can generate data for two approaches:
    1. Direct mapping: Input = 7 received bits, Output = 4 data bits
    2. Post-processing: Input = 7 bits from classical decoder, Output = 4 corrected data bits
    """
    
    def __init__(self, inputs, targets):
        """
        Initialize dataset.
        
        Parameters:
        -----------
        inputs : numpy.ndarray
            Input features (shape: N x 7)
        targets : numpy.ndarray
            Target labels (shape: N x 4)
        """
        self.inputs = torch.FloatTensor(inputs)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class DirectMappingDecoder(nn.Module):
    """
    Neural network that directly maps 7 received bits to 4 data bits.
    
    Architecture:
    - Input: 7 received bits (hard decisions or soft values)
    - Hidden layers: Fully connected layers
    - Output: 4 data bits (sigmoid activation for binary classification)
    """
    
    def __init__(self, input_size=7, hidden_sizes=[64, 32], output_size=4, use_soft_input=False):
        """
        Initialize direct mapping decoder.
        
        Parameters:
        -----------
        input_size : int
            Size of input (7 for Hamming codeword)
        hidden_sizes : list
            Sizes of hidden layers
        output_size : int
            Size of output (4 for data bits)
        use_soft_input : bool
            If True, expects soft values (LLRs). If False, expects hard bits.
        """
        super(DirectMappingDecoder, self).__init__()
        
        self.use_soft_input = use_soft_input
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())  # Sigmoid for binary classification
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 7)
        
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, 4)
        """
        return self.network(x)


class PostProcessingDecoder(nn.Module):
    """
    Neural network that post-processes classical decoder output.
    
    Takes the 7-bit output from classical decoder and corrects remaining errors.
    Similar architecture to DirectMappingDecoder but trained on different data.
    """
    
    def __init__(self, input_size=7, hidden_sizes=[32, 16], output_size=4):
        """
        Initialize post-processing decoder.
        
        Parameters:
        -----------
        input_size : int
            Size of input (7 for codeword from classical decoder)
        hidden_sizes : list
            Sizes of hidden layers
        output_size : int
            Size of output (4 for data bits)
        """
        super(PostProcessingDecoder, self).__init__()
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 7)
        
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, 4)
        """
        return self.network(x)


def generate_training_data(num_samples, ebno_range, approach='direct', 
                          rate=4/7, seed=None):
    """
    Generate training data for ML decoder.
    
    Parameters:
    -----------
    num_samples : int
        Number of training samples to generate
    ebno_range : tuple or list
        Range of Eb/N0 values (min, max) for training
    approach : str
        'direct' for direct mapping, 'post' for post-processing
    rate : float
        Code rate. Default is 4/7 for Hamming(7,4)
    seed : int, optional
        Random seed
    
    Returns:
    --------
    inputs : numpy.ndarray
        Input features (shape: num_samples x 7)
    targets : numpy.ndarray
        Target labels (shape: num_samples x 4)
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    from src.hamming import encode
    from src.classical_decoder import ClassicalDecoder
    from src.channel import bpsk_modulate, bpsk_demodulate_hard, awgn_channel
    
    inputs = []
    targets = []
    
    decoder = ClassicalDecoder()
    num_codewords = (num_samples // 4) * 4  # Round to multiple of 4
    if num_codewords == 0:
        num_codewords = 4
    
    ebno_min, ebno_max = ebno_range
    
    for i in range(num_codewords // 4):
        # Generate random data bits
        data_bits = np.random.randint(0, 2, size=4, dtype=np.uint8)
        
        # Encode
        codeword = encode(data_bits)
        
        # Random Eb/N0 for this sample
        eb_no_db = np.random.uniform(ebno_min, ebno_max)
        
        # Modulate
        symbols = bpsk_modulate(codeword)
        
        # Add AWGN
        noisy_symbols = awgn_channel(symbols, eb_no_db, rate, seed=seed + i if seed else None)
        
        # Demodulate
        rx_bits = bpsk_demodulate_hard(noisy_symbols)
        
        if approach == 'direct':
            # Direct mapping: input = received bits, output = original data bits
            inputs.append(rx_bits.astype(np.float32))
            targets.append(data_bits.astype(np.float32))
        elif approach == 'post':
            # Post-processing: input = corrected codeword from classical decoder, output = original data bits
            _, corrected_word, _ = decoder.decode(rx_bits)
            inputs.append(corrected_word.astype(np.float32))
            targets.append(data_bits.astype(np.float32))
        else:
            raise ValueError(f"Unknown approach: {approach}")
    
    return np.array(inputs), np.array(targets)


def train_model(model, train_loader, val_loader, num_epochs=50, 
                learning_rate=0.001, device='cpu', verbose=True):
    """
    Train the ML decoder model.
    
    Parameters:
    -----------
    model : nn.Module
        Neural network model
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    num_epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimizer
    device : str
        Device to train on ('cpu' or 'cuda')
    verbose : bool
        Whether to print training progress
    
    Returns:
    --------
    train_losses : list
        Training losses per epoch
    val_losses : list
        Validation losses per epoch
    val_accuracies : list
        Validation accuracies per epoch
    """
    model = model.to(device)
    criterion = nn.BCELoss()  # Binary cross-entropy for binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Calculate accuracy (threshold at 0.5)
                predicted = (outputs > 0.5).float()
                correct += (predicted == targets).all(dim=1).sum().item()
                total += targets.size(0)
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = correct / total if total > 0 else 0.0
        val_accuracies.append(val_accuracy)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_accuracy:.4f}")
    
    return train_losses, val_losses, val_accuracies


def save_model(model, filepath):
    """
    Save trained model to file.
    
    Parameters:
    -----------
    model : nn.Module
        Trained model
    filepath : str
        Path to save model
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device='cpu'):
    """
    Load trained model from file.
    
    Parameters:
    -----------
    model : nn.Module
        Model architecture (must match saved model)
    filepath : str
        Path to saved model
    device : str
        Device to load model on
    
    Returns:
    --------
    model : nn.Module
        Model with loaded weights
    """
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    print(f"Model loaded from {filepath}")
    return model


class MLDecoder:
    """
    Wrapper class for ML-assisted decoder.
    
    Can use either direct mapping or post-processing approach.
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize ML decoder.
        
        Parameters:
        -----------
        model : nn.Module
            Trained neural network model
        device : str
            Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    def decode(self, received_bits):
        """
        Decode received bits using ML model.
        
        Parameters:
        -----------
        received_bits : array-like
            Received 7-bit word(s). Shape: (7,) or (N, 7)
        
        Returns:
        --------
        decoded_bits : numpy.ndarray
            Decoded 4-bit data. Shape: (4,) or (N, 4)
        """
        received_bits = np.array(received_bits, dtype=np.float32)
        
        # Handle single sample
        if received_bits.ndim == 1:
            received_bits = received_bits.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False
        
        # Convert to tensor
        inputs = torch.FloatTensor(received_bits).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(inputs)
            # Threshold at 0.5
            decoded = (outputs > 0.5).float().cpu().numpy()
        
        # Convert back to uint8
        decoded = decoded.astype(np.uint8)
        
        if single_sample:
            return decoded[0]
        return decoded
    
    def decode_batch(self, received_bits):
        """
        Decode a batch of received bits.
        
        Parameters:
        -----------
        received_bits : array-like
            Batch of received 7-bit words. Shape: (N, 7)
        
        Returns:
        --------
        decoded_bits : numpy.ndarray
            Decoded data bits. Shape: (N, 4)
        """
        return self.decode(received_bits)

