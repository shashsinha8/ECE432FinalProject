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


class DeepDirectMappingDecoder(nn.Module):
    """
    Deeper version of DirectMappingDecoder with more layers.
    
    Architecture: 7 → 128 → 64 → 32 → 16 → 4 (deeper than standard)
    Provides more capacity for learning complex error patterns.
    """
    
    def __init__(self, input_size=7, output_size=4, use_soft_input=False):
        """
        Initialize deep direct mapping decoder.
        
        Parameters:
        -----------
        input_size : int
            Size of input (7 for Hamming codeword)
        output_size : int
            Size of output (4 for data bits)
        use_soft_input : bool
            If True, expects soft values (LLRs). If False, expects hard bits.
        """
        super(DeepDirectMappingDecoder, self).__init__()
        
        self.use_soft_input = use_soft_input
        
        # Deeper architecture: 7 → 128 → 64 → 32 → 16 → 4
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 16)
        self.output = nn.Linear(16, output_size)
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass through deep network."""
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.relu(self.layer4(x))
        x = self.dropout(x)
        x = self.sigmoid(self.output(x))
        return x


class WideDirectMappingDecoder(nn.Module):
    """
    Wider version of DirectMappingDecoder with more neurons per layer.
    
    Architecture: 7 → 256 → 128 → 4 (wider than standard)
    Provides more capacity in each layer for learning complex patterns.
    """
    
    def __init__(self, input_size=7, output_size=4, use_soft_input=False):
        """
        Initialize wide direct mapping decoder.
        
        Parameters:
        -----------
        input_size : int
            Size of input (7 for Hamming codeword)
        output_size : int
            Size of output (4 for data bits)
        use_soft_input : bool
            If True, expects soft values (LLRs). If False, expects hard bits.
        """
        super(WideDirectMappingDecoder, self).__init__()
        
        self.use_soft_input = use_soft_input
        
        # Wider architecture: 7 → 256 → 128 → 4
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, output_size)
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass through wide network."""
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.output(x))
        return x


class ResidualDirectMappingDecoder(nn.Module):
    """
    DirectMappingDecoder with residual (skip) connections.
    
    Architecture: 7 → 128 → 64 → 32 → 4 with residual connections
    Residual connections help with gradient flow and learning identity mappings.
    """
    
    def __init__(self, input_size=7, output_size=4, use_soft_input=False):
        """
        Initialize residual direct mapping decoder.
        
        Parameters:
        -----------
        input_size : int
            Size of input (7 for Hamming codeword)
        output_size : int
            Size of output (4 for data bits)
        use_soft_input : bool
            If True, expects soft values (LLRs). If False, expects hard bits.
        """
        super(ResidualDirectMappingDecoder, self).__init__()
        
        self.use_soft_input = use_soft_input
        
        # Main path: 7 → 128 → 64 → 32 → 4
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, output_size)
        
        # Residual connections (projection layers when dimensions don't match)
        self.residual1 = nn.Linear(input_size, 128)  # 7 → 128
        self.residual2 = nn.Linear(128, 64)  # 128 → 64
        self.residual3 = nn.Linear(64, 32)  # 64 → 32
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass with residual connections."""
        # First residual block
        identity = self.residual1(x)
        out = self.relu(self.layer1(x))
        out = out + identity  # Residual connection
        out = self.dropout(out)
        
        # Second residual block
        identity = self.residual2(out)
        out = self.relu(self.layer2(out))
        out = out + identity  # Residual connection
        out = self.dropout(out)
        
        # Third residual block
        identity = self.residual3(out)
        out = self.relu(self.layer3(out))
        out = out + identity  # Residual connection
        out = self.dropout(out)
        
        # Output layer
        out = self.sigmoid(self.output(out))
        return out


def generate_training_data(num_samples, ebno_range, approach='direct', 
                          rate=4/7, seed=None, use_soft_input=False,
                          weighted_sampling=False, focus_ebno_range=None):
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
    use_soft_input : bool
        If True, use soft-decision inputs (LLRs). If False, use hard-decision bits.
    weighted_sampling : bool
        If True, use weighted sampling focusing on error-prone regions
    focus_ebno_range : tuple, optional
        (min, max) Eb/N0 range to focus on for weighted sampling.
        Default: (0.0, 5.0) - the error-prone region
    
    Returns:
    --------
    inputs : numpy.ndarray
        Input features (shape: num_samples x 7)
        If use_soft_input=True: LLR values (continuous)
        If use_soft_input=False: Hard bits (0 or 1)
    targets : numpy.ndarray
        Target labels (shape: num_samples x 4)
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    from src.hamming import encode
    from src.classical_decoder import ClassicalDecoder
    from src.channel import (
        bpsk_modulate, 
        bpsk_demodulate_hard, 
        bpsk_demodulate_soft,
        awgn_channel,
        ebno_to_noise_variance
    )
    
    inputs = []
    targets = []
    
    decoder = ClassicalDecoder()
    num_codewords = (num_samples // 4) * 4  # Round to multiple of 4
    if num_codewords == 0:
        num_codewords = 4
    
    ebno_min, ebno_max = ebno_range
    
    # Set up weighted sampling if requested
    if weighted_sampling:
        if focus_ebno_range is None:
            focus_ebno_range = (0.0, 5.0)  # Default: focus on error-prone region
        focus_min, focus_max = focus_ebno_range
        # Use 70% samples from focus region, 30% from full range
        focus_weight = 0.7
    else:
        focus_weight = 0.0
    
    for i in range(num_codewords // 4):
        # Generate random data bits
        data_bits = np.random.randint(0, 2, size=4, dtype=np.uint8)
        
        # Encode
        codeword = encode(data_bits)
        
        # Sample Eb/N0 with weighted distribution if requested
        if weighted_sampling and np.random.random() < focus_weight:
            # Sample from focus region (error-prone)
            eb_no_db = np.random.uniform(focus_min, focus_max)
        else:
            # Sample from full range
            eb_no_db = np.random.uniform(ebno_min, ebno_max)
        
        # Modulate
        symbols = bpsk_modulate(codeword)
        
        # Add AWGN
        noisy_symbols = awgn_channel(symbols, eb_no_db, rate, seed=seed + i if seed else None)
        
        if use_soft_input:
            # Use soft-decision (LLRs)
            # Calculate noise variance for proper LLR calculation
            noise_variance, _ = ebno_to_noise_variance(eb_no_db, rate)
            # LLR = 2 * symbol / sigma^2
            llrs = 2 * noisy_symbols / noise_variance
            
            # Normalize LLRs to reasonable range (clip to [-10, 10])
            llrs = np.clip(llrs, -10.0, 10.0)
            
            if approach == 'direct':
                # Direct mapping: input = LLRs, output = original data bits
                inputs.append(llrs.astype(np.float32))
                targets.append(data_bits.astype(np.float32))
            elif approach == 'post':
                # Post-processing: first decode with classical, then use LLRs
                rx_bits = bpsk_demodulate_hard(noisy_symbols)
                _, corrected_word, _ = decoder.decode(rx_bits)
                # For post-processing with soft, we could use LLRs of corrected word
                # But classical decoder gives hard bits, so we'll use those
                inputs.append(corrected_word.astype(np.float32))
                targets.append(data_bits.astype(np.float32))
            else:
                raise ValueError(f"Unknown approach: {approach}")
        else:
            # Use hard-decision bits
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


class CodewordLoss(nn.Module):
    """
    Codeword-level loss function that considers Hamming code structure.
    
    Combines bit-level BCE loss with codeword-level penalty.
    Penalizes incorrect codewords more than individual bit errors.
    """
    
    def __init__(self, bit_weight=0.5, codeword_weight=0.5):
        """
        Initialize codeword loss.
        
        Parameters:
        -----------
        bit_weight : float
            Weight for bit-level BCE loss (default: 0.5)
        codeword_weight : float
            Weight for codeword-level penalty (default: 0.5)
        """
        super(CodewordLoss, self).__init__()
        self.bit_weight = bit_weight
        self.codeword_weight = codeword_weight
        self.bce_loss = nn.BCELoss(reduction='none')
    
    def forward(self, outputs, targets):
        """
        Calculate codeword-level loss.
        
        Parameters:
        -----------
        outputs : torch.Tensor
            Model outputs (batch_size, 4) - probabilities
        targets : torch.Tensor
            Target labels (batch_size, 4) - binary values
        
        Returns:
        --------
        loss : torch.Tensor
            Scalar loss value
        """
        # Bit-level BCE loss
        bit_losses = self.bce_loss(outputs, targets)  # (batch_size, 4)
        bit_loss = bit_losses.mean()  # Average over all bits
        
        # Codeword-level penalty: extra penalty if entire codeword is wrong
        # Convert probabilities to binary predictions
        predicted = (outputs > 0.5).float()
        # Check if entire codeword matches
        codeword_correct = (predicted == targets).all(dim=1).float()  # (batch_size,)
        # Penalty: higher loss for incorrect codewords
        codeword_penalty = (1.0 - codeword_correct).mean()
        
        # Combined loss
        total_loss = self.bit_weight * bit_loss + self.codeword_weight * codeword_penalty
        
        return total_loss


def augment_training_data(inputs, targets, noise_level=0.1, augmentation_ratio=0.1, seed=None):
    """
    Augment training data by adding controlled noise.
    
    Parameters:
    -----------
    inputs : numpy.ndarray
        Input features (N, 7)
    targets : numpy.ndarray
        Target labels (N, 4)
    noise_level : float
        Standard deviation of noise to add (default: 0.1)
    augmentation_ratio : float
        Fraction of data to augment (default: 0.1 = 10%)
    seed : int, optional
        Random seed
    
    Returns:
    --------
    augmented_inputs : numpy.ndarray
        Augmented inputs (original + augmented)
    augmented_targets : numpy.ndarray
        Augmented targets (original + augmented)
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_augment = int(len(inputs) * augmentation_ratio)
    if num_augment == 0:
        return inputs, targets
    
    # Select random samples to augment
    indices = np.random.choice(len(inputs), num_augment, replace=False)
    
    augmented_inputs = []
    augmented_targets = []
    
    for idx in indices:
        # Add small noise to inputs
        noise = np.random.normal(0, noise_level, size=inputs[idx].shape)
        augmented_input = inputs[idx] + noise
        
        # Clip to valid range (for LLRs: [-10, 10], for bits: [0, 1])
        if inputs[idx].min() < 0:  # LLRs
            augmented_input = np.clip(augmented_input, -10.0, 10.0)
        else:  # Hard bits
            augmented_input = np.clip(augmented_input, 0.0, 1.0)
        
        augmented_inputs.append(augmented_input)
        augmented_targets.append(targets[idx])
    
    # Combine original and augmented
    all_inputs = np.vstack([inputs, np.array(augmented_inputs)])
    all_targets = np.vstack([targets, np.array(augmented_targets)])
    
    return all_inputs, all_targets


def train_model(model, train_loader, val_loader, num_epochs=50, 
                learning_rate=0.001, device='cpu', verbose=True, use_codeword_loss=False):
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
    use_codeword_loss : bool
        If True, use codeword-level loss instead of bit-level BCE
    
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
    if use_codeword_loss:
        criterion = CodewordLoss(bit_weight=0.5, codeword_weight=0.5)
    else:
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
    Supports both hard-decision and soft-decision (LLR) inputs.
    """
    
    def __init__(self, model, device='cpu', use_soft_input=False):
        """
        Initialize ML decoder.
        
        Parameters:
        -----------
        model : nn.Module
            Trained neural network model
        device : str
            Device to run inference on
        use_soft_input : bool
            If True, expects soft inputs (LLRs). If False, expects hard bits.
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.use_soft_input = use_soft_input
    
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

