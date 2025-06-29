"""
CNN model for gunshot classification from Mel spectrograms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class GunshotCNN(nn.Module):
    """CNN architecture for binary gunshot classification"""
    
    def __init__(self, n_mels=128, n_time_frames=173, num_classes=1, dropout_rate=0.5):
        """
        Initialize CNN model
        
        Args:
            n_mels: num of mel frequency bins
            n_time_frames: num of time frames in spectrogram
            num_classes: num of output classes (1 for binary)
            dropout_rate: dropout probability
        """
        super(GunshotCNN, self).__init__()
        
        self.n_mels = n_mels
        self.n_time_frames = n_time_frames
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Define CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size after convolutions
        self._calculate_flattened_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
    
    def _calculate_flattened_size(self):
        """Calculate size of flattened features after conv layers"""
        # Start with input size
        h, w = self.n_mels, self.n_time_frames
        
        # After conv1 + pool: h//2, w//2
        h, w = h // 2, w // 2
        
        # After conv2 + pool: h//4, w//4  
        h, w = h // 2, w // 2
        
        # After conv3 + pool: h//8, w//8
        h, w = h // 2, w // 2
        
        # Flattened size = channels * height * width
        self.flattened_size = 128 * h * w
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: input spectrogram tensor (batch_size, 1, n_mels, n_time_frames)
            
        Returns:
            output logits (batch_size, num_classes)
        """
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2  
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for visualization
        
        Args:
            x: input spectrogram tensor
            
        Returns:
            tuple of feature maps from each conv layer
        """
        # Get feature maps from each conv layer
        conv1_out = F.relu(self.bn1(self.conv1(x)))
        conv2_out = F.relu(self.bn2(self.conv2(self.pool(conv1_out))))
        conv3_out = F.relu(self.bn3(self.conv3(self.pool(conv2_out))))
        
        return conv1_out, conv2_out, conv3_out

class CNNConfig:
    """Configuration class for CNN model"""
    
    def __init__(self, n_mels=128, n_time_frames=173, num_classes=1, 
                 dropout_rate=0.5, learning_rate=0.001, weight_decay=1e-4):
        """
        Initialize CNN configuration
        
        Args:
            n_mels: mel frequency bins
            n_time_frames: time frames in spectrogram
            num_classes: output classes
            dropout_rate: dropout probability
            learning_rate: learning rate for optimizer
            weight_decay: weight decay for regularization
        """
        self.n_mels = n_mels
        self.n_time_frames = n_time_frames
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

def create_cnn_model(config):
    """
    Create CNN model from configuration
    
    Args:
        config: CNN configuration object
        
    Returns:
        initialized CNN model
    """
    model = GunshotCNN(
        n_mels=config.n_mels,
        n_time_frames=config.n_time_frames,
        num_classes=config.num_classes,
        dropout_rate=config.dropout_rate
    )
    
    return model

def count_parameters(model):
    """
    Count number of trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_cnn_model():
    """Test the CNN model with dummy data"""
    print("Testing CNN model...")
    
    # Create config
    config = CNNConfig()
    
    # Create model
    model = create_cnn_model(config)
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, config.n_mels, config.n_time_frames)
    
    # Test forward pass
    output = model(dummy_input)
    
    print(f"Model created successfully!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test feature maps
    feature_maps = model.get_feature_maps(dummy_input)
    print(f"Feature maps: {len(feature_maps)} layers")
    for i, fm in enumerate(feature_maps):
        print(f"  Layer {i+1}: {fm.shape}")

if __name__ == "__main__":
    test_cnn_model() 