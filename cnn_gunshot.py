"""
CNN model for gunshot classification from Mel spectrograms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class GunshotCNN(nn.Module):
    """CNN architecture matching the saved model with 3 conv blocks"""
    def __init__(self, n_mels=128, n_time_frames=173, num_classes=1, dropout_rate=0.6):
        super(GunshotCNN, self).__init__()
        self.n_mels = n_mels
        self.n_time_frames = n_time_frames
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # 2 conv blocks to match saved model exactly
        # Block 1: 1 -> 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2: 32 -> 64 channels  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size after pooling layers
        self._calculate_flattened_size()
        
        # Fully connected layers (matching saved model exactly)
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def _calculate_flattened_size(self):
        h, w = self.n_mels, self.n_time_frames
        h, w = h // 2, w // 2  # After pool1
        h, w = h // 2, w // 2  # After pool2
        self.flattened_size = 64 * h * w
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for visualization
        
        Args:
            x: input spectrogram tensor
            
        Returns:
            dict of feature maps at different layers
        """
        feature_maps = {}
        
        # Single conv layer
        x = F.relu(self.conv1(x))
        feature_maps['conv1'] = x
        x = self.pool1(x)
        x = self.pool1(x)
        
        return feature_maps

class CNNConfig:
    """Configuration class for CNN model"""
    
    def __init__(self, n_mels=128, n_time_frames=173, num_classes=1, 
                                   dropout_rate=0.7, learning_rate=0.00005, weight_decay=5e-3):
        """
        Initialize CNN configuration
        
        Args:
            n_mels: mel frequency bins
            n_time_frames: time frames in spectrogram
            num_classes: output classes
            dropout_rate: dropout probability (increased)
            learning_rate: learning rate for optimizer (decreased)
            weight_decay: weight decay for regularization (increased)
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
    for name, fm in feature_maps.items():
        print(f"  {name}: {fm.shape}")

if __name__ == "__main__":
    test_cnn_model() 