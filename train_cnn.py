"""
Training pipeline for CNN gunshot detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from cnn_gunshot import CNNConfig, create_cnn_model, count_parameters
from data_loader import create_data_loaders

class Trainer:
    """Training class for CNN model"""
    
    def __init__(self, config, train_loader, val_loader, device='cpu'):
        """
        Initialize trainer
        
        Args:
            config: CNN configuration
            train_loader: training data loader
            val_loader: validation data loader
            device: device to train on
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Create model and move to device
        self.model = create_cnn_model(config).to(device)
        
        # Loss and optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # LR scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Track training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        print(f"Model parameters: {count_parameters(self.model):,}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for data, targets in progress_bar:
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs.squeeze(), targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            predictions = torch.sigmoid(outputs).squeeze()
            predicted_labels = (predictions > 0.5).float()
            correct += (predicted_labels == targets).sum().item()
            total += targets.size(0)
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs.squeeze(), targets)
                
                predictions = torch.sigmoid(outputs).squeeze()
                predicted_labels = (predictions > 0.5).float()
                correct += (predicted_labels == targets).sum().item()
                total += targets.size(0)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs, save_dir='checkpoints'):
        """
        Train the model
        
        Args:
            num_epochs: number of training epochs
            save_dir: directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f"{save_dir}/best_model.pth", epoch)
                print("Saved best model!")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"{save_dir}/checkpoint_epoch_{epoch+1}.pth", epoch)
        
        # Save final model
        self.save_checkpoint(f"{save_dir}/final_model.pth", num_epochs - 1)
        
        print("\nTraining completed!")
        self.plot_training_history()
    
    def save_checkpoint(self, filepath, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_accuracies = checkpoint['val_accuracies']
        return checkpoint['epoch']
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Acc')
        ax2.plot(self.val_accuracies, label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main training function"""
    try:
        # Configuration
        config = CNNConfig(
            n_mels=128,
            n_time_frames=173,
            num_classes=1,
            dropout_rate=0.5,
            learning_rate=0.001,
            weight_decay=1e-4
        )
        
        # Data loaders
        print("Loading data...")
        train_loader, val_loader = create_data_loaders(
            clip_index_path="dataset/clip_index.csv",
            clips_dir="dataset/clips",
            batch_size=32,
            train_split=0.8,
            num_workers=4,  # Recommended for GPU (Kaggle)
            pin_memory=True  # Recommended for GPU (Kaggle)
        )
        
        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create trainer and train
        trainer = Trainer(config, train_loader, val_loader, device)
        trainer.train(num_epochs=50, save_dir='checkpoints')
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Make sure dataset/clip_index.csv and dataset/clips/ exist")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 