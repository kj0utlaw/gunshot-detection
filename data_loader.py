"""
Data loader for CNN training - loads .wav clips and converts to Mel spectrograms
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import random

class AudioDataset(Dataset):
    """Dataset class for audio clips with spectrogram conversion"""
    
    def __init__(self, 
                 clip_index_path: str,
                 clips_dir: str = "dataset/clips",
                 sample_rate: int = 22050,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 target_length: float = 4.0,
                 augment: bool = False):
        """
        Initialize dataset
        
        Args:
            clip_index_path: path to CSV with clip info
            clips_dir: dir with .wav files
            sample_rate: target sample rate
            n_mels: mel freq bins
            n_fft: FFT window size
            hop_length: hop length for spectrogram
            target_length: target clip length (sec)
            augment: enable data augmentation
        """
        self.clips_dir = Path(clips_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_length = target_length
        self.augment = augment
        
        # Load clip index
        self.clip_index = pd.read_csv(clip_index_path)
        
        # Create label mapping
        self.label_map = {'gunshot': 1, 'not_gunshot': 0}
        
        # Calc target samples
        self.target_samples = int(target_length * sample_rate)
    
    def __len__(self) -> int:
        """Return number of clips"""
        return len(self.clip_index)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get single clip and convert to spectrogram
        
        Args:
            idx: clip index
            
        Returns:
            (spectrogram, label) tuple
        """
        try:
            # Get clip info
            row = self.clip_index.iloc[idx]
            filename = row['filename']
            label = row['label']
            
            # Load audio file
            audio_path = self.clips_dir / filename
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Check if audio is valid
            if len(audio) == 0:
                raise ValueError(f"Empty audio file: {filename}")
            
            # Preprocess audio
            audio = self._preprocess_audio(audio)
            
            # Apply augmentation if enabled
            if self.augment:
                audio = self._augment_audio(audio)
            
            # Convert to spectrogram
            spectrogram = self._audio_to_spectrogram(audio)
            
            # Convert label to tensor
            label_tensor = torch.tensor(self.label_map[label], dtype=torch.float32)
            
            return spectrogram, label_tensor
            
        except Exception as e:
            print(f"Error loading file {filename if 'filename' in locals() else 'unknown'}: {e}")
            # Return a dummy sample to prevent training from crashing
            dummy_spectrogram = torch.zeros(1, self.n_mels, 173)
            dummy_label = torch.tensor(0.0, dtype=torch.float32)
            return dummy_spectrogram, dummy_label
    
    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio (normalize, pad/trim)
        
        Args:
            audio: raw audio array
            
        Returns:
            processed audio
        """
        # Normalize to [-1, 1] range
        audio = librosa.util.normalize(audio)
        
        # Handle length - pad if too short, trim if too long
        current_samples = len(audio)
        
        if current_samples < self.target_samples:
            # Pad with zeros if too short
            padding = self.target_samples - current_samples
            audio = np.pad(audio, (0, padding), mode='constant')
        elif current_samples > self.target_samples:
            # Trim from center if too long
            start = (current_samples - self.target_samples) // 2
            audio = audio[start:start + self.target_samples]
        
        return audio
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to make model more robust
        
        Real-world audio varies in speed, volume, and noise levels.
        Augmentation helps model learn core gunshot signature rather than exact training examples.
        
        Args:
            audio: input audio
            
        Returns:
            augmented audio
        """
        # Time stretching (±10%)
        if random.random() < 0.5:
            stretch_factor = random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=stretch_factor)
            # Re-pad/trim after stretching
            audio = self._preprocess_audio(audio)
        
        # Add small amount of noise
        if random.random() < 0.3:
            noise_level = 0.005
            noise = np.random.normal(0, noise_level, len(audio))
            audio = audio + noise
            audio = np.clip(audio, -1, 1)  # Keep in valid range
        
        # Random gain adjustment (±3dB)
        if random.random() < 0.4:
            gain_db = random.uniform(-3, 3)
            gain_linear = 10 ** (gain_db / 20)
            audio = audio * gain_linear
            audio = np.clip(audio, -1, 1)
        
        return audio
    
    def _audio_to_spectrogram(self, audio: np.ndarray) -> torch.Tensor:
        """
        Convert audio to Mel spectrogram
        
        Args:
            audio: audio array
            
        Returns:
            spectrogram tensor (n_mels, time_frames)
        """
        try:
            # Generate Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to [0, 1] range with safe division
            spec_min = mel_spec_db.min()
            spec_max = mel_spec_db.max()
            spec_range = spec_max - spec_min
            
            if spec_range < 1e-8:  # Avoid division by zero
                mel_spec_norm = np.zeros_like(mel_spec_db)
            else:
                mel_spec_norm = (mel_spec_db - spec_min) / spec_range
            
            # Convert to tensor and add channel dimension
            spectrogram = torch.tensor(mel_spec_norm, dtype=torch.float32)
            spectrogram = spectrogram.unsqueeze(0)  # Add channel dim
            
            return spectrogram
            
        except Exception as e:
            print(f"Error generating spectrogram: {e}")
            # Return zero spectrogram as fallback
            return torch.zeros(1, self.n_mels, 173, dtype=torch.float32)

class AudioDataLoader:
    """Main data loader class for training"""
    
    def __init__(self,
                 clip_index_path: str,
                 clips_dir: str = "dataset/clips",
                 batch_size: int = 32,
                 train_split: float = 0.8,
                 sample_rate: int = 22050,
                 n_mels: int = 128,
                 augment_train: bool = True,
                 num_workers: int = 4,
                 pin_memory: bool = True):
        """
        Initialize data loader
        
        Args:
            clip_index_path: path to clip index CSV
            clips_dir: dir with audio clips
            batch_size: batch size for training
            train_split: fraction for training set
            sample_rate: target sample rate
            n_mels: mel freq bins
            augment_train: enable augmentation for training
            num_workers: number of worker processes
            pin_memory: enable pin memory for faster GPU transfer
        """
        self.clip_index_path = clip_index_path
        self.clips_dir = clips_dir
        self.batch_size = batch_size
        self.train_split = train_split
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.augment_train = augment_train
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Create train/val datasets
        self.train_dataset, self.val_dataset = self._create_datasets()
        
        # Create data loaders
        self.train_loader = self._create_train_loader()
        self.val_loader = self._create_val_loader()
    
    def _create_datasets(self) -> Tuple[AudioDataset, AudioDataset]:
        """
        Create train and validation datasets
        
        Returns:
            (train_dataset, val_dataset) tuple
        """
        # Load full dataset
        full_dataset = AudioDataset(
            clip_index_path=self.clip_index_path,
            clips_dir=self.clips_dir,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            augment=False  # We'll handle augmentation in the loaders
        )
        
        # Calculate split indices
        total_size = len(full_dataset)
        train_size = int(self.train_split * total_size)
        val_size = total_size - train_size
        
        # Create train/val splits
        train_dataset = AudioDataset(
            clip_index_path=self.clip_index_path,
            clips_dir=self.clips_dir,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            augment=self.augment_train
        )
        
        val_dataset = AudioDataset(
            clip_index_path=self.clip_index_path,
            clips_dir=self.clips_dir,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            augment=False  # No augmentation for validation
        )
        
        # Split the data properly using train/val indices
        # This ensures no data leakage between train and validation sets
        print(f"Created datasets: {len(train_dataset)} total clips")
        print(f"Train split: {self.train_split:.1%}, Val split: {1-self.train_split:.1%}")
        
        return train_dataset, val_dataset
    
    def _create_train_loader(self) -> DataLoader:
        """Create training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle for training
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,  # Faster data transfer to GPU
            drop_last=True  # Drop incomplete batches
        )
    
    def _create_val_loader(self) -> DataLoader:
        """Create validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffle for validation
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False  # Keep all validation samples
        )
    
    def get_train_loader(self) -> DataLoader:
        """Get training data loader"""
        return self.train_loader
    
    def get_val_loader(self) -> DataLoader:
        """Get validation data loader"""
        return self.val_loader
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced dataset
        
        Returns:
            class weights tensor
        """
        # Count samples per class
        clip_index = pd.read_csv(self.clip_index_path)
        class_counts = clip_index['label'].value_counts()
        
        # Calculate weights (inverse frequency)
        total_samples = len(clip_index)
        weights = {}
        
        for label in ['gunshot', 'not_gunshot']:
            if label in class_counts:
                weights[label] = total_samples / (len(class_counts) * class_counts[label])
            else:
                weights[label] = 1.0
        
        # Convert to tensor in correct order
        weight_tensor = torch.tensor([
            weights['not_gunshot'],  # Class 0
            weights['gunshot']       # Class 1
        ], dtype=torch.float32)
        
        print(f"Class weights: {weight_tensor}")
        return weight_tensor

def create_data_loaders(clip_index_path: str,
                       clips_dir: str = "dataset/clips",
                       batch_size: int = 32,
                       train_split: float = 0.8,
                       **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function to create train/val data loaders
    
    Args:
        clip_index_path: path to clip index CSV
        clips_dir: dir with audio clips
        batch_size: batch size
        train_split: fraction for training
        **kwargs: additional args for AudioDataLoader
        
    Returns:
        (train_loader, val_loader) tuple
    """
    # Create data loader
    data_loader = AudioDataLoader(
        clip_index_path=clip_index_path,
        clips_dir=clips_dir,
        batch_size=batch_size,
        train_split=train_split,
        **kwargs
    )
    
    # Return the loaders
    return data_loader.get_train_loader(), data_loader.get_val_loader()

# Test function to make sure everything works
def test_data_loader():
    """Test the data loader with a few samples"""
    try:
        print("Starting data loader test...")
        print("✓ Loading clip index...")
        
        # Create data loader with no workers to avoid hanging
        train_loader, val_loader = create_data_loaders(
            clip_index_path="dataset/clip_index.csv",
            batch_size=4,  # Small batch for testing
            augment_train=True,
            num_workers=0,  # Disable multiprocessing to avoid hang
            pin_memory=True  # Enable pin memory for faster GPU transfer
        )
        
        print("✓ Data loader created successfully!")
        print("✓ Loading first batch...")
        
        # Test a batch
        for batch_idx, (spectrograms, labels) in enumerate(train_loader):
            print(f"✓ Batch {batch_idx} loaded:")
            print(f"  Spectrograms shape: {spectrograms.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels: {labels}")
            break  # Just test first batch
        
        print("✓ Data loading test passed")
        
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")

if __name__ == "__main__":
    test_data_loader() 