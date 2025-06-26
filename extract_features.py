"""
Feature extraction module for gunshot detection system.

This module extracts both handcrafted features (MFCCs, ZCR, RMS, spectral centroid)
and generates Mel spectrograms for CNN input from audio clips in the dataset.
"""

import numpy as np
import pandas as pd
import librosa
import os
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress librosa warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

class FeatureExtractor:
    """Extracts features from audio clips for gunshot detection."""
    
    def __init__(self, 
                 clips_dir: str = "dataset/clips",
                 output_dir: str = "features",
                 sample_rate: int = 22050,
                 n_mfcc: int = 13,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 save_spectrograms: bool = True,
                 spectrogram_format: str = "numpy"):
        """
        Initialize the feature extractor.
        
        Args:
            clips_dir: Directory containing audio clips
            output_dir: Directory to save extracted features
            sample_rate: Target sample rate for audio processing
            n_mfcc: Number of MFCC coefficients to extract
            n_mels: Number of mel frequency bins for spectrograms
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            save_spectrograms: Whether to save spectrograms as files
            spectrogram_format: Format to save spectrograms ('numpy', 'png', or 'both')
        """
        self.clips_dir = Path(clips_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.save_spectrograms = save_spectrograms
        self.spectrogram_format = spectrogram_format
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        if self.save_spectrograms:
            (self.output_dir / "spectrograms").mkdir(exist_ok=True)
            if self.spectrogram_format in ["png", "both"]:
                (self.output_dir / "spectrograms" / "png").mkdir(exist_ok=True)
            if self.spectrogram_format in ["numpy", "both"]:
                (self.output_dir / "spectrograms" / "numpy").mkdir(exist_ok=True)
    
    def extract_handcrafted_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract handcrafted features from audio clip.
        
        Args:
            audio: Audio waveform as numpy array
            sr: Sample rate
            
        Returns:
            Dictionary of extracted features
        """
    features = {}

        # MFCCs (captures tone & timbre)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        for i, coef in enumerate(mfcc):
            features[f'mfcc_{i+1}'] = np.mean(coef)
            features[f'mfcc_{i+1}_std'] = np.std(coef)
        
        # Zero-crossing rate (helps with volume and frequency characteristics)
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # Root Mean Square energy (loudness)
        rms = librosa.feature.rms(y=audio)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # Spectral centroid (brightness / how high-pitched)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features['centroid_mean'] = np.mean(centroid)
        features['centroid_std'] = np.std(centroid)
        
        # Spectral bandwidth (frequency spread)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features['bandwidth_mean'] = np.mean(bandwidth)
        features['bandwidth_std'] = np.std(bandwidth)
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features['rolloff_mean'] = np.mean(rolloff)
        features['rolloff_std'] = np.std(rolloff)
        
        # Additional features for gunshot detection
        # Spectral contrast (captures spectral peaks and valleys)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features['contrast_mean'] = np.mean(contrast)
        features['contrast_std'] = np.std(contrast)
        
        # Chroma features (harmonic content)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        return features
    
    def generate_mel_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Generate Mel spectrogram from audio clip.
        
        Args:
            audio: Audio waveform as numpy array
            sr: Sample rate
            
        Returns:
            Mel spectrogram as numpy array
        """
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def save_spectrogram(self, spectrogram: np.ndarray, filename: str, base_name: str):
        """
        Save spectrogram in specified format.
        
        Args:
            spectrogram: Mel spectrogram array
            filename: Original audio filename
            base_name: Base name without extension
        """
        if not self.save_spectrograms:
            return
        
        if self.spectrogram_format in ["png", "both"]:
            # Save as PNG image
            plt.figure(figsize=(10, 6))
            plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel Spectrogram: {filename}')
            plt.xlabel('Time Frames')
            plt.ylabel('Mel Frequency Bins')
            plt.tight_layout()
            
            png_path = self.output_dir / "spectrograms" / "png" / f"{base_name}.png"
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        if self.spectrogram_format in ["numpy", "both"]:
            # Save as numpy array
            np_path = self.output_dir / "spectrograms" / "numpy" / f"{base_name}.npy"
            np.save(np_path, spectrogram)
    
    def process_single_clip(self, clip_path: Path, label: str) -> Optional[Dict]:
        """
        Process a single audio clip and extract features.
        
        Args:
            clip_path: Path to audio clip
            label: Label for the clip
            
        Returns:
            Dictionary containing features and metadata, or None if processing failed
        """
        try:
            # Load audio
            audio, sr = librosa.load(clip_path, sr=self.sample_rate)
            
            # Extract handcrafted features
            features = self.extract_handcrafted_features(audio, sr)
            
            # Generate mel spectrogram
            spectrogram = self.generate_mel_spectrogram(audio, sr)
            
            # Add metadata
            features['filename'] = clip_path.name
            features['label'] = label
            features['duration'] = len(audio) / sr
            features['sample_rate'] = sr
            features['spectrogram_shape'] = spectrogram.shape
            
            # Save spectrogram
            base_name = clip_path.stem
            self.save_spectrogram(spectrogram, clip_path.name, base_name)
            
            return features
            
        except Exception as e:
            print(f"Error processing {clip_path.name}: {str(e)}")
            return None
    
    def extract_features_from_dataset(self, clip_index_path: str = "dataset/clip_index.csv") -> str:
        """
        Extract features from all clips in the dataset.
        
        Args:
            clip_index_path: Path to the clip index CSV file
            
        Returns:
            Path to the output features CSV file
        """
        # Load clip index
        clip_index = pd.read_csv(clip_index_path)
        
        print(f"Processing {len(clip_index)} audio clips...")
        
        all_features = []
        failed_clips = []
        
        # Process each clip with progress bar
        for _, row in tqdm(clip_index.iterrows(), total=len(clip_index), desc="Extracting features"):
            filename = row['filename']
            label = row['label']
            clip_path = self.clips_dir / filename
            
            if not clip_path.exists():
                print(f"Warning: Clip file not found: {clip_path}")
                failed_clips.append(filename)
                continue
            
            features = self.process_single_clip(clip_path, label)
            
            if features is not None:
                all_features.append(features)
            else:
                failed_clips.append(filename)
        
        # Create DataFrame and save
        if all_features:
            df = pd.DataFrame(all_features)
            output_csv = self.output_dir / "extracted_features.csv"
            df.to_csv(output_csv, index=False)
            
            print(f"\nFeature extraction completed!")
            print(f"Successfully processed: {len(all_features)} clips")
            print(f"Failed to process: {len(failed_clips)} clips")
            print(f"Features saved to: {output_csv}")
            
            if failed_clips:
                failed_file = self.output_dir / "failed_clips.json"
                with open(failed_file, 'w') as f:
                    json.dump(failed_clips, f, indent=2)
                print(f"Failed clips list saved to: {failed_file}")
            
            # Save feature statistics
            self._save_feature_statistics(df)
            
            return str(output_csv)
        else:
            print("No features were successfully extracted!")
            return ""
    
    def _save_feature_statistics(self, df: pd.DataFrame):
        """Save statistics about the extracted features."""
        stats = {
            'total_clips': len(df),
            'labels': df['label'].value_counts().to_dict(),
            'feature_columns': list(df.columns),
            'numeric_features': len([col for col in df.columns if col not in ['filename', 'label']]),
            'average_duration': df['duration'].mean(),
            'duration_std': df['duration'].std()
        }
        
        stats_file = self.output_dir / "feature_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Feature statistics saved to: {stats_file}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract features from audio clips")
    parser.add_argument("--clips-dir", default="dataset/clips", 
                       help="Directory containing audio clips")
    parser.add_argument("--output-dir", default="features", 
                       help="Output directory for features")
    parser.add_argument("--clip-index", default="dataset/clip_index.csv",
                       help="Path to clip index CSV file")
    parser.add_argument("--sample-rate", type=int, default=22050,
                       help="Target sample rate")
    parser.add_argument("--n-mfcc", type=int, default=13,
                       help="Number of MFCC coefficients")
    parser.add_argument("--n-mels", type=int, default=128,
                       help="Number of mel frequency bins")
    parser.add_argument("--save-spectrograms", action="store_true",
                       help="Save spectrograms as files")
    parser.add_argument("--spectrogram-format", choices=["numpy", "png", "both"], 
                       default="numpy", help="Format for saving spectrograms")
    
    args = parser.parse_args()
    
    # Initialize feature extractor
    extractor = FeatureExtractor(
        clips_dir=args.clips_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        n_mels=args.n_mels,
        save_spectrograms=args.save_spectrograms,
        spectrogram_format=args.spectrogram_format
    )
    
    # Extract features
    extractor.extract_features_from_dataset(args.clip_index)


if __name__ == "__main__":
    main()
