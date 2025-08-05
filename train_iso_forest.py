"""
Train Isolation Forest model on dataset/Training data.

This script:
1. Extracts features from dataset/Training clips
2. Trains Isolation Forest on non-gunshot clips (to learn "normal" sounds)
3. Saves the trained model for use in production/testing
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import subprocess
import sys

def extract_training_features():
    """Extract features from dataset/Training clips"""
    print("Extracting features from training dataset...")
    
    # Check if training features already exist
    if os.path.exists("dataset/Training/training_features.csv"):
        print("Training features already exist, skipping extraction...")
        return "dataset/Training/training_features.csv"
    
    # Run feature extraction on training dataset
    subprocess.run([
        sys.executable, 'extract_features.py',
        '--clips-dir', 'dataset/Training/clips',
        '--output-dir', 'dataset/Training',
        '--clip-index', 'dataset/Training/clip_index.csv'
    ], check=True)
    
    return "dataset/Training/extracted_features.csv"

def filter_non_gunshot_data(features_csv):
    """Use all training data (both gunshot and non-gunshot clips) for training"""
    print("Using all training data (both gunshot and non-gunshot clips)...")
    
    df = pd.read_csv(features_csv)
    
    print(f"Total training clips: {len(df)}")
    print(f"Gunshot clips: {len(df[df['label'] == 'gunshot'])}")
    print(f"Non-gunshot clips: {len(df[df['label'] == 'not_gunshot'])}")
    
    # Use all data for training (don't filter)
    training_features_path = "dataset/Training/training_features.csv"
    df.to_csv(training_features_path, index=False)
    
    return training_features_path

def train_iso_forest_model(features_csv, contamination=0.1):
    """Train Isolation Forest model and save it"""
    print(f"Training Isolation Forest with contamination={contamination}...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Train and save model
    subprocess.run([
        sys.executable, 'iso_forest.py',
        '--features-csv', features_csv,
        '--contamination', str(contamination),
        '--save-model'
    ], check=True)
    
    model_path = f"models/iso_forest_model_{contamination}"
    print(f"Model saved to {model_path}_model.joblib")
    
    return model_path

def main():
    parser = argparse.ArgumentParser(description="Train Isolation Forest model on training dataset")
    parser.add_argument("--contamination", type=float, default=0.5,
                       help="Contamination parameter for Isolation Forest (default: 0.5, max allowed)")
    parser.add_argument("--skip-feature-extraction", action="store_true",
                       help="Skip feature extraction if already done")
    
    args = parser.parse_args()
    
    print("=== Training Isolation Forest Model ===")
    print(f"Contamination: {args.contamination}")
    print("Note: Using 0.5 contamination (maximum allowed). For high recall, use custom threshold in inference.")
    print()
    
    # Step 1: Extract features from training dataset
    if not args.skip_feature_extraction:
        features_csv = extract_training_features()
    else:
        features_csv = "dataset/Training/extracted_features.csv"
    
    # Step 2: Filter to non-gunshot data
    training_features = filter_non_gunshot_data(features_csv)
    
    # Step 3: Train Isolation Forest model
    model_path = train_iso_forest_model(training_features, args.contamination)
    
    print()
    print("=== Training Complete ===")
    print(f"Model saved to: {model_path}_model.joblib")
    print(f"Scaler saved to: {model_path}_scaler.joblib")
    print(f"Metadata saved to: {model_path}_metadata.json")
    print()
    print("The model is now ready for use in the test pipeline!")

if __name__ == "__main__":
    main() 