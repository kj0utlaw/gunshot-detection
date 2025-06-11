import numpy as np # numerical operations
import pandas as pd # data handling
import os # file paths
from pathlib import Path # path handling

# Combines features, spectrograms, and labels into a dataset
def build_dataset(features_csv, spectrogram_dir, output_path="dataset.csv"):
    """
    Combines traditional audio features and spectrogram paths into a single dataset.
    
    Inputs:
        features_csv: Path to CSV with audio features (from extract_features.py)
        spectrogram_dir: Directory containing spectrogram .npy files
        output_path: Where to save the combined dataset
    """
    # Load audio features
    features_df = pd.read_csv(features_csv)
    
    # Get list of spectrogram files
    spec_files = [f for f in os.listdir(spectrogram_dir) if f.endswith('.npy')]
    
    # Create list to store spectrogram paths
    spec_paths = []
    
    # Match spectrograms to features using labels
    for idx, row in features_df.iterrows():
        label = row['label']
        # Find matching spectrogram file
        matching_spec = [f for f in spec_files if f.startswith(f"{label}_")][0]
        spec_path = os.path.join(spectrogram_dir, matching_spec)
        spec_paths.append(spec_path)
    
    # Add spectrogram paths to features dataframe
    features_df['spectrogram_path'] = spec_paths
    
    # Save combined dataset
    features_df.to_csv(output_path, index=False)  # save without row numbers
    print(f"Dataset saved to {output_path}")  # show where file was saved
    print(f"Total samples: {len(features_df)}")  # count of audio clips
    print(f"Features: {len(features_df.columns) - 2}")  # count features (excluding label & path)
    print(f"Classes: {features_df['label'].unique()}")  # show unique labels (e.g., gunshot, other)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python build_dataset.py path_to_features.csv path_to_spectrograms/")
    else:
        build_dataset(sys.argv[1], sys.argv[2]) 