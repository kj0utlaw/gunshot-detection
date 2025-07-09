"""
Isolation Forest for unsupervised gunshot/anomaly detection in audio.

Loads extracted features and applies Isolation Forest to detect anomalous audio segments.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Tuple, List, Optional

# Import our feature loading function
from extract_features import load_features_for_ml

class IsolationForestDetector:
    """Isolation Forest-based anomaly detector for audio segments"""
    
    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize the Isolation Forest detector
        
        Args:
            contamination: expected proportion of anomalies (0.1 = 10%)
            random_state: for reproducible results
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.anomaly_scores = None
    
    def load_data(self, features_csv_path: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load and prepare features for Isolation Forest
        
        Args:
            features_csv_path: path to extracted_features.csv
            
        Returns:
            X_scaled: scaled feature matrix
            feature_names: list of feature column names
        """
        print(f"Loading features from: {features_csv_path}")
        
        # Use our helper function from extract_features.py
        X_scaled, feature_names, scaler = load_features_for_ml(features_csv_path)
        
        # Store for later use
        self.scaler = scaler
        self.feature_names = feature_names
        
        print(f"Loaded {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features")
        print(f"Feature names: {feature_names[:5]}...")  # Show first 5
        
        return X_scaled, feature_names 