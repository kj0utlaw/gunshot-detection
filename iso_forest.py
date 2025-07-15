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
    
    def train(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Train Isolation Forest and get anomaly scores
        
        Args:
            X_scaled: scaled feature matrix
            
        Returns:
            anomaly_scores: array of anomaly scores (negative = more anomalous)
        """
        print("Training Isolation Forest...")
        
        # Create and fit the model
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state
        )
        
        # Fit the model and get predictions
        predictions = self.model.fit_predict(X_scaled)
        
        # Get anomaly scores (negative scores = more anomalous)
        self.anomaly_scores = self.model.score_samples(X_scaled)
        
        print(f"Training complete! Found {np.sum(predictions == -1)} anomalies out of {len(predictions)} samples")
        
        return self.anomaly_scores 

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels for the given data.
        Returns: array of labels (-1 = anomaly, 1 = normal)
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.predict(X_scaled)

if __name__ == "__main__":
    # Example usage
    features_csv = "features/extracted_features.csv"
    detector = IsolationForestDetector(contamination=0.1)
    X_scaled, feature_names = detector.load_data(features_csv)
    detector.train(X_scaled)
    labels = detector.predict(X_scaled)
    print("Anomaly labels (first 20):", labels[:20])
    print(f"Total anomalies detected: {(labels == -1).sum()} out of {len(labels)} samples") 