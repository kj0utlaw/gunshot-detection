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
import joblib  # For saving/loading scikit-learn models

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
        
        try:
            print(f"Loaded {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features")
        except Exception as e:
            print(f"Error loading features: {e}")
        
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
    
    def save_model(self, model_path: str):
        """
        Save the trained model and scaler to disk
        
        Args:
            model_path: path to save the model (without extension)
        """
        if self.model is None:
            raise ValueError("No trained model to save.")
        
        # Save the model
        joblib.dump(self.model, f"{model_path}_model.joblib")
        
        # Save the scaler
        joblib.dump(self.scaler, f"{model_path}_scaler.joblib")
        
        # Save metadata
        metadata = {
            'contamination': self.contamination,
            'random_state': self.random_state,
            'feature_names': self.feature_names
        }
        with open(f"{model_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_path}_model.joblib")
        print(f"Scaler saved to {model_path}_scaler.joblib")
        print(f"Metadata saved to {model_path}_metadata.json")
    
    def load_model(self, model_path: str):
        """
        Load a trained model from disk
        
        Args:
            model_path: path to the model (without extension)
        """
        try:
            # Load the model
            self.model = joblib.load(f"{model_path}_model.joblib")
            
            # Load the scaler
            self.scaler = joblib.load(f"{model_path}_scaler.joblib")
            
            # Load metadata
            with open(f"{model_path}_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            self.contamination = metadata['contamination']
            self.random_state = metadata['random_state']
            self.feature_names = metadata['feature_names']
            
            print(f"Model loaded from {model_path}_model.joblib")
            return True
            
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
            return False

    def save_results(self, features_csv_path: str, labels: np.ndarray, scores: np.ndarray, output_csv: str = "anomaly_results.csv"):
        """
        Save anomaly labels and scores alongside filenames to a CSV.
        Also save a version with just scores for custom thresholding.
        """
        df = pd.read_csv(features_csv_path)
        df['anomaly_label'] = labels
        df['anomaly_score'] = scores
        
        # Sort by anomaly score (most anomalous first)
        df_sorted = df.sort_values('anomaly_score', ascending=False).reset_index(drop=True)
        
        # Save both versions
        df.to_csv(output_csv, index=False)
        df_sorted.to_csv("anomaly_results_sorted.csv", index=False)
        
        print(f"Results saved to {output_csv}")
        print(f"Sorted results (by score) saved to anomaly_results_sorted.csv")
        
        # Print score statistics
        print(f"Anomaly score range: {scores.min():.4f} to {scores.max():.4f}")
        print(f"Mean score: {scores.mean():.4f}")
        print(f"Median score: {np.median(scores):.4f}")

    def get_custom_threshold_results(self, features_csv_path: str, threshold_percentile: float = 80.0):
        """
        Get results using a custom threshold based on percentile.
        
        Args:
            features_csv_path: path to features CSV
            threshold_percentile: percentile to use as threshold (e.g., 80 = top 20% most anomalous)
        """
        if self.anomaly_scores is None:
            raise ValueError("Model not trained yet.")
            
        # Calculate threshold
        threshold = np.percentile(self.anomaly_scores, threshold_percentile)
        
        # Apply custom threshold
        custom_labels = np.where(self.anomaly_scores <= threshold, -1, 1)
        
        print(f"Custom threshold (percentile {threshold_percentile}): {threshold:.4f}")
        print(f"Custom labels - anomalies: {np.sum(custom_labels == -1)}, normal: {np.sum(custom_labels == 1)}")
        
        # Save custom results
        df = pd.read_csv(features_csv_path)
        df['custom_anomaly_label'] = custom_labels
        df['anomaly_score'] = self.anomaly_scores
        df_sorted = df.sort_values('anomaly_score', ascending=False).reset_index(drop=True)
        
        custom_output = f"anomaly_results_custom_{threshold_percentile}.csv"
        df_sorted.to_csv(custom_output, index=False)
        print(f"Custom results saved to {custom_output}")
        
        return custom_labels, threshold

    def plot_anomaly_scores(self, scores: np.ndarray):
        """
        Plot histogram of anomaly scores.
        """
        plt.figure(figsize=(8, 4))
        plt.hist(scores, bins=50, color='skyblue')
        plt.title('Anomaly Score Distribution')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Count')
        plt.show()

    def calculate_recall(self, features_csv_path: str, custom_labels: np.ndarray):
        """
        Calculate recall (percentage of actual gunshots detected).
        
        Args:
            features_csv_path: path to features CSV with true labels
            custom_labels: predicted anomaly labels (-1 = anomaly, 1 = normal)
        """
        df = pd.read_csv(features_csv_path)
        
        # Count actual gunshots
        actual_gunshots = (df['label'] == 'gunshot').sum()
        
        # Count gunshots that were detected as anomalies
        detected_gunshots = ((df['label'] == 'gunshot') & (custom_labels == -1)).sum()
        
        # Calculate recall
        recall = (detected_gunshots / actual_gunshots) * 100 if actual_gunshots > 0 else 0
        
        print(f"\nRECALL ANALYSIS:")
        print(f"   - Actual gunshots in dataset: {actual_gunshots}")
        print(f"   - Gunshots detected as anomalies: {detected_gunshots}")
        print(f"   - Recall: {recall:.1f}% of gunshots detected")
        
        return recall, detected_gunshots, actual_gunshots

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Isolation Forest for anomaly detection")
    parser.add_argument("--features-csv", default="features/extracted_features.csv",
                       help="Path to extracted features CSV file")
    parser.add_argument("--contamination", type=float, default=0.5,
                       help="Contamination parameter for Isolation Forest")
    parser.add_argument("--threshold-percentile", type=float, default=90.0,
                       help="Percentile threshold for custom anomaly detection")
    parser.add_argument("--model-path", default=None,
                       help="Path to pre-trained model (without extension). If not provided, will train new model.")
    parser.add_argument("--save-model", action="store_true",
                       help="Save the trained model for future use")
    
    args = parser.parse_args()
    
    # Initialize detector with specified contamination
    detector = IsolationForestDetector(contamination=args.contamination)
    
    # Try to load pre-trained model first
    if args.model_path and detector.load_model(args.model_path):
        print("Using pre-trained model")
        # Load data for prediction (but don't retrain)
        X_scaled, feature_names = detector.load_data(args.features_csv)
        # Get predictions using loaded model
        labels = detector.predict(X_scaled)
        scores = detector.model.score_samples(X_scaled)
        detector.anomaly_scores = scores
    else:
        print("Training new model")
        # Load data and train new model
        X_scaled, feature_names = detector.load_data(args.features_csv)
        detector.train(X_scaled)
        
        # Get standard predictions
        labels = detector.predict(X_scaled)
        scores = detector.anomaly_scores
        
        # Save model if requested
        if args.save_model:
            model_path = f"models/iso_forest_model_{args.contamination}"
            import os
            os.makedirs("models", exist_ok=True)
            detector.save_model(model_path)
    
    print(f"Total anomalies detected: {(labels == -1).sum()} out of {len(labels)} samples/n")
    
    # Save standard results
    detector.save_results(args.features_csv, labels, scores)
    
    # Use specified percentile for maximum recall
    print(f"\nMAXIMUM RECALL: {args.threshold_percentile}TH PERCENTILE THRESHOLD")
    print("=" * 40)
    
    # Use specified percentile for maximum recall
    custom_labels, threshold = detector.get_custom_threshold_results(args.features_csv, args.threshold_percentile)
    
    # Calculate and display recall
    detector.calculate_recall(args.features_csv, custom_labels)
    
    print(f"\nRESULTS ({args.threshold_percentile}th percentile):")
    print(f"   - Threshold: {threshold:.4f}")
    print(f"   - Anomalies detected: {np.sum(custom_labels == -1)} out of {len(custom_labels)} samples")
    print(f"   - Coverage: {np.sum(custom_labels == -1)/len(custom_labels)*100:.1f}% of samples")
    print(f"   - Results saved to: anomaly_results_custom_{args.threshold_percentile}.csv")
    
    # Plot scores
    # detector.plot_anomaly_scores(detector.anomaly_scores)  # COMMENTED OUT FOR AUTOMATED TESTING