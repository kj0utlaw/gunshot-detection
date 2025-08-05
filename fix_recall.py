#!/usr/bin/env python3
"""
Fix recall calculation in results file.
Correct formula: recall = TP / Ground Truth (if Ground Truth > 0, else 0)
"""

import pandas as pd
import numpy as np

def fix_recall_calculation(input_file='full_dataset_results.csv', output_file='corrected_results.csv'):
    """
    Fix recall calculation using correct formula:
    recall = TP / Ground Truth (if Ground Truth > 0, else 0)
    Cap recall at 1.0 if TP > Ground Truth
    """
    
    print(f"Loading results from: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check if required columns exist
    required_cols = ['Ground Truth', 'Estimated True Positives']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    print("\nBefore fixing recall:")
    print(f"Recall range: {df['recall'].min():.3f} to {df['recall'].max():.3f}")
    print(f"Mean recall: {df['recall'].mean():.3f}")
    
    # Fix recall calculation
    print("\nFixing recall calculation...")
    
    # Calculate correct recall
    def calculate_correct_recall(row):
        ground_truth = row['Ground Truth']
        estimated_tp = row['Estimated True Positives']
        
        if ground_truth > 0:
            recall = estimated_tp / ground_truth
            # Cap at 1.0 if TP > Ground Truth
            return min(recall, 1.0)
        else:
            return 0.0
    
    # Apply the correction
    df['recall'] = df.apply(calculate_correct_recall, axis=1)
    
    print("\nAfter fixing recall:")
    print(f"Recall range: {df['recall'].min():.3f} to {df['recall'].max():.3f}")
    print(f"Mean recall: {df['recall'].mean():.3f}")
    
    # Show some examples
    print("\nSample results:")
    sample_cols = ['filename', 'Ground Truth', 'Estimated True Positives', 'recall']
    available_cols = [col for col in sample_cols if col in df.columns]
    print(df[available_cols].head(10))
    
    # Save corrected results
    df.to_csv(output_file, index=False)
    print(f"\nCorrected results saved to: {output_file}")
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"Files with Ground Truth > 0: {(df['Ground Truth'] > 0).sum()}")
    print(f"Files with Estimated TP > 0: {(df['Estimated True Positives'] > 0).sum()}")
    print(f"Files with recall = 1.0: {(df['recall'] == 1.0).sum()}")
    print(f"Files with recall = 0.0: {(df['recall'] == 0.0).sum()}")

if __name__ == "__main__":
    fix_recall_calculation() 