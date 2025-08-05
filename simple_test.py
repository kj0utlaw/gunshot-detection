#!/usr/bin/env python3

import os
import sys
import subprocess
import pandas as pd
import json
from comprehensive_test import load_ground_truth, compare_results, calculate_metrics

def test_single_file(audio_file, max_clips=50):
    """Test a single file through the pipeline"""
    print(f"\n{'='*60}")
    print(f"TESTING FILE: {audio_file}")
    print(f"{'='*60}")
    
    # Create temp directory
    temp_dir = f"temp_test_{audio_file.replace('.wav', '')}"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Step 1: Get ground truth
        ground_truth = load_ground_truth(f"dataset/Testing/Sounds/{audio_file}")
        print(f"Ground truth: {len(ground_truth)} gunshots")
        
        # Step 2: Copy limited number of clips
        clips_dir = os.path.join(temp_dir, "spikes")
        os.makedirs(clips_dir, exist_ok=True)
        
        # Get clips from detected_clips
        all_clips = [f for f in os.listdir("detected_clips") if f.endswith('.wav')]
        test_clips = all_clips[:max_clips]  # Just take first 50 clips
        
        print(f"Using {len(test_clips)} clips for testing")
        
        # Copy clips
        for clip in test_clips:
            src = os.path.join("detected_clips", clip)
            dst = os.path.join(clips_dir, clip)
            os.system(f"cp '{src}' '{dst}'")
        
        # Step 3: Create clip index
        clip_entries = []
        for clip in test_clips:
            clip_entries.append({
                'filename': clip,
                'path': os.path.join("spikes", clip),
                'label': 'unknown'
            })
        
        clip_index_path = os.path.join(temp_dir, "clip_index.csv")
        pd.DataFrame(clip_entries).to_csv(clip_index_path, index=False)
        
        # Step 4: Run feature extraction
        print("Running feature extraction...")
        features_dir = os.path.join(temp_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        
        cmd = [
            "python3", "extract_features.py",
            "--clips-dir", clips_dir,
            "--output-dir", features_dir,
            "--clip-index", clip_index_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Feature extraction failed: {result.stderr}")
            return None
        
        # Step 5: Run Isolation Forest
        print("Running Isolation Forest...")
        features_csv = os.path.join(features_dir, "extracted_features.csv")
        iso_model_path = "iso_forest_model.joblib"
        
        cmd = [
            "python3", "iso_forest.py",
            "--features-csv", features_csv,
            "--model-path", iso_model_path,
            "--contamination", "0.5"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Isolation Forest failed: {result.stderr}")
            return None
        
        # Step 6: Run CNN
        print("Running CNN inference...")
        cnn_output = os.path.join(temp_dir, "cnn_results.csv")
        
        cmd = [
            "python3", "cnn_infer.py",
            "--clip-dir", clips_dir,
            "--checkpoint", "best_model.pth",
            "--output", cnn_output,
            "--threshold", "0.9"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"CNN inference failed: {result.stderr}")
            return None
        
        # Step 7: Load and analyze results
        if os.path.exists(cnn_output):
            cnn_results = pd.read_csv(cnn_output)
            gunshot_predictions = cnn_results[cnn_results['prediction'] == 'gunshot']['filename'].tolist()
            
            print(f"\nCNN Results:")
            print(f"Total clips processed: {len(cnn_results)}")
            print(f"Gunshot predictions: {len(gunshot_predictions)}")
            print(f"Non-gunshot predictions: {len(cnn_results) - len(gunshot_predictions)}")
            
            # Compare with ground truth
            results = compare_results(ground_truth, gunshot_predictions)
            
            print(f"\nComparison Results:")
            print(f"True Positives: {results['true_positives']}")
            print(f"False Positives: {results['false_positives']}")
            print(f"False Negatives: {results['false_negatives']}")
            
            # Calculate metrics
            tp = results['true_positives']
            fp = results['false_positives']
            fn = results['false_negatives']
            tn = len(cnn_results) - tp - fp - fn
            
            metrics = calculate_metrics(tp, fp, tn, fn)
            
            print(f"\nPerformance Metrics:")
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
            print(f"F1 Score: {metrics['f1_score']:.3f}")
            
            return metrics
        else:
            print("No CNN results file found!")
            return None
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
    finally:
        # Cleanup
        os.system(f"rm -rf {temp_dir}")

def main():
    """Test on a few files"""
    test_files = [
        "kp01_20151208_000000.wav",  # Has 7 gunshots
        "kp03_20130618_000000.wav"   # Has 10 gunshots
    ]
    
    print("SIMPLE MODEL TEST")
    print("="*60)
    
    all_metrics = []
    
    for file in test_files:
        metrics = test_single_file(file, max_clips=200)  # Test with more clips
        if metrics:
            all_metrics.append(metrics)
    
    if all_metrics:
        print(f"\n{'='*60}")
        print("OVERALL RESULTS")
        print(f"{'='*60}")
        
        # Average metrics
        avg_accuracy = sum(m['accuracy'] for m in all_metrics) / len(all_metrics)
        avg_precision = sum(m['precision'] for m in all_metrics) / len(all_metrics)
        avg_recall = sum(m['recall'] for m in all_metrics) / len(all_metrics)
        avg_f1 = sum(m['f1_score'] for m in all_metrics) / len(all_metrics)
        
        print(f"Average Accuracy: {avg_accuracy:.3f}")
        print(f"Average Precision: {avg_precision:.3f}")
        print(f"Average Recall: {avg_recall:.3f}")
        print(f"Average F1 Score: {avg_f1:.3f}")

if __name__ == "__main__":
    main() 