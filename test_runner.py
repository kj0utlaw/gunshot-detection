#!/usr/bin/env python3
"""
Simple Test Runner for Gunshot Detection System

This script provides an easy way to test the gunshot detection pipeline.
It can run on a single file or multiple files with different configurations.
"""

import os
import sys
import argparse
from full_dataset_test import test_single_file_realistic

def main():
    parser = argparse.ArgumentParser(description='Test Gunshot Detection System')
    parser.add_argument('--file', type=str, help='Single audio file to test (e.g., kp01_20151208_000000.wav)')
    parser.add_argument('--max-clips', type=int, default=300, help='Maximum number of clips to process (default: 300)')
    parser.add_argument('--threshold', type=float, default=0.9, help='CNN confidence threshold (default: 0.9)')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer clips (50)')
    
    args = parser.parse_args()
    
    print("🔫 Gunshot Detection System - Test Runner")
    print("="*50)
    
    if args.quick:
        args.max_clips = 50
        print("🚀 Quick test mode enabled")
    
    if args.file:
        # Test single file
        if not os.path.exists(f"dataset/Testing/Sounds/{args.file}"):
            print(f"❌ Error: File {args.file} not found in dataset/Testing/Sounds/")
            print("Available files:")
            test_dir = "dataset/Testing/Sounds"
            if os.path.exists(test_dir):
                files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
                for f in files[:5]:  # Show first 5 files
                    print(f"  - {f}")
                if len(files) > 5:
                    print(f"  ... and {len(files)-5} more files")
            return
        
        print(f"🎯 Testing single file: {args.file}")
        print(f"📊 Max clips: {args.max_clips}")
        print(f"🎚️  Threshold: {args.threshold}")
        print()
        
        result = test_single_file_realistic(args.file, max_clips=args.max_clips, cnn_threshold=args.threshold)
        
        if result:
            print("\n" + "="*50)
            print("✅ TEST COMPLETED SUCCESSFULLY")
            print("="*50)
            print(f"📁 File: {result['file']}")
            print(f"🎯 Ground Truth Gunshots: {result['ground_truth_count']}")
            print(f"🔍 Predictions: {result['predictions']}")
            print(f"✅ Estimated True Positives: {result['estimated_tp']}")
            print(f"❌ False Positives: {result['fp']}")
            print(f"❌ False Negatives: {result['fn']}")
            print(f"✅ True Negatives: {result['tn']}")
            print()
            print("📈 Performance Metrics:")
            print(f"   Accuracy: {result['accuracy']:.3f}")
            print(f"   Precision: {result['precision']:.3f}")
            print(f"   Recall: {result['recall']:.3f}")
            print(f"   F1 Score: {result['f1_score']:.3f}")
            print(f"   FPR: {result['fpr']:.3f}")
            print(f"   Specificity: {result['specificity']:.3f}")
        else:
            print("❌ Test failed!")
    
    else:
        # Show available options
        print("🎯 Gunshot Detection Test Runner")
        print()
        print("Usage examples:")
        print("  python3 test_runner.py --file kp01_20151208_000000.wav")
        print("  python3 test_runner.py --file kp01_20151208_000000.wav --quick")
        print("  python3 test_runner.py --file kp01_20151208_000000.wav --threshold 0.8")
        print("  python3 test_runner.py --file kp01_20151208_000000.wav --max-clips 100")
        print()
        print("For full dataset testing:")
        print("  python3 full_dataset_test.py")
        print()
        print("Available test files:")
        test_dir = "dataset/Testing/Sounds"
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
            for f in files[:10]:  # Show first 10 files
                print(f"  - {f}")
            if len(files) > 10:
                print(f"  ... and {len(files)-10} more files")
        else:
            print("  ❌ No test files found in dataset/Testing/Sounds/")

if __name__ == "__main__":
    main() 