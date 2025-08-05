# 🔧 Gunshot Detection Model - Technical Architecture (Updated)

## Full Pipeline Overview

This system combines an unsupervised anomaly detector with a supervised binary classifier to identify gunshots in noisy acoustic environments. The pipeline supports arbitrarily long or live audio and automatically segments and analyzes potential gunshot events.

### Unsupervised Phase
- Purpose: Detect anomalous or high-interest segments in raw audio using Isolation Forest.
- Methods:
  - Spike detection: Finds sudden acoustic events via local contrast on RMS energy.
  - Isolation Forest: Uses unsupervised anomaly detection on MFCCs and spectral features to flag unusual patterns.

### Supervised Phase
- Purpose: Confirm whether each anomaly is a gunshot or not using a trained CNN classifier.
- Methods:
  - Spectrogram CNN: Analyzes time–frequency visual patterns from Mel spectrograms (128x173).
  - Architecture: 3 conv blocks + 3 FC layers with dropout and batch normalization.

## Architecture Flow
```
.wav file → Chunked/streamed audio
           ↓
       Spike Detection
           ↓
  Extract Features & Spectrograms
           ↓
  ┌─────────────┐     ┌────────────┐
  │Isolation    │ → → │ CNN        │
  │Forest       │     │ Classifier │
  └─────────────┘     └────────────┘
           ↓
  Predicted gunshot events w/ timestamps + probabilities
```

## Implemented Components

### Core Pipeline Files
- `spike_detector.py`: Segments audio into high-energy windows using local RMS contrast
- `extract_features.py`: Extracts MFCCs, spectral features, and temporal features from audio segments
- `iso_forest.py`: Isolation Forest anomaly detection with configurable contamination
- `train_iso_forest.py`: Training script for Isolation Forest model
- `cnn_gunshot.py`: CNN architecture for binary classification (gunshot/non-gunshot)
- `train_cnn.py`: Training script with early stopping and learning rate scheduling
- `cnn_infer.py`: CNN inference with configurable threshold

### Data Processing
- `data_loader.py`: PyTorch data loading with augmentation (time stretching, noise injection)
- `generate_spectrogram.py`: Mel spectrogram generation for CNN input

### Evaluation & Testing
- `full_dataset_test.py`: Comprehensive evaluation on all test files
- `test_pipeline.py`: Single file pipeline testing
- `fix_recall.py`: Corrects recall calculation in results
- `comprehensive_test.py`: Parallel processing evaluation

### Utility Files
- `meta/`: Documentation and project structure
- `corrected_results.csv`: Final evaluation results
- `full_dataset_results.csv`: Detailed performance metrics

## CNN Architecture Details

```
Input: (1, 128, 173) Mel spectrogram
├── Conv1: 32 filters, 3x3, ReLU, BatchNorm
├── Conv2: 64 filters, 3x3, ReLU, BatchNorm  
├── Conv3: 128 filters, 3x3, ReLU, BatchNorm
├── FC1: 88064 → 128, ReLU, Dropout(0.6)
├── FC2: 128 → 32, ReLU, Dropout(0.6)
└── FC3: 32 → 1 (logits)
```

## Training Configuration

### CNN Training
- Loss: BCEWithLogitsLoss with class weights
- Optimizer: Adam (lr=0.0005, weight_decay=5e-4)
- Batch size: 16
- Epochs: 30 with early stopping
- Data augmentation: Time stretching (±20%), noise injection, gain variation

### Isolation Forest
- Contamination: 0.5 (50% anomalies)
- Threshold percentile: 90.0 for high recall
- Features: MFCCs, spectral centroid, rolloff, bandwidth, contrast

## Performance Metrics

Recent results on 104 test files:
- **Average Recall**: 93.8%
- **Average Precision**: 4.8%
- **Average F1 Score**: 9.0%
- **Average FPR**: 68.2%

## Usage

### Training Models
```bash
# Train Isolation Forest
python3 train_iso_forest.py

# Train CNN
python3 train_cnn.py
```

### Evaluation
```bash
# Test single file
python3 test_pipeline.py

# Test entire dataset
python3 full_dataset_test.py
```

---

## Project Summary

> We've developed a tool that listens to hours of sound recordings and automatically finds gunshots. It first scans for sudden or unusual noises using spike detection and Isolation Forest, then double-checks each one using a trained CNN that has learned what gunshots usually look and sound like. The system works in noisy environments and provides precise timestamps for each detection. 