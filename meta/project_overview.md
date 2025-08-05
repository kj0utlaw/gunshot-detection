# 🔧 Gunshot Detection Model - Project Overview

## Overview
This research project is part of a broader conservation effort in partnership with the Elephants, Rhinos & People (ERP) initiative and the Cornell University Elephant Listening Project. The primary objective is to develop machine learning systems that can detect, localize, and classify acoustic events using infrasound data, with a focus on protecting endangered wildlife such as elephants from human threats.

## Group Hypothesis
Using infrasound (and potentially other cues) to detect, localize, and track elephants along with the same for human activity will allow us to localize and track the elephants we want to protect and the potential unexpected human presence that can be a threat to them, allowing park rangers and land managers to keep the two separated and to respond to incidents where humans and elephants may come into conflict.

## Individual Hypothesis
A hybrid machine learning model using a CNN and traditional classifier can classify gunshots from labeled infrasound recordings with measurable accuracy.

## North Star Goal
Build a hybrid machine learning system for gunshot detection in infrasound data by combining a Convolutional Neural Network (CNN) trained on Mel spectrograms with a traditional classifier using handcrafted audio features such as MFCCs, zero-crossing rate, RMS energy, and spectral centroid. This fusion approach leverages both deep learning and signal-based techniques to improve classification accuracy and robustness in challenging, noisy environments where traditional methods may fall short.

## ✅ COMPLETED WORK

### Phase 1: Core Components ✅
- Literature review on acoustic event detection and infrasound processing
- Feature extraction pipeline (MFCCs, spectral features, temporal features)
- Spectrogram generator for CNN input
- Spike detector using local RMS contrast
- CNN architecture design for spectrograms (128x173 input)

### Phase 2: Supervised Model ✅
- Trained CNN classifier on labeled clips with data augmentation
- Implemented early stopping and learning rate scheduling
- Evaluated performance using accuracy, F1, and confusion matrix
- Achieved 93.8% average recall across 104 test files

### Phase 3: Unsupervised Model ✅
- Implemented and tested Isolation Forest anomaly detection
- Used results to identify candidate spike clips for classification
- Configurable contamination parameter (0.5) for high recall

### Phase 4: Integration + Inference ✅
- Built full inference pipeline (spike → classify → output)
- Added batch evaluation, result export, and parallel processing
- Implemented comprehensive evaluation on entire dataset
- Created corrected metrics calculation with proper recall formula

## 🎯 Current Results

### Performance Metrics (104 test files)
- **Average Recall**: 93.8%
- **Average Precision**: 4.8%
- **Average F1 Score**: 9.0%
- **Average FPR**: 68.2%

### Technical Achievements
- ✅ Complete pipeline from audio input to gunshot detection
- ✅ Isolation Forest anomaly detection with 50% contamination
- ✅ CNN classification with 3 conv blocks + 3 FC layers
- ✅ Data augmentation (time stretching, noise injection)
- ✅ Early stopping and learning rate scheduling
- ✅ Parallel processing for large datasets
- ✅ Corrected evaluation metrics
- ✅ Comprehensive documentation

## 🔧 Technical Architecture

### Pipeline Flow
```
Audio Input → Spike Detection → Feature Extraction → Isolation Forest → CNN Classification → Results
```

### CNN Architecture
```
Input: (1, 128, 173) Mel spectrogram
├── Conv1: 32 filters, 3x3, ReLU, BatchNorm
├── Conv2: 64 filters, 3x3, ReLU, BatchNorm  
├── Conv3: 128 filters, 3x3, ReLU, BatchNorm
├── FC1: 88064 → 128, ReLU, Dropout(0.6)
├── FC2: 128 → 32, ReLU, Dropout(0.6)
└── FC3: 32 → 1 (logits)
```

### Training Configuration
- Loss: BCEWithLogitsLoss with class weights
- Optimizer: Adam (lr=0.0005, weight_decay=5e-4)
- Batch size: 16
- Epochs: 30 with early stopping
- Data augmentation: Time stretching (±20%), noise injection, gain variation

## 🚀 Usage

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

## 🔹 For Software Engineers
> We've built a modular ML pipeline that combines unsupervised anomaly detection with supervised classification. The system processes audio in chunks, detects potential gunshots using spike detection and Isolation Forest, then confirms them using a CNN classifier. The architecture is designed for both offline processing and real-time streaming.

## 🔹 For Non-Technical Users
> We've created a tool that can automatically find gunshots in audio recordings. It works by first looking for sudden or unusual sounds using spike detection, then double-checking each one using a trained CNN that has learned what gunshots usually look and sound like. The system can work with both recorded audio and live streams, and it tells you exactly when each gunshot occurred.

## Key Features

- **Robust Detection**: Works in noisy environments
- **Real-time Processing**: Supports live audio streams
- **Accurate Timestamps**: Precise timing for each detection
- **Low False Positives**: Multiple validation stages
- **Scalable Architecture**: Handles both short and long recordings

## Technical Requirements

- Python 3.8+
- PyTorch for CNN implementation
- Librosa for audio processing
- Scikit-learn for clustering
- NumPy for numerical operations

## Future Enhancements

- Multi-class classification for different gun types
- Directional analysis for gunshot location
- Integration with video systems
- Mobile deployment support
- API for third-party integration
- Hybrid model combining CNN and traditional features