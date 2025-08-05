# Gunshot Detection Pipeline

A comprehensive machine learning pipeline for detecting gunshots in audio recordings using Isolation Forest anomaly detection and Convolutional Neural Networks (CNN).

## 🐘 Background: The Poaching Crisis

- ~20,000 elephants and hundreds of rhinos are poached annually.
- Elephant populations declined from ~10 million (1900) to ~415,000 (today).
- Rhino populations dropped from ~500,000 (1900) to ~27,000 (today).
- In regions like Central Africa, annual income averages ~$1,700, while rhino horns sell for up to $300,000.
- Gunshots are a key acoustic signature of poaching events in protected wildlife habitats.
- Detecting these events quickly is critical to enable ranger intervention and save endangered species.

## 🎯 Overview

This project implements a multi-stage gunshot detection system:
1. **Spike Detection**: Identifies high-energy audio events
2. **Feature Extraction**: Extracts MFCC and spectral features from audio clips
3. **Isolation Forest**: Unsupervised anomaly detection to filter potential gunshots
4. **CNN Classification**: Deep learning model for binary classification (gunshot vs. non-gunshot)

## 📁 Project Structure

```
ugr/
├── data/                          # Audio data and processed features
│   ├── features/                  # Extracted features
│   ├── spectrograms/             # Generated spectrograms
│   └── wav_files/                # Original audio files
├── dataset/                       # Training and testing datasets
│   ├── Training/                 # Training data
│   └── Testing/                  # Testing data
├── checkpoints/                   # Model checkpoints
├── meta/                         # Documentation
├── spike_detection_results/       # Spike detection outputs
├── best_model.pth                # Trained CNN model
├── iso_forest_model.joblib       # Trained Isolation Forest model
└── [Scripts]                     # Core pipeline scripts
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision librosa scikit-learn pandas numpy matplotlib
```

### Model Files
The trained CNN model (`best_model.pth`) is not included in the repository due to size limitations. You can:
1. Train your own model using `train_cnn.py`
2. Download the pre-trained model from [link to be provided]
3. Contact the authors for model access

### Training Models

1. **Train Isolation Forest**:
```bash
python3 train_iso_forest.py
```

2. **Train CNN**:
```bash
python3 train_cnn.py
```

### Running Full Pipeline

**Test single file**:
```bash
python3 test_pipeline.py
```

**Test entire dataset**:
```bash
python3 full_dataset_test.py
```

## 📊 Core Components

### 1. Spike Detection (`spike_detector.py`)
- Identifies high-energy events in audio files
- Generates short audio clips for further processing
- Configurable energy threshold and clip duration

### 2. Feature Extraction (`extract_features.py`)
- Extracts MFCC features from audio clips
- Generates spectral features (centroid, rolloff, bandwidth)
- Outputs CSV with feature vectors

### 3. Isolation Forest (`iso_forest.py` / `train_iso_forest.py`)
- Unsupervised anomaly detection
- Filters potential gunshot events
- Configurable contamination parameter (0.0-0.5)
- Uses percentile-based thresholding for high recall

### 4. CNN Classification (`cnn_gunshot.py` / `cnn_infer.py`)
- Convolutional Neural Network for binary classification
- Input: Mel spectrograms (1 channel, 128x173)
- Architecture: 3 conv blocks + 3 FC layers
- Loss: BCEWithLogitsLoss with Adam optimizer
- Early stopping and learning rate scheduling
- Handles class imbalance using weighted loss function (BCEWithLogitsLoss with class weights)

## 🎛️ Configuration

### CNN Model Parameters
```python
# Architecture
input_channels = 1
num_classes = 1
dropout_rate = 0.6

# Training
learning_rate = 0.0005
weight_decay = 5e-4
batch_size = 16
num_epochs = 30
```

### Isolation Forest Parameters
```python
contamination = 0.5  # 50% anomalies
threshold_percentile = 90.0  # High recall threshold
```

## 📈 Performance Metrics

The pipeline evaluates performance using:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / Ground truth gunshots
- **F1 Score**: Harmonic mean of precision and recall
- **FPR**: False positive rate
- **Specificity**: True negatives / (True negatives + False positives)

### Recent Results (104 test files)
- **Average Recall**: 93.8%
- **Average Precision**: 4.8%
- **Average F1 Score**: 9.0%
- **Average FPR**: 68.2%

## 🔧 Key Scripts

### Training Scripts
- `train_iso_forest.py`: Trains Isolation Forest on training data
- `train_cnn.py`: Trains CNN with data augmentation and early stopping

### Inference Scripts
- `cnn_infer.py`: CNN inference with configurable threshold
- `iso_forest.py`: Isolation Forest inference and scoring

### Evaluation Scripts
- `full_dataset_test.py`: Comprehensive evaluation on all test files
- `fix_recall.py`: Corrects recall calculation in results
- `test_pipeline.py`: Single file pipeline testing

### Utility Scripts
- `extract_features.py`: Feature extraction from audio clips
- `data_loader.py`: PyTorch data loading and augmentation
- `comprehensive_test.py`: Parallel processing evaluation

## 🐛 Troubleshooting

### Common Issues

1. **CNN always predicts 100% confidence**:
   - Check sample rate matching (should be 4000 Hz)
   - Verify model architecture matches saved weights
   - Adjust logit threshold in `cnn_infer.py`

2. **Low recall values**:
   - Increase Isolation Forest contamination
   - Lower CNN threshold for more sensitive detection
   - Check ground truth loading and timestamp matching

3. **High disk usage**:
   - Use `--max-clips` to limit processing
   - Enable caching with `--skip-features`
   - Clean up temporary directories

### Debug Scripts
- `debug_cnn.py`: Analyze CNN predictions and confidence
- `debug_timestamps.py`: Check timestamp matching
- `simple_test.py`: Quick single-file testing

## 📝 Data Format

### Input Audio
- Format: WAV files
- Sample rate: 4000 Hz
- Duration: Variable (typically hours)

### Ground Truth
- Format: Tab-separated CSV
- Columns: start_time, end_time, filename, tag
- Tag values: 'gun' for gunshots

### Output Results
- Format: CSV with performance metrics
- Columns: filename, Ground Truth, Estimated TP, Recall, Precision, etc.

## 🔬 Advanced Usage

### Threshold Experimentation
```python
# In full_dataset_test.py
cnn_threshold = 0.9  # Try: 0.5, 0.8, 0.9, 0.95
```

### Parallel Processing
```bash
python3 comprehensive_test.py --num-workers 4 --batch-size 50
```

### Custom Evaluation
```python
# Load and analyze results
import pandas as pd
results = pd.read_csv('corrected_results.csv')
print(f"Average Recall: {results['Recall'].mean():.3f}")
```

## 📚 Technical Details

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

### Feature Extraction
- **MFCC**: 13 coefficients
- **Spectral features**: centroid, rolloff, bandwidth, contrast
- **Temporal features**: zero-crossing rate, energy

### Data Augmentation
- Time stretching (±20%)
- Noise injection
- Gain variation (±6dB)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset provided by Cornell University Elephant Listening Project, collected from Nouabalé-Ndoki and Dzanga-Sangha National Parks.
- PyTorch for deep learning framework
- Librosa for audio processing
- Scikit-learn for machine learning utilities

## 🧠 Project Context

This work was conducted as part of the Chico STEM Connections Collaborative (CSC²) Undergraduate Research Program at California State University, Chico, in partnership with the Cornell University Elephant Listening Project. While each student pursued their own focus, this project specifically targeted automated detection of human threats (gunshots) using acoustic data.

---

**Last Updated**: December 2024  
**Version**: 1.0.0 