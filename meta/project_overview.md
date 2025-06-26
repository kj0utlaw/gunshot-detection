# 🔧 Gunshot Detection Model - Project Overview

## Overview
This research project is part of a broader conservation effort in partnership with the Elephants, Rhinos & People (ERP) initiative. The primary objective is to develop machine learning systems that can detect, localize, and classify acoustic events using infrasound data, with a focus on protecting endangered wildlife such as elephants from human threats.

## Group Hypothesis
Using infrasound (and potentially other cues) to detect, localize, and track elephants along with the same for human activity will allow us to localize and track the elephants we want to protect and the potential unexpected human presence that can be a threat to them, allowing park rangers and land managers to keep the two separated and to respond to incidents where humans and elephants may come into conflict.

## Individual Hypothesis
A hybrid machine learning model using a CNN and traditional classifier can classify gunshots from labeled infrasound recordings with measurable accuracy.

## North Star Goal
Build a hybrid machine learning system for gunshot detection in infrasound data by combining a Convolutional Neural Network (CNN) trained on Mel spectrograms with a traditional classifier using handcrafted audio features such as MFCCs, zero-crossing rate, RMS energy, and spectral centroid. This fusion approach leverages both deep learning and signal-based techniques to improve classification accuracy and robustness in challenging, noisy environments where traditional methods may fall short.

## Experimental Plan (8 Weeks)

**Week 1:**
- Literature review on acoustic event detection and infrasound processing
- Investigate how acoustic events are identified and classified in noisy environments
- Explore what characteristics are typically extracted to enable classification
- Learn to use audio processing tools (e.g., librosa, Raven Pro)
- Brush up on Python and get comfortable with the tools and syntax
- Set up project infrastructure and review dataset structure

**Week 2:**
- Begin research into machine learning and deep learning fundamentals
- Study acoustic classification models
- Experiment with handcrafted feature extraction (MFCCs, ZCR, etc.)
- Organize and validate early feature outputs
- Explore ERP/ELP tools and data formats
- Use Edge TPU for small-scale training and practice workflows

**Week 3:**
- Finalize feature extraction pipeline
- Generate Mel spectrograms from labeled audio
- Train traditional ML classifier on extracted features
- Implement cross-validation to evaluate classifier and monitor overfitting
- Plan comparative analysis using confusion matrices

**Week 4:**
- Train CNN on Mel spectrograms
- Tune CNN architecture and data formatting (e.g., augmentation)
- Compare CNN and traditional classifier performance
- Use confusion matrices for model interpretation
- Begin transitioning to NFS for larger training workloads

**Week 5:**
- Design and build late-fusion architecture
- Combine CNN and traditional classifier into unified pipeline
- Train hybrid model on shared validation/test sets
- Log detailed performance metrics

**Week 6:**
- Perform hyperparameter tuning on all models
- Investigate additional ensemble techniques
- Conduct ablation studies on features/components
- Prepare metric visualizations

**Week 7:**
- Evaluate final model using precision, recall, accuracy, F1 score, confusion matrix
- Run hybrid model on test set
- Perform failure analysis and identify edge cases

**Week 8:**
- Freeze and document best-performing model
- Finalize visualizations and results
- Prepare presentation, slides, and code handoff
- Reflect on limitations and future directions
- Write final report

### 🔹 For Software Engineers
> We're building a modular ML pipeline that combines unsupervised anomaly detection with supervised classification. The system processes audio in chunks, detects potential gunshots using spike detection and clustering, then confirms them using a CNN classifier. The architecture is designed for both offline processing and real-time streaming.

### 🔹 For Non-Technical Users
> We're creating a tool that can automatically find gunshots in audio recordings. It works by first looking for sudden or unusual sounds, then double-checking each one to make sure it's actually a gunshot. The system can work with both recorded audio and live streams, and it tells you exactly when each gunshot occurred.

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