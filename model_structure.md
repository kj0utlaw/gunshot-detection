# 🔧 Gunshot Detection Model - Technical Architecture (Updated)

## 1. Full Pipeline Overview

This system combines an unsupervised anomaly detector with a supervised binary classifier to identify gunshots in noisy acoustic environments. The pipeline supports arbitrarily long or live audio and automatically segments and analyzes potential gunshot events.

### Unsupervised Phase
- Purpose: Detect anomalous or high-interest segments in raw audio using clustering or spike-based logic.
- Methods:
  - Spike detection: Finds sudden acoustic events via local contrast on RMS energy.
  - Feature-based anomaly detection: Uses KMeans, Isolation Forest, or One-Class SVM on MFCCs and other handcrafted features to flag unusual patterns.

### Supervised Phase
- Purpose: Confirm whether each anomaly is a gunshot or not using a trained classifier.
- Methods:
  - Spectrogram CNN: Analyzes time–frequency visual patterns from Mel spectrograms.
  - Waveform CNN (optional): Learns temporal shapes directly from raw waveform spikes.
  - Hybrid fusion model: Combines CNN embeddings with handcrafted features and anomaly scores for final decision.

---

## 2. Architecture Flow

```
.wav file → Chunked/streamed audio
           ↓
       Spike Detector
           ↓
  Extract Features & Spectrograms
           ↓
  ┌─────────────┐     ┌────────────┐
  │Anomaly Model│ → → │ Classifier │
  └─────────────┘     └────────────┘
           ↓
  Predicted gunshot events w/ timestamps + probabilities
```

---

## 3. File Responsibilities (Updated)

These remain mostly consistent with the existing file list but with clearer separation of the two stages:

- `spike_detector.py`: Segments audio into high-energy windows using local RMS contrast
- `unsupervised_model.py`: Applies KMeans or anomaly algorithms to identify unusual segments from extracted features
- `cnn_model.py`: CNN to classify spectrogram or waveform segment as gunshot/non-gunshot
- `hybrid_model.py`: Fuses CNN output, anomaly scores, and raw features for binary classification
- `predict_batch.py`: Orchestrates the full pipeline: audio input → spike detection → feature extraction → classification → result output

---

## 4. Development Plan

### Phase 1: Core Components
- ✅ Feature extraction (MFCC, ZCR, RMS, centroid)
- ✅ Spectrogram generator
- ✅ Spike detector (local RMS contrast)
- ✅ CNN architecture design for spectrograms

### Phase 2: Unsupervised Model
- Implement and test KMeans, Isolation Forest, and One-Class SVM
- Use results to identify candidate spike clips for classification

### Phase 3: Supervised Model
- Train CNN and hybrid classifier on labeled clips
- Evaluate performance using accuracy, F1, and confusion matrix

### Phase 4: Integration + Inference
- Build full inference pipeline (spike → classify → output)
- Add batch evaluation, result export, and live stream support

---

## 5. Project Summary (For Different Audiences)

### 🔹 For a Software Engineer:
> We're building a modular ML pipeline that fuses unsupervised anomaly detection and supervised classification to detect gunshots in uncleaned acoustic data. It streams audio in chunks, flags high-contrast events via a spike detector, then uses clustering and feature extraction to isolate potential anomalies. These clips are then classified using a CNN and feature fusion model. The full system runs offline or live, and outputs structured predictions with timestamps and confidence.

### 🔹 For a Non-Technical Audience:
> We're developing a tool that listens to hours of sound recordings and automatically finds gunshots. It first scans for sudden or unusual noises, then double-checks each one using a trained model that has learned what gunshots usually look and sound like. It works even if the background noise is messy, and tells you exactly when each gunshot happened. 