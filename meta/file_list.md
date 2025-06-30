# Gunshot Detection Model - File Structure

## COMPLETED COMPONENTS

- `spike_detector.py` - Segments audio into high-energy windows using local RMS contrast
- `extract_features.py` - Extracts MFCCs, ZCR, RMS, and other handcrafted features from audio segments
- `generate_spectrogram.py` - Generates Mel spectrograms for CNN input
- `extract_and_index_clips.py` - Creates balanced dataset from gunshot training data

## CNN PIPELINE (IN PROGRESS)

- `data_loader.py` - Loads .wav clips, converts to Mel spectrograms, handles batching and augmentation
- `cnn_gunshot.py` - CNN architecture for binary classification (gunshot/non-gunshot)
- `train_cnn.py` - Training script with checkpointing and learning rate scheduling
- `evaluate_cnn.py` - Evaluation metrics, confusion matrix, ROC curves, 5-fold CV

## FUTURE COMPONENTS

- `unsupervised_model.py` - Applies KMeans or anomaly algorithms to identify unusual segments
- `hybrid_model.py` - Fuses CNN output, anomaly scores, and raw features for classification
- `predict_batch.py` - Orchestrates the full pipeline: audio input→spike detection→feature extraction→classification

## UTILITY FILES

- `config.py` - Configuration settings and hyperparameters
- `utils.py` - Common utility functions and helpers
- `evaluation.py` - Evaluation metrics and visualization tools

## TEST FILES

- `test_spike_detector.py` - Unit tests for spike detection
- `test_cnn_model.py` - Unit tests for CNN classification
- `test_data_loader.py` - Unit tests for data loading and preprocessing

## DOCUMENTATION

- `README.md` - Project overview and setup instructions
- `model_structure.md` - Technical architecture and pipeline details
- `api_docs.md` - API documentation for each module
- `development_guide.md` - Development guidelines and best practices 