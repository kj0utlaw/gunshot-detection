# 🔧 Gunshot Detection Model - File Structure

## COMPLETED COMPONENTS ✅

| File                        | Purpose                                                                                                    | Dependencies                       |
|-----------------------------|------------------------------------------------------------------------------------------------------------|------------------------------------|
| `spike_detector.py`         | Segments audio into high-energy windows using local RMS contrast                                           | `librosa` `numpy`                  |
| `extract_features.py`       | Extracts MFCCs, ZCR, RMS, and other handcrafted features from audio segments                               | `librosa` `numpy`                  |
| `generate_spectrogram.py`   | Generates Mel spectrograms for CNN input                                                                   | `librosa` `numpy`                  |
| `extract_and_index_clips.py`| Creates balanced dataset from gunshot training data                                                        | `pydub` `pandas`                   |

## CNN PIPELINE (IN PROGRESS) 🚧

| File                        | Purpose                                                                                                    | Dependencies                       |
|-----------------------------|------------------------------------------------------------------------------------------------------------|------------------------------------|
| `data_loader.py`            | Loads .wav clips, converts to Mel spectrograms, handles batching and augmentation                         | `torch` `librosa` `numpy`          |
| `cnn_model.py`              | CNN architecture for binary classification (gunshot/non-gunshot)                                          | `torch` `numpy`                    |
| `train_cnn.py`              | Training script with TPU optimization and checkpointing                                                    | `torch` `numpy` `scikit-learn`     |
| `evaluate_cnn.py`           | Evaluation metrics, confusion matrix, ROC curves, 5-fold CV                                               | `torch` `numpy` `matplotlib`       |

## FUTURE COMPONENTS ❌

| File                        | Purpose                                                                                                    | Dependencies                       |
|-----------------------------|------------------------------------------------------------------------------------------------------------|------------------------------------|
| `unsupervised_model.py`     | Applies KMeans or anomaly algorithms to identify unusual segments from extracted features                  | `scikit-learn` `numpy`             |
| `hybrid_model.py`           | Fuses CNN output, anomaly scores, and raw features for binary classification                               | `torch` `numpy` `scikit-learn`     |
| `predict_batch.py`          | Orchestrates the full pipeline: audio input→spike detection→feature extraction→classification→result output| All above modules                  |

## UTILITY FILES

| File                        | Purpose                                                                                                    | Dependencies                       |
|-----------------------------|------------------------------------------------------------------------------------------------------------|------------------------------------|
| `config.py`                 | Configuration settings and hyperparameters                                                                 | None                               |
| `utils.py`                  | Common utility functions and helpers                                                                       | `numpy` `librosa`                  |
| `evaluation.py`             | Evaluation metrics and visualization tools                                                                 | `numpy` `matplotlib` `scikit-learn`|

## TEST FILES

| File                        | Purpose                                                                                                    | Dependencies                       |
|-----------------------------|------------------------------------------------------------------------------------------------------------|------------------------------------|
| `test_spike_detector.py`    | Unit tests for spike detection                                                                             | `pytest` `numpy`                   |
| `test_cnn_model.py`         | Unit tests for CNN classification                                                                          | `pytest` `torch` `numpy`           |
| `test_data_loader.py`       | Unit tests for data loading and preprocessing                                                              | `pytest` `torch` `numpy`           |

## DOCUMENTATION

| File                        | Purpose                                                                                                    | Dependencies                       |
|-----------------------------|------------------------------------------------------------------------------------------------------------|------------------------------------|
| `README.md`                 | Project overview and setup instructions                                                                    | None                               |
| `model_structure.md`        | Technical architecture and pipeline details                                                                | None                               |
| `api_docs.md`               | API documentation for each module                                                                          | None                               |
| `development_guide.md`      | Development guidelines and best practices                                                                  | None                               | 