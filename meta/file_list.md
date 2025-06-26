# 🔧 Gunshot Detection Model - File Structure

## UNSUPERVISED PIPELINE

| File                        | Purpose                                                                                                    | Dependencies                       |
|-----------------------------|------------------------------------------------------------------------------------------------------------|------------------------------------|
| `spike_detector.py`         | Segments audio into high-energy windows using local RMS contrast                                           | `librosa` `numpy`                  |
| `unsupervised_model.py`     | Applies KMeans or anomaly algorithms to identify unusual segments from extracted features                  | `scikit-learn` `numpy`             |
| `feature_extractor.py`      | Extracts MFCCs, ZCR, RMS, and other handcrafted features from audio segments                               | `librosa` `numpy`                  |
| `spectrogram_generator.py`  | Generates Mel spectrograms for CNN input                                                                   | `librosa` `numpy`                  |

## SUPERVISED PIPELINE

| File                        | Purpose                                                                                                    | Dependencies                       |
|-----------------------------|------------------------------------------------------------------------------------------------------------|------------------------------------|
| `cnn_model.py`              | CNN to classify spectrogram or waveform segment as gunshot/non-gunshot                                     | `torch` `numpy`                    |
| `hybrid_model.py`           | Fuses CNN output, anomaly scores, and raw features for binary classification                               | `torch` `numpy` `scikit-learn`     |
| `predict_batch.py`          | Orchestrates the full pipeline: audio input→spike detection→feature extraction→classification→result output| All above modules                  |

## UTILITY FILES

| File                        | Purpose                                                                                                    | Dependencies                       |
|-----------------------------|------------------------------------------------------------------------------------------------------------|------------------------------------|
| `config.py`                 | Configuration settings and hyperparameters                                                                 | None                               |
| `utils.py`                  | Common utility functions and helpers                                                                       | `numpy` `librosa`                  |
| `data_loader.py`            | Data loading and preprocessing utilities                                                                   | `torch` `numpy` `librosa`          |
| `evaluation.py`             | Evaluation metrics and visualization tools                                                                 | `numpy` `matplotlib` `scikit-learn`|

## TEST FILES

| File                        | Purpose                                                                                                    | Dependencies                       |
|-----------------------------|------------------------------------------------------------------------------------------------------------|------------------------------------|
| `test_spike_detector.py`    | Unit tests for spike detection                                                                             | `pytest` `numpy`                   |
| `test_unsupervised_model.py`| Unit tests for anomaly detection                                                                           | `pytest` `numpy` `scikit-learn`    |
| `test_cnn_model.py`         | Unit tests for CNN classification                                                                          | `pytest` `torch` `numpy`           |
| `test_hybrid_model.py`      | Unit tests for hybrid model fusion                                                                         | `pytest` `torch` `numpy`           |

## DOCUMENTATION

| File                        | Purpose                                                                                                    | Dependencies                       |
|-----------------------------|------------------------------------------------------------------------------------------------------------|------------------------------------|
| `README.md`                 | Project overview and setup instructions                                                                    | None                               |
| `model_structure.md`        | Technical architecture and pipeline details                                                                | None                               |
| `api_docs.md`               | API documentation for each module                                                                          | None                               |
| `development_guide.md`      | Development guidelines and best practices                                                                  | None                               | 