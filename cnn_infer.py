import os
import torch
import librosa
import numpy as np
import argparse
from cnn_gunshot import GunshotCNN

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
N_TIME_FRAMES = 173  # adjust if needed to match training
CLIP_DIR = 'spikes'  # directory with clips from spike detection
CHECKPOINT_PATH = 'best_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(checkpoint_path):
    """
    Load the trained GunshotCNN model from checkpoint.
    """
    model = GunshotCNN(n_mels=N_MELS, n_time_frames=N_TIME_FRAMES, num_classes=1)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # Full training checkpoint with optimizer state, etc.
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Just model weights
        model.load_state_dict(checkpoint)
    
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_wav(wav_path):
    """
    Load a WAV file and convert to Mel spectrogram (dB), matching training preprocessing.
    Returns a torch tensor of shape (1, n_mels, n_time_frames).
    """
    audio, sr = librosa.load(wav_path, sr=4000)  # Force 4kHz to match dataset
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Pad or trim to fixed time frames
    if mel_spec_db.shape[1] < N_TIME_FRAMES:
        pad_width = N_TIME_FRAMES - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0,0),(0,pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :N_TIME_FRAMES]
    # Normalize (optional, match training)
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
    tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, n_time_frames)
    return tensor

def predict(model, tensor, threshold=0.8):
    """
    Run the model on a single Mel spectrogram tensor.
    Returns 1 for gunshot, 0 for not_gunshot.
    """
    tensor = tensor.unsqueeze(0).to(DEVICE)  # (1, 1, n_mels, n_time_frames)
    with torch.no_grad():
        output = model(tensor)
        # Use realistic threshold for better precision/recall balance
        logit_threshold = -22.0  # Balanced threshold
        prob = torch.sigmoid(output).item()
        # Higher logit = more likely gunshot
        pred = 1 if output.item() > logit_threshold else 0
    return pred, prob

def sanity_check_model(model):
    """
    Sanity check: test model on known samples to verify it's working correctly.
    """
    print("\n" + "="*50)
    print("SANITY CHECK - Testing Model on Known Samples")
    print("="*50)
    
    # Create dummy inputs
    dummy_gunshot = torch.randn(1, 1, N_MELS, N_TIME_FRAMES)
    dummy_not_gunshot = torch.randn(1, 1, N_MELS, N_TIME_FRAMES)
    
    # Test predictions
    model.eval()
    with torch.no_grad():
        # Test gunshot-like input
        output_gunshot = model(dummy_gunshot)
        prob_gunshot = torch.sigmoid(output_gunshot).item()
        
        # Test not-gunshot-like input  
        output_not_gunshot = model(dummy_not_gunshot)
        prob_not_gunshot = torch.sigmoid(output_not_gunshot).item()
    
    print(f"Gunshot-like input probability: {prob_gunshot:.4f}")
    print(f"Not-gunshot-like input probability: {prob_not_gunshot:.4f}")
    
    # Check if model is broken
    if prob_gunshot > 0.9 and prob_not_gunshot > 0.9:
        print("❌ MODEL IS BROKEN: Both inputs give >90% confidence")
        print("   This indicates the model is not learning discrimination")
        return False
    elif prob_gunshot < 0.1 and prob_not_gunshot < 0.1:
        print("❌ MODEL IS BROKEN: Both inputs give <10% confidence")
        print("   This indicates the model is not learning discrimination")
        return False
    else:
        print("✅ MODEL APPEARS TO BE LEARNING: Different probabilities for different inputs")
        return True

def main():
    """
    Run CNN inference on all clips in the specified directory.
    Outputs predictions for each clip.
    """
    parser = argparse.ArgumentParser(description="CNN inference for gunshot detection")
    parser.add_argument("--clip-dir", default="spikes",
                       help="Directory containing audio clips to process")
    parser.add_argument("--checkpoint", default="best_model.pth",
                       help="Path to trained model checkpoint")
    parser.add_argument("--output", default="cnn_infer_results.csv",
                       help="Output CSV file for results")
    parser.add_argument("--threshold", type=float, default=0.9,
                       help="Confidence threshold for gunshot detection (default: 0.9)")
    parser.add_argument("--threshold-percentile", type=float, default=None,
                       help="Use percentile-based threshold instead of fixed threshold (e.g., 90.0 for top 10%)")
    parser.add_argument("--sanity-check", action="store_true",
                       help="Run sanity check on model before inference")
    
    args = parser.parse_args()
    
    model = load_model(args.checkpoint)
    print(f"Loaded model from {args.checkpoint}")
    
    # Run sanity check if requested
    if args.sanity_check:
        sanity_check_model(model)
    
    if not os.path.exists(args.clip_dir):
        print(f"Error: Clip directory not found: {args.clip_dir}")
        return
    
    clip_files = [f for f in os.listdir(args.clip_dir) if f.endswith('.wav')]
    if not clip_files:
        print(f"No .wav files found in {args.clip_dir}")
        return
    
    print(f"Running inference on {len(clip_files)} clips from {args.clip_dir}...")
    results = []
    
    for fname in clip_files:
        path = os.path.join(args.clip_dir, fname)
        tensor = preprocess_wav(path)
        pred, prob = predict(model, tensor, args.threshold)
        label = 'gunshot' if pred == 1 else 'not_gunshot'
        print(f"{fname}: {label} (prob={prob:.2f})")
        results.append({'filename': fname, 'prediction': label, 'probability': prob})
    
    # Save results to CSV
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main() 