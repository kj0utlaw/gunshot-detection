import numpy as np # numerical operations
import librosa # audio processing
import os # file paths
import pandas as pd # reading CSV files
from pathlib import Path # file paths

# Convert audio clip to Mel spectrogram (freq content over time)
def generate_mel_spectrogram(audio_clip, sr, n_mels=64, n_fft=2048, hop_length=512):
    """
    Convert audio clip to Mel spectrogram

    Inputs:
        audio_clip: audio data
        sr: sample rate
        n_mels: freq bands (default: 64)
        n_fft: window size (default: 2048)
        hop_length: step size (default: 512)

    Returns:
        Mel spectrogram (freq vs time)
    """
    # Generate Mel spectrogram (energy over time at diff freqs)
    mel_spec = librosa.feature.melspectrogram(
        y=audio_clip,
        sr=sr,
        n_mels=n_mels,        # 64 bands
        n_fft=n_fft,          # 2048 samples = ~46ms
        hop_length=hop_length  # 512 samples = ~11.6ms
    )

    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db

# Clean audio (normalize + remove silence)
def preprocess_audio(audio_clip, sr):
    """
    Clean audio clip (normalize + trim silence)

    Inputs:
        audio_clip: audio to clean
        sr: sample rate

    Returns:
        Cleaned audio clip
    """
    # Normalize to [-1, 1]
    audio_norm = librosa.util.normalize(audio_clip)

    # Remove silence
    audio_trimmed, _ = librosa.effects.trim(audio_norm, top_db=20)

    return audio_trimmed

# Save spectrogram as .npy file
def save_spectrogram(spectrogram, output_path):
    """
    Save spectrogram to file

    Inputs:
        spectrogram: spectrogram to save
        output_path: save location
    """
    # Save as .npy
    np.save(output_path, spectrogram)

# Main: process audio file & generate spectrograms for each clip
def process_audio_file(audio_path, selection_table_path, output_dir="spectrograms"):
    """
    Process audio file + selection table, create spectrograms for each clip

    Inputs:
        audio_path: path to audio file
        selection_table_path: path to CSV with clip timestamps
        output_dir: save location (default: "spectrograms")
    """
    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    # Load audio and selection table
    y, sr = librosa.load(audio_path, sr=None)  # keep original sample rate
    selections = pd.read_csv(selection_table_path)

    # Process each clip
    for idx, row in selections.iterrows():
        # Get clip info
        start = row["Begin Time (s)"]
        end = row["End Time (s)"]
        label = row["Species"] if "Species" in row else row["Label"]

        # Cut clip from audio
        start_sample = int(start * sr)  # convert time to samples
        end_sample = int(end * sr)
        clip = y[start_sample:end_sample]

        # Clean and convert
        clip_processed = preprocess_audio(clip, sr)
        mel_spec = generate_mel_spectrogram(clip_processed, sr)

        # Create filename with label and index
        filename = f"{label}_{idx:04d}.npy"
        output_path = os.path.join(output_dir, filename)
        save_spectrogram(mel_spec, output_path)

        print(f"Processed clip {idx+1}/{len(selections)}: {filename}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python generate_spectrogram.py path_to_audio.wav path_to_selection_table.csv")
    else:
        process_audio_file(sys.argv[1], sys.argv[2])
