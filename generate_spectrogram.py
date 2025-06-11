import numpy as np # numerical operations
import librosa # audio processing
import os # file paths
import pandas as pd # reading CSV files
from pathlib import Path # file paths

# Converts audio clip to Mel spectrogram (shows frequency content over time)
def generate_mel_spectrogram(audio_clip, sr, n_mels=64, n_fft=2048, hop_length=512):
    """
    Takes an audio clip and converts it to a Mel spectrogram.

    Inputs:
        audio_clip: The audio data
        sr: Sample rate of the audio
        n_mels: Number of frequency bands (default: 64)
        n_fft: Window size for analysis (default: 2048)
        hop_length: Step size between windows (default: 512)

    Returns:
        A Mel spectrogram showing how sound frequencies change over time
    """
    # Convert to Mel spectrogram (energy over time at different frequencies)
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

# Cleans audio by normalizing and removing silence
def preprocess_audio(audio_clip, sr):
    """
    Cleans up an audio clip by normalizing volume and removing silence.

    Inputs:
        audio_clip: The audio data to clean
        sr: Sample rate of the audio

    Returns:
        Cleaned audio clip
    """
    # Normalize and scale to [-1, 1]
    audio_norm = librosa.util.normalize(audio_clip)

    # Remove silence
    audio_trimmed, _ = librosa.effects.trim(audio_norm, top_db=20)

    return audio_trimmed

# Saves spectrogram as .npy file for fast loading
def save_spectrogram(spectrogram, output_path):
    """
    Saves a spectrogram to a file.

    Inputs:
        spectrogram: The spectrogram to save
        output_path: Where to save the file
    """
    # Save as .npy
    np.save(output_path, spectrogram)

# Main function: process audio file & generate spectrogram for each clip
def process_audio_file(audio_path, selection_table_path, output_dir="spectrograms"):
    """
    Takes an audio file and selection table, then creates spectrograms for each clip.

    Inputs:
        audio_path: Path to the audio file
        selection_table_path: Path to the CSV file with clip timestamps
        output_dir: Where to save the spectrograms (default: "spectrograms")
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
