import numpy as np
import pandas as pd # for csv
import librosa # for audio
import os

def load_data(audio_path, selection_table_path):
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # Load selection table (CSV exported from Raven Pro)
    selections = pd.read_csv(selection_table_path)

    return y, sr, selections

def slice_audio(y, sr, start_time, end_time):
    start_sample = int(start_time * sr)		# converts start time to a sample index
    end_sample = int(end_time * sr)		# comverts end time to sample index
    return y[start_sample:end_sample]		# slices original waveform(y) from start sample to end sample

def extract_features_from_clip(clip, sr):
    features = {}

    # MFCCs (take mean across time axis) // captures tone & timbre (shape of sound)
    mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=13) 	# extract 13 MFCC
								# 13 is standard; captures essentials, minimizes noise

    for i, coef in enumerate(mfcc):				# Loop thru each MFCC
       features[f'mfcc_{i+1}'] = np.mean(coef)			  # store avg val of each

    # Zero-crossing rate (ZCR) // how often signal crosses zero line (helps w/ volume and freq)
    zcr = librosa.feature.zero_crossing_rate(clip)		# gets ZRC
    features['zcr'] = np.mean(zcr)				# stores avg ZRC

    # Root Mean Square (RMS) energy (loudness)
    rms = librosa.feature.rms(clip)				# gets RMS
    features['rms'] = np.mean(rms)				# stores avg RMS

    # Spectral centroid (brightnes / how high-pitched)
    centroid = librosa.feature.spectral_centroid(y=clip, sr=sr) # gets spectral centroid
    features['centroid'] = np.mean(centroid)			# stores avg spectral centroid

    # return full dictionary for clip
    return features # MFCC 1-13, ZRC, RMS, spectral centroid

def build_feature_dataset(audio_path, selection_table_path, output_csv="features.csv"):
    y, sr, selections = load_data(audio_path, selection_table_path) # load waveform, sample rate, and selection table
    all_features = [] # Create empty list to store feature dict for each clip

    for _, row in selections.iterrows():
        start = row["Begin Time (s)"]
        end = row["End Time (s)"]
        label = row["Species"] if "Species" in row else row["Label"]  # adjust to match your file

        clip = slice_audio(y, sr, start, end)		# isolate audio segment
        features = extract_features_from_clip(clip, sr)	# extract acoustic features
        features["label"] = label			# add ground-truth label (gunshot or other)
        all_features.append(features)			# append feature dict to list

    df = pd.DataFrame(all_features)			# Convert to DataFrame
    df.to_csv(output_csv, index=False)			# Save to CSV
    print(f"Feature dataset saved to {output_csv}")

if __name__ == "__main__":
    # Command-line entry point
    import sys
    if len(sys.argv) != 3:
        print("Usage: python extract_features.py path_to_audio.wav path_to_selection_table.csv")
    else:
        build_feature_dataset(sys.argv[1], sys.argv[2])
