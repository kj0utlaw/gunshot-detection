"""
This file is mostly one time use to prepare clips for training.
It extracts gunshot sound clips from the provided datasets, ensuring a balanced set of positive and negative samples
"""
import os
import csv
import random
from pydub import AudioSegment
import pandas as pd
from tqdm import tqdm

# --- CONFIGURABLE PARAMETERS ---
# Input paths
ECOGUNS_ANNOTATION = 'gunshots/Training/ecoguns/Guns_Training_ecoGuns_SST.txt'
ECOGUNS_SOUNDS = 'gunshots/Training/ecoguns/Sounds'
PNNN_ANNOTATION = 'gunshots/Training/pnnn_dep1-7/nn_Grid50_guns_dep1-7_Guns_Training.txt'
PNNN_SOUNDS = 'gunshots/Training/pnnn_dep1-7/Sounds'
PNNN_CLIPS = 'gunshots/Training/pnnn_dep1-7/Sound_Clips'

# Output paths
OUTPUT_CLIPS = 'dataset/clips'
OUTPUT_CSV = 'dataset/clip_index.csv'

# Clip duration (seconds)
MIN_CLIP_SEC = 4
MAX_CLIP_SEC = 6
NEG_PER_POS = 1  # Number of negatives per positive

# --- UTILITY FUNCTIONS ---
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_random_segment(audio, duration_ms, avoid_segments=None):
    """Get a random segment from audio, avoiding overlap with avoid_segments (list of (start, end) in ms)."""
    audio_len = len(audio)
    tries = 0
    while tries < 100:
        start = random.randint(0, audio_len - duration_ms)
        end = start + duration_ms
        if avoid_segments:
            overlap = any(max(start, s) < min(end, e) for s, e in avoid_segments)
            if overlap:
                tries += 1
                continue
        return audio[start:end]
    return None  # Could not find a non-overlapping segment

def parse_annotation_file(path, file_col, start_col, end_col):
    """Parse annotation file, return list of (filename, start_sec, end_sec)."""
    events = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            try:
                fname = row[file_col].strip()
                start = float(row[start_col])
                end = float(row[end_col])
                events.append((fname, start, end))
            except Exception:
                continue
    return events

def extract_clip(audio_path, start_sec, end_sec, out_path):
    audio = AudioSegment.from_wav(audio_path)
    start_ms = int(start_sec * 1000)
    end_ms = int(end_sec * 1000)
    clip = audio[start_ms:end_ms]
    # Adjust to 4-6s
    clip_len = len(clip)
    if clip_len < MIN_CLIP_SEC * 1000:
        # Pad with silence
        pad_ms = MIN_CLIP_SEC * 1000 - clip_len
        clip += AudioSegment.silent(duration=pad_ms)
    elif clip_len > MAX_CLIP_SEC * 1000:
        # Truncate
        clip = clip[:MAX_CLIP_SEC * 1000]
    clip.export(out_path, format='wav')
    return out_path

def copy_and_index_sound_clips(src_dir, out_dir, index_rows):
    for fname in tqdm(os.listdir(src_dir), desc='Copying pre-extracted positives'):
        if fname.endswith('.wav'):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(out_dir, f'pos_{fname}')
            AudioSegment.from_wav(src).export(dst, format='wav')
            index_rows.append({'filename': os.path.basename(dst), 'label': 'gunshot'})

def extract_positives(events, sounds_dir, out_dir, index_rows, prefix):
    for i, (fname, start, end) in enumerate(tqdm(events, desc=f'Extracting {prefix} positives')):
        audio_path = os.path.join(sounds_dir, fname)
        if not os.path.exists(audio_path):
            continue
        # Center the clip if too long/short
        duration = end - start
        target_len = min(max(duration, MIN_CLIP_SEC), MAX_CLIP_SEC)
        mid = (start + end) / 2
        start_adj = max(0, mid - target_len / 2)
        end_adj = start_adj + target_len
        out_name = f'{prefix}_pos_{i:05d}.wav'
        out_path = os.path.join(out_dir, out_name)
        extract_clip(audio_path, start_adj, end_adj, out_path)
        index_rows.append({'filename': out_name, 'label': 'gunshot'})

def extract_negatives(events_by_file, sounds_dir, out_dir, index_rows, prefix):
    for fname, pos_events in tqdm(events_by_file.items(), desc=f'Extracting {prefix} negatives'):
        audio_path = os.path.join(sounds_dir, fname)
        if not os.path.exists(audio_path):
            continue
        audio = AudioSegment.from_wav(audio_path)
        avoid = [(int(s*1000), int(e*1000)) for s, e in pos_events]
        for i in range(len(pos_events) * NEG_PER_POS):
            dur_ms = random.randint(MIN_CLIP_SEC*1000, MAX_CLIP_SEC*1000)
            seg = get_random_segment(audio, dur_ms, avoid_segments=avoid)
            if seg is not None:
                out_name = f'{prefix}_neg_{i:05d}.wav'
                out_path = os.path.join(out_dir, out_name)
                seg.export(out_path, format='wav')
                index_rows.append({'filename': out_name, 'label': 'not_gunshot'})

# --- MAIN SCRIPT ---
def main():
    ensure_dir(OUTPUT_CLIPS)
    index_rows = []

    # 1. Copy pre-extracted positives
    copy_and_index_sound_clips(PNNN_CLIPS, OUTPUT_CLIPS, index_rows)

    # 2. Extract positives from annotation files
    eco_events = parse_annotation_file(ECOGUNS_ANNOTATION, 'Begin File', 'begin time', 'end time')
    pnnn_events = parse_annotation_file(PNNN_ANNOTATION, 'Begin File', 'Begin Time (s)', 'End Time (s)')
    extract_positives(eco_events, ECOGUNS_SOUNDS, OUTPUT_CLIPS, index_rows, 'eco')
    extract_positives(pnnn_events, PNNN_SOUNDS, OUTPUT_CLIPS, index_rows, 'pnnn')

    # 3. Extract negatives
    # Group events by file for negative sampling
    eco_by_file = {}
    for fname, start, end in eco_events:
        eco_by_file.setdefault(fname, []).append((start, end))
    pnnn_by_file = {}
    for fname, start, end in pnnn_events:
        pnnn_by_file.setdefault(fname, []).append((start, end))
    extract_negatives(eco_by_file, ECOGUNS_SOUNDS, OUTPUT_CLIPS, index_rows, 'eco')
    extract_negatives(pnnn_by_file, PNNN_SOUNDS, OUTPUT_CLIPS, index_rows, 'pnnn')

    # 4. Write CSV index
    ensure_dir(os.path.dirname(OUTPUT_CSV))
    pd.DataFrame(index_rows).to_csv(OUTPUT_CSV, index=False)
    print(f'Dataset index written to {OUTPUT_CSV}')

if __name__ == '__main__':
    main() 