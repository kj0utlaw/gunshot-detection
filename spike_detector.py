import numpy as np
import librosa
import soundfile as sf
from typing import List, Tuple, Optional, Generator, Dict, Any
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import os
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import json
from datetime import datetime
from scipy import signal

# Set up logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SpikeEvent:
    """Represents a detected spike event with its metadata."""
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds
    peak_time: float   # Time of peak amplitude
    peak_amplitude: float  # Peak amplitude value
    chunk_index: int   # Index of the chunk where spike was found
    local_index: int   # Index within the chunk
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

class StreamingSpikeDetector:
    def __init__(self, save_clips: bool = False, chunk_duration: float = 10.0):
        """
        Initialize streaming spike detector with configurable parameters.
        
        Args:
            save_clips: Whether to save individual audio clips (default: False)
            chunk_duration: Duration of each processing chunk in seconds (default: 10.0)
        """
        # Fixed parameters
        self.sample_rate = 44100
        self.clip_duration = 0.5  # 500ms clips
        self.chunk_duration = chunk_duration  # Reduced from 60.0 to 10.0 seconds
        self.rms_window = int(0.02 * self.sample_rate)  # 20ms RMS window
        self.smooth_window = int(0.5 * self.sample_rate)  # 500ms smoothing window
        
        # Derived parameters
        self.clip_samples = int(self.clip_duration * self.sample_rate)
        self.chunk_samples = int(self.chunk_duration * self.sample_rate)
        
        # Buffer for handling spikes that cross chunk boundaries
        self.buffer = np.array([])
        self.buffer_duration = self.clip_duration
        
        # Output configuration
        self.save_clips = save_clips
        if self.save_clips:
            self.output_dir = Path("detected_clips")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _compute_rms_vectorized(self, audio: np.ndarray) -> np.ndarray:
        """Compute short-time RMS energy using vectorized operations."""
        try:
            # Ensure audio is float32
            audio = audio.astype(np.float32)
            
            # Pad audio to handle edge cases
            padded = np.pad(audio, (self.rms_window//2, self.rms_window//2), mode='edge')
            
            # Use convolution for efficient RMS computation
            squared = padded ** 2
            rms = np.sqrt(np.convolve(squared, np.ones(self.rms_window)/self.rms_window, mode='valid'))
            
            return rms.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error in RMS computation: {str(e)}")
            return np.array([], dtype=np.float32)
    
    def _compute_baseline_vectorized(self, rms: np.ndarray) -> np.ndarray:
        """Compute smoothed baseline using vectorized operations."""
        try:
            # Ensure RMS is float32
            rms = rms.astype(np.float32)
            
            # Pad RMS for edge handling
            padded = np.pad(rms, (self.smooth_window//2, self.smooth_window//2), mode='edge')
            
            # Use convolution for efficient moving average
            baseline = np.convolve(padded, np.ones(self.smooth_window)/self.smooth_window, mode='valid')
            
            return baseline.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error in baseline computation: {str(e)}")
            return np.array([], dtype=np.float32)
    
    def _detect_spikes_vectorized(self, audio: np.ndarray) -> List[int]:
        """
        Detect spikes using vectorized contrast-based method.
        
        Args:
            audio: Audio chunk array
            
        Returns:
            List of sample indices where spikes occur
        """
        try:
            # Ensure audio is float32
            audio = audio.astype(np.float32)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Compute RMS energy and baseline using vectorized operations
            rms = self._compute_rms_vectorized(audio)
            baseline = self._compute_baseline_vectorized(rms)
            
            # Ensure shapes match
            min_len = min(len(rms), len(baseline))
            rms = rms[:min_len]
            baseline = baseline[:min_len]
            
            # Compute contrast
            contrast = rms - baseline
            
            # Clean NaNs
            contrast[np.isnan(contrast)] = 0
            
            # Find peaks using scipy.signal.find_peaks with more sensitive criteria
            peaks, _ = signal.find_peaks(
                contrast,
                height=0.01,  # Lower height threshold for better sensitivity
                distance=int(0.05 * self.sample_rate),  # Shorter minimum distance (50ms)
                prominence=0.005  # Lower prominence threshold
            )
            
            return peaks.tolist()
            
        except Exception as e:
            logger.error(f"Error in spike detection: {str(e)}")
            return []
    
    def _get_chunk_generator(self, audio_path: str) -> Generator[np.ndarray, None, None]:
        """Create a generator that yields audio chunks."""
        logger.info(f"Starting to process file: {audio_path}")
        
        try:
            # Get file info
            info = sf.info(audio_path)
            duration = info.duration
            logger.info(f"Audio duration: {duration:.2f} seconds")
            
            # Calculate number of samples per chunk
            samples_per_chunk = int(self.chunk_duration * info.samplerate)
            
            # Open file for streaming
            with sf.SoundFile(audio_path) as audio_file:
                chunk_count = 0
                while True:
                    # Read chunk
                    chunk = audio_file.read(samples_per_chunk)
                    if len(chunk) == 0:  # End of file
                        break
                        
                    chunk_count += 1
                    logger.debug(f"Processing chunk {chunk_count}, shape: {chunk.shape}")
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Error in chunk generator: {str(e)}")
            raise
    
    def _process_chunk(self, 
                      chunk: np.ndarray, 
                      chunk_index: int,
                      chunk_start_time: float) -> List[SpikeEvent]:
        """Process a single chunk of audio."""
        try:
            # Combine buffer with current chunk
            if len(self.buffer) > 0:
                chunk = np.concatenate([self.buffer, chunk])
            
            # Detect spikes using vectorized method
            peaks = self._detect_spikes_vectorized(chunk)
            
            # Create spike events
            events = []
            for peak in peaks:
                peak_time = chunk_start_time + (peak / self.sample_rate)
                start_time = max(0, peak_time - self.clip_duration/2)
                end_time = peak_time + self.clip_duration/2
                
                events.append(SpikeEvent(
                    start_time=start_time,
                    end_time=end_time,
                    peak_time=peak_time,
                    peak_amplitude=chunk[peak],
                    chunk_index=chunk_index,
                    local_index=peak
                ))
            
            # Update buffer
            buffer_samples = int(self.buffer_duration * self.sample_rate)
            self.buffer = chunk[-buffer_samples:] if len(chunk) > buffer_samples else chunk
            
            return events
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_index}: {str(e)}")
            return []
    
    def process_file(self, audio_path: str) -> List[SpikeEvent]:
        """
        Process an audio file in chunks.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of all detected spike events
        """
        try:
            # Get total duration for progress bar
            info = sf.info(audio_path)
            duration = info.duration
            total_chunks = int(np.ceil(duration / self.chunk_duration))
            logger.info(f"Total duration: {duration:.2f} seconds")
            logger.info(f"Will process in {total_chunks} chunks")
            
            all_events = []
            chunk_generator = self._get_chunk_generator(audio_path)
            
            # Create progress bar with more information and more frequent updates
            pbar = tqdm(total=total_chunks, 
                       desc="Processing audio",
                       unit="chunk",
                       position=0,
                       leave=True,
                       mininterval=0.1)  # Update at least every 0.1 seconds
            
            try:
                for chunk_index, chunk in enumerate(chunk_generator):
                    try:
                        chunk_start_time = chunk_index * (self.chunk_samples / self.sample_rate)
                        events = self._process_chunk(chunk, chunk_index, chunk_start_time)
                        all_events.extend(events)
                        
                        # Save clips only if enabled
                        if self.save_clips:
                            self._save_clips(chunk, events, chunk_start_time)
                        
                        # Update progress bar with more info
                        pbar.update(1)
                        pbar.set_postfix({
                            'spikes': len(events),
                            'total_spikes': len(all_events),
                            'position': f"{chunk_start_time:.1f}s"
                        })
                        pbar.refresh()  # Force refresh the display
                        
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_index}: {str(e)}")
                        continue
            finally:
                pbar.close()
            
            logger.info(f"Processing complete. Found {len(all_events)} spikes total.")
            return all_events
            
        except Exception as e:
            logger.error(f"Error in process_file: {str(e)}")
            raise
    
    def _save_clips(self, 
                   chunk: np.ndarray,
                   events: List[SpikeEvent],
                   chunk_start_time: float):
        """Save detected clips to files."""
        if not self.save_clips:
            return
            
        for event in events:
            try:
                start_sample = int((event.start_time - chunk_start_time) * self.sample_rate)
                end_sample = int((event.end_time - chunk_start_time) * self.sample_rate)
                
                # Ensure valid indices
                start_sample = max(0, start_sample)
                end_sample = min(len(chunk), end_sample)
                
                if end_sample > start_sample:
                    clip = chunk[start_sample:end_sample]
                    if len(clip) > 0:
                        output_path = self.output_dir / f"spike_{event.peak_time:.2f}.wav"
                        sf.write(output_path, clip, self.sample_rate)
            except Exception as e:
                logger.error(f"Error saving clip: {str(e)}")
                continue

def process_single_file(args: Tuple[str, bool, float]) -> List[SpikeEvent]:
    """Process a single file (for parallel processing)."""
    audio_path, save_clips, chunk_duration = args
    detector = StreamingSpikeDetector(save_clips=save_clips, chunk_duration=chunk_duration)
    return detector.process_file(audio_path)

def save_events_to_json(events: List[SpikeEvent], output_path: str):
    """Save spike events to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump([event.to_dict() for event in events], f, indent=2)

def main():
    """Process audio files and detect spikes."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect spikes in audio files')
    parser.add_argument('audio_paths', nargs='+', help='Paths to the audio files to process')
    parser.add_argument('--save-clips', action='store_true', help='Save individual audio clips')
    parser.add_argument('--chunk-duration', type=float, default=10.0, help='Duration of each processing chunk in seconds')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count(), help='Number of parallel workers')
    parser.add_argument('--output-dir', type=str, default='spike_detection_results', help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare arguments for parallel processing
    process_args = [(path, args.save_clips, args.chunk_duration) for path in args.audio_paths]
    
    # Process files in parallel
    all_events = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for events in tqdm(executor.map(process_single_file, process_args), 
                          total=len(args.audio_paths),
                          desc="Processing files"):
            all_events.extend(events)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"spike_events_{timestamp}.json"
    save_events_to_json(all_events, str(output_path))
    
    print(f"\nProcessing complete!")
    print(f"Found {len(all_events)} spikes total")
    print(f"Results saved to {output_path}")
    if args.save_clips:
        print(f"Audio clips saved to {output_dir / 'detected_clips'}")

if __name__ == "__main__":
    main() 