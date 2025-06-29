import numpy as np 
import librosa # audio processing
import soundfile as sf # read audio files
from typing import List, Tuple, Optional, Generator, Dict, Any # type hints
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import os
from tqdm import tqdm # progress bar
import multiprocessing as mp # parallel processing
from concurrent.futures import ProcessPoolExecutor # parallel processing
import json 
from datetime import datetime
from scipy import signal # find peaks

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SpikeEvent:
    """Holds spike event data"""
    start_time: float  # spike start time (sec)
    end_time: float    # spike end time (sec)
    peak_time: float   # exact peak time (sec)
    peak_amplitude: float  # peak amplitude value
    chunk_index: int   # chunk index where found
    local_index: int   # position within chunk
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON save"""
        return asdict(self)

class StreamingSpikeDetector:
    def __init__(self, save_clips: bool = False, chunk_duration: float = 10.0):
        """
        Initialize spike detector
        
        Args:
            save_clips: save individual clips or not
            chunk_duration: chunk length (sec)
        """
        # Core settings
        self.sample_rate = 44100 # 44.1 kHz
        self.clip_duration = 0.5  # 500ms clips around spikes
        self.chunk_duration = chunk_duration  # changed from 60s to 10s for memory
        self.rms_window = int(0.02 * self.sample_rate)  # 20ms RMS window
        self.smooth_window = int(0.5 * self.sample_rate)  # 500ms smoothing
        
        # Derived values
        self.clip_samples = int(self.clip_duration * self.sample_rate)
        self.chunk_samples = int(self.chunk_duration * self.sample_rate)
        
        # Buffer for cross-chunk spikes
        self.buffer = np.array([])
        self.buffer_duration = self.clip_duration
        
        # Output settings
        self.save_clips = save_clips
        if self.save_clips:
            self.output_dir = Path("detected_clips")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _compute_rms_vectorized(self, audio: np.ndarray) -> np.ndarray:
        """Compute RMS energy (vectorized for speed)"""
        try:
            # Use float32 for efficiency
            audio = audio.astype(np.float32)
            
            # Pad for edge handling
            padded = np.pad(audio, (self.rms_window//2, self.rms_window//2), mode='edge')
            
            # Fast RMS via convolution
            squared = padded ** 2
            rms = np.sqrt(np.convolve(squared, np.ones(self.rms_window)/self.rms_window, mode='valid'))
            
            return rms.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error in RMS computation: {str(e)}")
            return np.array([], dtype=np.float32)
    
    def _compute_baseline_vectorized(self, rms: np.ndarray) -> np.ndarray:
        """Compute smoothed baseline (moving avg)"""
        try:
            # Use float32
            rms = rms.astype(np.float32)
            
            # Pad for edge handling
            padded = np.pad(rms, (self.smooth_window//2, self.smooth_window//2), mode='edge')
            
            # Fast moving avg via convolution
            baseline = np.convolve(padded, np.ones(self.smooth_window)/self.smooth_window, mode='valid')
            
            return baseline.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error in baseline computation: {str(e)}")
            return np.array([], dtype=np.float32)
    
    def _detect_spikes_vectorized(self, audio: np.ndarray) -> List[int]:
        """
        Find spikes using RMS vs baseline contrast
        
        Args:
            audio: audio chunk to analyze
            
        Returns:
            List of spike sample indices
        """
        try:
            # Use float32
            audio = audio.astype(np.float32)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Get RMS and baseline
            rms = self._compute_rms_vectorized(audio)
            baseline = self._compute_baseline_vectorized(rms)
            
            # Match lengths
            min_len = min(len(rms), len(baseline))
            rms = rms[:min_len]
            baseline = baseline[:min_len]
            
            # Calculate contrast (RMS - baseline)
            contrast = rms - baseline
            
            # Clean NaN values
            contrast[np.isnan(contrast)] = 0
            
            # Find peaks (good settings for gunshots)
            peaks, _ = signal.find_peaks(
                contrast,
                height=0.01,  # min height
                distance=int(0.05 * self.sample_rate),  # min 50ms between peaks
                prominence=0.005  # min prominence
            )
            
            return peaks.tolist()
            
        except Exception as e:
            logger.error(f"Error in spike detection: {str(e)}")
            return []
    
    def _get_chunk_generator(self, audio_path: str) -> Generator[np.ndarray, None, None]:
        """Generator for streaming audio chunks"""
        logger.info(f"Starting to process file: {audio_path}")
        
        try:
            # Get file info
            info = sf.info(audio_path)
            duration = info.duration
            logger.info(f"Audio duration: {duration:.2f} seconds")
            
            # Calc samples per chunk
            samples_per_chunk = int(self.chunk_duration * info.samplerate)
            
            # Stream chunks
            with sf.SoundFile(audio_path) as audio_file:
                chunk_count = 0
                while True:
                    # Read next chunk
                    chunk = audio_file.read(samples_per_chunk)
                    if len(chunk) == 0:  # EOF
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
        """Process chunk and find spikes"""
        try:
            # Add prev chunk buffer if exists
            if len(self.buffer) > 0:
                chunk = np.concatenate([self.buffer, chunk])
            
            # Find spikes in chunk
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
            
            # Update buffer for next chunk
            buffer_samples = int(self.buffer_duration * self.sample_rate)
            self.buffer = chunk[-buffer_samples:] if len(chunk) > buffer_samples else chunk
            
            return events
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_index}: {str(e)}")
            return []
    
    def process_file(self, audio_path: str) -> List[SpikeEvent]:
        """
        Process entire audio file and find all spikes
        
        Args:
            audio_path: path to audio file
            
        Returns:
            List of all spike events
        """
        try:
            # Get file info for progress
            info = sf.info(audio_path)
            duration = info.duration
            total_chunks = int(np.ceil(duration / self.chunk_duration))
            logger.info(f"Total duration: {duration:.2f} seconds")
            logger.info(f"Will process in {total_chunks} chunks")
            
            all_events = []
            chunk_generator = self._get_chunk_generator(audio_path)
            
            # Setup progress bar
            pbar = tqdm(total=total_chunks, 
                       desc="Processing audio",
                       unit="chunk",
                       position=0,
                       leave=True,
                       mininterval=0.1)  # Update every 0.1s
            
            try:
                for chunk_index, chunk in enumerate(chunk_generator):
                    try:
                        chunk_start_time = chunk_index * (self.chunk_samples / self.sample_rate)
                        events = self._process_chunk(chunk, chunk_index, chunk_start_time)
                        all_events.extend(events)
                        
                        # Save clips if requested - Format: python3 spike_detector.py --save_clips --input_audio path/to/audio.wav
                        if self.save_clips:
                            self._save_clips(chunk, events, chunk_start_time)
                        
                        # Update progress
                        pbar.update(1)
                        pbar.set_postfix({
                            'spikes': len(events),
                            'total_spikes': len(all_events),
                            'position': f"{chunk_start_time:.1f}s"
                        })
                        pbar.refresh()  # Force update
                        
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
    
    def _save_clips(self, chunk: np.ndarray, events: List[SpikeEvent], chunk_start_time: float):
        """Save detected clips as audio files"""
        if not self.save_clips:
            return
            
        for event in events:
            try:
                # Calc sample indices for clip
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
    """Process single file (for parallel processing)"""
    audio_path, save_clips, chunk_duration = args
    detector = StreamingSpikeDetector(save_clips=save_clips, chunk_duration=chunk_duration)
    return detector.process_file(audio_path)

def save_events_to_json(events: List[SpikeEvent], output_path: str):
    """Save spike events to JSON file"""
    with open(output_path, 'w') as f:
        json.dump([event.to_dict() for event in events], f, indent=2)

def main():
    """Main - process audio files and detect spikes"""
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
    
    # Prepare args for parallel processing
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