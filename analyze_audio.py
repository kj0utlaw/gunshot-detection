import numpy as np	# math
import librosa		# analyze audio
import sys		# allows system functions (like grabbing files)
import os

def analyze_wav(file_path):
    from scipy.io import wavfile		# Imports audio-reading function from scipy lib
    sr, y = wavfile.read(file_path)	# sr = sample rate(audio resolution), y = waveform data (array)
    y = y.astype(float) / np.max(np.abs(y))	# convert data from int to decimal, finds max; normalize data

    # get audio length in sec
    duration = len(y) / sr			# gets audio length
    max_amplitude = np.max(np.abs(y))		# finds max amp(loudest); prevents neg values
    loudest_sample_index = np.argmax(np.abs(y))	# finds loudest moment (highest value in array)
    loudest_time = loudest_sample_index / sr	# convert to seconds

    # "what freq are present?"
    fft = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(y), 1 / sr)	# create list of freq values to match freq in Hz
    magnitude = np.abs(fft)			# computes stength of each freq

    positive_freqs = freqs[:len(freqs) // 2]	# keeps positive freqs; second half is a mirror of first)
    positive_magnitude = magnitude[:len(magnitude) // 2] # keeps things in sync: 1 array for freq, 1 for how loud that freq is

    threshold = np.max(positive_magnitude) * 0.01	# sets threshold of 1% to avoid background noise
    valid_indices = np.where(positive_magnitude > threshold)[0] # gets indexes of freqs; skips anything too small

    # get highest real freq
    max_freq = positive_freqs[valid_indices[-1]] if len(valid_indices) > 0 else 0

    print(f"File: {os.path.basename(file_path)}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Max Amplitude (Volume): {max_amplitude:.4f}")
    print(f"Loudest Time: {loudest_time:.2f} seconds")
    print(f"Estimated Max Frequency: {max_freq:.2f} Hz")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_audio.py path_to_audio.wav")
    else:
        analyze_wav(sys.argv[1])
