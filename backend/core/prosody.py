# backend/core/prosody.py
import numpy as np
import parselmouth
from parselmouth.praat import call
import warnings

warnings.filterwarnings("ignore")

class StreamProsody:
    def __init__(self, sampling_rate=16000):
        """
        Initializes a 2-second rolling buffer for Prosody extraction.
        Praat requires sufficient pitch periods to calculate Jitter/Shimmer.
        """
        self.sampling_rate = sampling_rate
        # 2 seconds = 32000 samples at 16kHz
        self.buffer_size = self.sampling_rate * 2  
        self.rolling_buffer = np.zeros(self.buffer_size, dtype=np.float32)

    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        """
        Updates the rolling buffer and extracts Pitch variance, Jitter, and Shimmer.
        Returns a dictionary of the prosodic features.
        """
        chunk_length = len(audio_chunk)
        
        # Shift the buffer left and append the new chunk
        self.rolling_buffer = np.roll(self.rolling_buffer, -chunk_length)
        self.rolling_buffer[-chunk_length:] = audio_chunk
        
        # Convert numpy array to Parselmouth Sound object
        sound = parselmouth.Sound(self.rolling_buffer, sampling_frequency=self.sampling_rate)
        
        # Extract features using Praat scripts
        # f0min=75Hz, f0max=600Hz covers standard human vocal range
        try:
            pitch = call(sound, "To Pitch", 0.0, 75.0, 600.0)
            point_process = call(sound, "To PointProcess (periodic, cc)", 75.0, 600.0)
            
            # 1. Pitch Variance (Standard Deviation of F0)
            stdev_f0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")
            
            # 2. Local Jitter (Period-to-period duration variation)
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            
            # 3. Local Shimmer (Period-to-period amplitude variation)
            shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            # Handle NaN values (occurs if the buffer is pure silence)
            if np.isnan(stdev_f0): stdev_f0 = 0.0
            if np.isnan(jitter): jitter = 0.0
            if np.isnan(shimmer): shimmer = 0.0
                
        except Exception as e:
            # Fallback if Praat fails to find periodicity (e.g., coughing, pure static)
            stdev_f0, jitter, shimmer = 0.0, 0.0, 0.0

        return {
            "pitch_variance": stdev_f0,
            "jitter": jitter,
            "shimmer": shimmer
        }

if __name__ == "__main__":
    prosody_engine = StreamProsody()
    
    # Simulate a 250ms chunk of audio (adding sine wave to simulate vocal tone)
    t = np.linspace(0, 0.25, int(16000 * 0.25), endpoint=False)
    simulated_vocal = np.sin(2 * np.pi * 150 * t).astype(np.float32) * 0.5 
    
    # Run a few times to fill the buffer
    for _ in range(8):
        features = prosody_engine.process_chunk(simulated_vocal)
        
    print("Extracted Prosodic Features (Simulated Tone):")
    print(f"Pitch Variance: {features['pitch_variance']:.2f} Hz")
    print(f"Jitter: {features['jitter']:.5f}")
    print(f"Shimmer: {features['shimmer']:.5f}")
