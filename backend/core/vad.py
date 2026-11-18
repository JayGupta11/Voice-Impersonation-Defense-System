# backend/core/vad.py
import torch
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class StreamVAD:
    def __init__(self, threshold=0.5, sampling_rate=16000):
        self.sampling_rate = sampling_rate
        self.threshold = threshold
        
        # Silero VAD v5 requires exact window sizes for streaming inference
        self.window_size_samples = 512 if sampling_rate == 16000 else 256
        
        print("Loading Silero VAD from local cache...")
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Evaluates a raw PCM audio chunk for human speech.
        Internally slices the chunk into 512-sample windows for Silero inference.
        """
        tensor_chunk = torch.from_numpy(audio_chunk).float()
        num_windows = len(tensor_chunk) // self.window_size_samples
        
        with torch.no_grad():
            for i in range(num_windows):
                start_idx = i * self.window_size_samples
                end_idx = start_idx + self.window_size_samples
                window = tensor_chunk[start_idx:end_idx]
                
                # The model maintains its own internal state across sequential calls
                speech_prob = self.model(window, self.sampling_rate).item()
                
                if speech_prob >= self.threshold:
                    return True  # Speech detected in this micro-chunk
                    
        return False

if __name__ == "__main__":
    vad = StreamVAD()
    
    # Simulate a 250ms audio chunk of pure silence (4000 samples) at 16kHz
    chunk_size = int(16000 * 0.25) 
    silent_chunk = np.zeros(chunk_size, dtype=np.float32)
    
    result = vad.is_speech(silent_chunk)
    print(f"Speech detected in silent chunk: {result}")
