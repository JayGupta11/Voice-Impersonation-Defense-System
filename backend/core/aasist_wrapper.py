# backend/core/aasist_wrapper.py
import sys
import os
import torch
import json
import numpy as np
import torch.nn.functional as F

# Append the cloned repo so Python can find the Clova AI model definitions
sys.path.append(os.path.join(os.path.dirname(__file__), 'aasist_repo'))
from models.AASIST import Model

class StreamAASIST:
    def __init__(self, weight_path="models/weights/aasist/AASIST.pth"):
        """
        Initializes the AASIST model and the 64,600 sample rolling buffer.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Loading AASIST Anti-Spoofing Model on {self.device}...")
        
        # Load the Clova AI default configuration
        with open("backend/core/aasist_repo/config/AASIST.conf", "r") as f:
            config = json.load(f)
            
        # Initialize the model architecture and load our cached weights
        self.model = Model(config["model_config"])
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # AASIST requires exactly 64,600 samples per inference pass
        self.buffer_size = 64600
        self.rolling_buffer = np.zeros(self.buffer_size, dtype=np.float32)

    def process_chunk(self, audio_chunk: np.ndarray) -> float:
        """
        Appends a new 250ms chunk to the rolling buffer and returns the spoof probability.
        """
        chunk_length = len(audio_chunk)
        
        # Shift the buffer left and append the new chunk to the end
        self.rolling_buffer = np.roll(self.rolling_buffer, -chunk_length)
        self.rolling_buffer[-chunk_length:] = audio_chunk
        
        # Convert to tensor and add batch dimension: shape (1, 64600)
        tensor_input = torch.from_numpy(self.rolling_buffer).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, output = self.model(tensor_input)
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(output, dim=1)
            
            # Index 0 is the Spoof/Deepfake probability, Index 1 is Bona-fide (Human)
            spoof_prob = probabilities[0][0].item() 
            
        return spoof_prob

if __name__ == "__main__":
    aasist_engine = StreamAASIST()
    
    # Simulate a new 250ms chunk arriving from the telephony server
    simulated_chunk = np.random.randn(4000).astype(np.float32) * 0.01 
    
    risk_score = aasist_engine.process_chunk(simulated_chunk)
    print(f"Deepfake Probability Score: {risk_score * 100:.2f}%")
