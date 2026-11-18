# backend/core/speaker_id.py
import torch
import numpy as np
import warnings
from speechbrain.inference.speaker import EncoderClassifier

warnings.filterwarnings("ignore")

class StreamSpeakerID:
    def __init__(self, savedir="models/weights/speechbrain"):
        """
        Initializes the ECAPA-TDNN model for Speaker Verification.
        """
        # FIX: Ensure device is a string, not a torch.device object
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Loading SpeechBrain ECAPA-TDNN on {self.device}...")
        
        # Load the locally cached model 
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir=savedir,
            run_opts={"device": self.device}
        )
        self.classifier.eval()

    def extract_embedding(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Extracts a 192-dimensional speaker embedding from the audio chunk.
        """
        # Convert string back to torch.device for tensor allocation
        torch_dev = torch.device(self.device)
        tensor_chunk = torch.from_numpy(audio_chunk).float().unsqueeze(0).to(torch_dev)
        
        with torch.no_grad():
            embeddings = self.classifier.encode_batch(tensor_chunk)
            embedding_vector = embeddings.squeeze().cpu().numpy()
            
        return embedding_vector

    def verify_speaker(self, embedding_1: np.ndarray, embedding_2: np.ndarray, threshold: float = 0.25) -> tuple[bool, float]:
        """
        Calculates cosine similarity between two embeddings.
        Returns (is_match, similarity_score).
        """
        score = np.dot(embedding_1, embedding_2) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))
        return bool(score >= threshold), float(score)

if __name__ == "__main__":
    speaker_engine = StreamSpeakerID()
    
    simulated_audio_1 = np.random.randn(4000).astype(np.float32)
    simulated_audio_2 = np.random.randn(4000).astype(np.float32) * 0.5 
    
    emb_1 = speaker_engine.extract_embedding(simulated_audio_1)
    emb_2 = speaker_engine.extract_embedding(simulated_audio_2)
    
    is_match, similarity = speaker_engine.verify_speaker(emb_1, emb_2)
    
    print(f"Extracted Embedding Shape: {emb_1.shape}")
    print(f"Cosine Similarity Score: {similarity:.4f}")
    print(f"Are these the same speaker? {is_match}")
