# scripts/download_models.py
import os
import torch
import urllib.request
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

def cache_silero_vad():
    print("Downloading/Caching Silero VAD...")
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False)
    print("Silero VAD cached successfully.")

def cache_speechbrain():
    print("Downloading/Caching SpeechBrain ECAPA-TDNN...")
    os.makedirs("models/weights/speechbrain", exist_ok=True)
    
    # We pass LocalStrategy.COPY to bypass the Windows symlink privilege error
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="models/weights/speechbrain",
        local_strategy=LocalStrategy.COPY
    )
    print("SpeechBrain ECAPA-TDNN cached successfully.")

def download_aasist():
    print("Downloading AASIST pre-trained weights...")
    aasist_dir = "models/weights/aasist"
    os.makedirs(aasist_dir, exist_ok=True)
    
    url = "https://raw.githubusercontent.com/clovaai/aasist/main/models/weights/AASIST.pth"
    dest = os.path.join(aasist_dir, "AASIST.pth")
    
    if not os.path.exists(dest):
        urllib.request.urlretrieve(url, dest)
        print(f"AASIST downloaded to {dest}")
    else:
        print("AASIST already exists locally.")

if __name__ == "__main__":
    print("--- Starting Offline Model Caching ---")
    
    # Suppress the HuggingFace fallback warning to keep the terminal clean
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    cache_silero_vad()
    cache_speechbrain()
    download_aasist()
    print("--- All models successfully cached for offline use ---")
