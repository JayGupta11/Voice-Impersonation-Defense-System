# backend/core/pipeline.py
import numpy as np
import os
import xgboost as xgb
from .vad import StreamVAD
from .aasist_wrapper import StreamAASIST
from .speaker_id import StreamSpeakerID
from .prosody import StreamProsody

class DefensePipeline:
    def __init__(self, target_speaker_embedding: np.ndarray = None):
        """
        Initializes the full defense pipeline.
        target_speaker_embedding: The historical ground-truth embedding of the CXO.
        """
        print("Initializing Full AI Defense Pipeline...")
        self.vad = StreamVAD()
        self.aasist = StreamAASIST()
        self.speaker_id = StreamSpeakerID()
        self.prosody = StreamProsody()
        
        # In a full deployment, this would be fetched from Qdrant Vector DB
        self.target_embedding = target_speaker_embedding 

        # Load the ML Fraud Classifier
        self.classifier = xgb.XGBClassifier()
        
        # Resolve the absolute path to the models directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, 'models', 'fraud_classifier.json')
        
        try:
            self.classifier.load_model(model_path)
            print("XGBoost Fraud Classifier loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load classifier: {e}")
            self.classifier = None

    def process_stream_chunk(self, audio_chunk: np.ndarray) -> dict:
        """
        Processes a single 250ms chunk through all models and calculates risk.
        """
        # 1. Voice Activity Detection
        if not self.vad.is_speech(audio_chunk):
            return {"status": "silence", "risk_score": 0.0}

        # 2. AASIST Anti-Spoofing (Deepfake Artifacts)
        deepfake_prob = float(self.aasist.process_chunk(audio_chunk))

        # 3. Prosody Analysis (Indian Accent Microvariations)
        prosody_features = self.prosody.process_chunk(audio_chunk)
        jitter = float(prosody_features["jitter"])
        
        # 4. Speaker Verification (Identity Matching)
        similarity = 0.0
        identity_mismatch = 0.0
        if self.target_embedding is not None:
            current_embedding = self.speaker_id.extract_embedding(audio_chunk)
            is_match, similarity = self.speaker_id.verify_speaker(current_embedding, self.target_embedding)
            identity_mismatch = 1.0 if not is_match else 0.0
            similarity = float(similarity)

        # 5. ML Risk Engine Integration
        if self.classifier:
            # XGBoost expects a 2D array matching the training feature order: [aasist_prob, similarity, jitter]
            features = np.array([[deepfake_prob, similarity, jitter]])
            
            # predict_proba returns [[prob_safe, prob_fraud]]. We extract index 1 (Fraud)
            fraud_probability = float(self.classifier.predict_proba(features)[0][1])
            risk_score = round(fraud_probability * 100, 2)
        else:
            # Fallback heuristic if the JSON model fails to load
            prosody_anomaly = 1.0 if jitter < 0.005 else 0.0
            risk_score = (deepfake_prob * 40) + (identity_mismatch * 30) + (prosody_anomaly * 20)
            risk_score = round(min(max(risk_score, 0), 100), 2)

        # Update dynamic alert levels based on ML confidence
        alert_level = "Safe"
        if risk_score >= 45.0: 
            alert_level = "Suspicious"
        if risk_score >= 75.0: 
            alert_level = "Critical"

        return {
            "status": "speech_detected",
            "alert_level": alert_level,
            "risk_score": risk_score,
            "breakdown": {
                "aasist_prob": round(deepfake_prob, 4),
                "speaker_similarity": round(similarity, 4),
                "jitter": round(jitter, 5)
            }
        }
