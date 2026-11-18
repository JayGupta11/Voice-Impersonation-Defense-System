import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

print("Generating synthetic dataset for hackathon proof-of-concept...")

# 1. Authentic Human Speech (Low AASIST, High Jitter, High Similarity)
human = pd.DataFrame({
    'aasist_prob': np.random.uniform(0.0, 0.1, 500),
    'similarity': np.random.uniform(0.7, 1.0, 500),
    'jitter': np.random.uniform(0.015, 0.03, 500),
    'label': 0 # 0 = Safe
})

# 2. High-Quality Deepfake (High AASIST, Normal Jitter, High Similarity)
deepfake = pd.DataFrame({
    'aasist_prob': np.random.uniform(0.7, 1.0, 500),
    'similarity': np.random.uniform(0.6, 0.9, 500),
    'jitter': np.random.uniform(0.015, 0.03, 500),
    'label': 1 # 1 = Fraud
})

# 3. Robotic TTS / Low-Quality Clone (Variable AASIST, Low Jitter, Low Similarity)
robotic = pd.DataFrame({
    'aasist_prob': np.random.uniform(0.4, 0.9, 500),
    'similarity': np.random.uniform(0.1, 0.5, 500),
    'jitter': np.random.uniform(0.001, 0.005, 500), # Unnaturally smooth
    'label': 1 # 1 = Fraud
})

# Combine and train
df = pd.concat([human, deepfake, robotic])
X = df[['aasist_prob', 'similarity', 'jitter']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training XGBoost Classifier...")
model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model trained with accuracy: {accuracy * 100:.2f}%")

model.save_model("models/fraud_classifier.json")
print("Saved to models/fraud_classifier.json")
