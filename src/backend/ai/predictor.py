import os
import joblib
import numpy as np
from typing import Dict, Any, Optional
from src.backend.ai.trainer import MODELS_DIR

def load_model(model_name: str) -> Optional[Any]:
    path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def predict_settings(image_features: np.ndarray, model_name: str) -> Dict[str, float]:
    """
    Predicts settings for a given feature vector using the specified model.
    """
    data = load_model(model_name)
    if not data:
        raise FileNotFoundError(f"Model {model_name} not found.")

    model = data['model']
    target_keys = data['target_keys']

    # Predict
    # input must be 2D array (1, n_features)
    preds = model.predict(image_features.reshape(1, -1))[0]

    return dict(zip(target_keys, preds))
