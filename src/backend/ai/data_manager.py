import os
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, List

DATA_DIR = "data/training_vectors"

def save_training_sample(image_features: np.ndarray, settings: Dict[str, Any], filename: str) -> None:
    """
    Saves a training sample (features + target settings) to disk.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Sanitize filename
    safe_name = "".join([c for c in filename if c.isalpha() or c.isdigit() or c in ('-', '_')]).rstrip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(DATA_DIR, f"{safe_name}_{timestamp}.npz")

    # We filter settings to only relevant targets for the AI
    # (ignoring crop, rotation, local adjustments, etc.)
    target_keys = [
        'temperature', 'exposure', 'contrast', 'saturation', 
        'cr_balance', 'mg_balance', 'yb_balance',
        'shadow_temp', 'highlight_temp',
        'shadow_cr', 'shadow_mg', 'shadow_yb',
        'highlight_cr', 'highlight_mg', 'highlight_yb',
        'gamma', 'grade_shadows', 'grade_highlights',
        'black_point', 'white_point'
    ]
    
    targets = {k: settings.get(k, 0.0) for k in target_keys}

    np.savez_compressed(out_path, features=image_features, targets=targets, source_filename=filename)

def get_dataset_stats() -> Dict[str, int]:
    """
    Returns statistics about collected data.
    """
    if not os.path.exists(DATA_DIR):
        return {'count': 0}
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npz')]
    return {'count': len(files)}
