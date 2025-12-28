import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from src.backend.ai.data_manager import DATA_DIR

MODELS_DIR = "models"

def train_model(model_name: str = "default_style") -> str:
    """
    Trains a model on all available .npz data and saves it.
    Filters to keep only the latest sample for each source file.
    Returns: Status message.
    """
    if not os.path.exists(DATA_DIR):
        return "No training data found."

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npz')]
    if not files:
        return "No training data files found."

    # Deduplicate logic: Map source_filename -> (timestamp, data_dict)
    # Since filenames contain timestamp "safe_name_YYYYMMDD_HHMMSS.npz", 
    # we can just rely on the 'source_filename' key inside the npz if present,
    # or fallback to file timestamp parsing.
    
    latest_samples = {} # source_filename -> (timestamp_str, filepath)

    for f in files:
        # Extract timestamp from filename format: name_YYYYMMDD_HHMMSS.npz
        # We assume the last two parts separated by underscore are date and time.
        try:
            parts = f.replace('.npz', '').split('_')
            if len(parts) >= 2:
                ts_str = f"{parts[-2]}_{parts[-1]}" # YYYYMMDD_HHMMSS
            else:
                ts_str = "00000000_000000"
            
            # We need to peek inside to get the real source filename to key off of
            # But peeking all files is slow.
            # Optimization: Use the prefix of the filename as the key?
            # Filename: safe_name_timestamp.npz
            # If safe_name is unique enough (it strips some chars), it might collide.
            # Better to read 'source_filename' from npz since we just added it.
            # If it's missing (old data), fallback to prefix.
            
            data = np.load(os.path.join(DATA_DIR, f), allow_pickle=True)
            if 'source_filename' in data:
                src_key = str(data['source_filename'])
            else:
                # Fallback for older files
                src_key = "_".join(parts[:-2])
            
            if src_key not in latest_samples:
                latest_samples[src_key] = (ts_str, f, data)
            else:
                if ts_str > latest_samples[src_key][0]:
                    latest_samples[src_key] = (ts_str, f, data)
                    
        except Exception as e:
            print(f"Error scanning {f}: {e}")

    if not latest_samples:
        return "No valid samples found after scanning."

    X_list = []
    y_list = []
    
    # Use the first found valid sample to determine target keys
    first_data = next(iter(latest_samples.values()))[2]
    target_keys = sorted(first_data['targets'].item().keys())

    for _, _, data in latest_samples.values():
        try:
            X_list.append(data['features'])
            targets = data['targets'].item()
            y_row = [targets.get(k, 0.0) for k in target_keys] # .get for safety
            y_list.append(y_row)
        except Exception as e:
            print(f"Skipping sample during build: {e}")

    if not X_list:
        return "Failed to load any valid data."

    X = np.array(X_list)
    y = np.array(y_list)

    # Train
    # Random Forest is robust and good for this tabular regression
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(rf)
    model.fit(X, y)

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    out_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    
    # Save model and target keys metadata
    joblib.dump({
        'model': model,
        'target_keys': target_keys,
        'n_samples': len(X)
    }, out_path)

    return f"Training complete on {len(X)} unique samples (deduplicated). Saved to {model_name}.pkl"
