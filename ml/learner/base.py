import os
import json
import pickle
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data.binance.data_downloader import data_downloader



def setup_trainer_directory(base_path, symbol, time_horizon, model_name, metadata, model_object=None):

    # Create folder path
    folder_path = os.path.join(base_path, symbol, time_horizon, model_name)
    os.makedirs(folder_path, exist_ok=True)

    # Paths
    metadata_path = os.path.join(folder_path, 'metadata.json')
    if model_name == "lstm_model":
        model_file_path = os.path.join(folder_path, 'model_weights.h5')
    else:
        model_file_path = os.path.join(folder_path, 'model_file.pkl')

    # Add created_at if not present
    if 'created_at' not in metadata:
        metadata['created_at'] = datetime.utcnow().isoformat() + 'Z'

    # Save metadata JSON
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    # Save model file if provided
    if model_object is not None:
        with open(model_file_path, 'wb') as f:
            pickle.dump(model_object, f)

    return {
        'folder': folder_path,
        'metadata': metadata_path,
        'model_file': model_file_path  # This is where the .pkl or .h5 is saved
    }

