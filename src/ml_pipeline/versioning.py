import json
import os
from datetime import datetime
from typing import Dict, Any

def generate_model_version(execution_date=None) -> str:
    """Generate model version based on timestamp or execution date."""
    if execution_date:
        return execution_date.strftime("%Y%m%d_%H%M%S")
    else:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_metadata(model_version: str, dataset: str, model_type: str, accuracy: float, 
                 metadata_path: str = "models/metadata.json") -> Dict[str, Any]:
    """Save model metadata to file."""
    metadata = {
        "model_version": model_version,
        "dataset": dataset,
        "model_type": model_type,
        "accuracy": float(accuracy)
    }
    
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[ml_pipeline.versioning] Saved metadata to {metadata_path}")
    return metadata
