import json
import os
from typing import Dict, Any
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, metrics_path: str = "models/metrics.json") -> Dict[str, Any]:
    """Evaluate model and save metrics to file."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create metrics dictionary
    metrics = {
        "accuracy": float(accuracy),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    # Save metrics to file
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump({"accuracy": accuracy}, f, indent=2)
    
    print(f"[ml_pipeline.evaluation] Model accuracy: {accuracy:.4f}")
    print(f"[ml_pipeline.evaluation] Saved metrics to {metrics_path}")
    
    return metrics
