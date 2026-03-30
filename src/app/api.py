# src/app/api.py
import joblib
import numpy as np
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Request schema for Breast Cancer dataset (30 features)
class BreastCancerRequest(BaseModel):
    # Mean radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal_dimension
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    # Standard error features
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    # Worst features
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

def create_app(model_path: str = "models/model_20260330_011806.pkl"):
    """
    Creates a FastAPI app that serves predictions for the Breast Cancer model.
    """
    # Helpful guard so students get a clear error if they forgot to train first
    if not Path(model_path).exists():
        raise RuntimeError(
            f"Model file not found at '{model_path}'. "
            "Train the model first (run the ml_training_pipeline_v2 DAG)."
        )

    model = joblib.load(model_path)
    app = FastAPI(title="Breast Cancer Model API")

    # Map numeric predictions to class names
    target_names = {0: "malignant", 1: "benign"}

    @app.get("/")
    def root():
        return {
            "message": "Breast Cancer model is ready for inference!",
            "classes": target_names,
        }

    @app.post("/predict")
    def predict(request: BreastCancerRequest):
        # Convert request into the correct shape (1 x 30)
        features = [
            request.mean_radius, request.mean_texture, request.mean_perimeter, request.mean_area, request.mean_smoothness,
            request.mean_compactness, request.mean_concavity, request.mean_concave_points, request.mean_symmetry, request.mean_fractal_dimension,
            request.radius_error, request.texture_error, request.perimeter_error, request.area_error, request.smoothness_error,
            request.compactness_error, request.concavity_error, request.concave_points_error, request.symmetry_error, request.fractal_dimension_error,
            request.worst_radius, request.worst_texture, request.worst_perimeter, request.worst_area, request.worst_smoothness,
            request.worst_compactness, request.worst_concavity, request.worst_concave_points, request.worst_symmetry, request.worst_fractal_dimension
        ]
        X = np.array([features])
        try:
            idx = int(model.predict(X)[0])
        except Exception as e:
            # Surface any shape/validation issues as a 400 instead of a 500
            raise HTTPException(status_code=400, detail=str(e))
        return {"prediction": target_names[idx], "class_index": idx}

    @app.get("/model/info")
    def model_info():
        """Return model metadata if available."""
        metadata_path = "models/metadata.json"
        if Path(metadata_path).exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return metadata
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading metadata: {str(e)}")
        else:
            return {
                "message": "No metadata found. Train the model using ml_training_pipeline_v2 DAG.",
                "model_path": model_path,
                "classes": target_names
            }

    # return the FastAPI app
    return app
