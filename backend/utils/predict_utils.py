"""
Prediction utilities to run inference against stored models.
"""
from typing import Any, List
import numpy as np
import sys
from pathlib import Path

# Ensure backend is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.models.store import MODEL_STORE  # type: ignore[import-not-found]
from fastapi import HTTPException


def predict_from_store(model_name: str, input_list: List[float]) -> Any:
    """Look up model in MODEL_STORE and run prediction.
    input_list should be a flat list of numbers matching the model's expected feature length.
    """
    if model_name not in MODEL_STORE:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Train it first.")
    entry = MODEL_STORE[model_name]
    model = entry.get('model')
    n_features = entry.get('n_features')
    if n_features is None:
        # fallback
        n_features = len(input_list)
    # validate input length
    if len(input_list) != n_features:
        raise HTTPException(status_code=400, detail=f"Input length {len(input_list)} does not match model's expected feature count {n_features}.")
    X = np.array(input_list, dtype=float).reshape(1, -1)
    # Support models that are sklearn-like or our dummy predict function
    try:
        if hasattr(model, 'predict'):
            pred = model.predict(X)
            # if prediction returns array-like
            if hasattr(pred, '__len__'):
                out = pred[0]
            else:
                out = pred
        elif callable(model):
            out = model(X)
        else:
            raise HTTPException(status_code=500, detail="Stored model does not have a predict method")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    # cast numpy types
    if hasattr(out, 'item'):
        try:
            out = out.item()  # type: ignore[attr-defined]
        except Exception:
            pass
    return out
