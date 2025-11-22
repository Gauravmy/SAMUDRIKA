"""
FastAPI backend for frontend ↔ backend ↔ ML integration testing.

Endpoints:
POST /train-linear
POST /train-multiple-linear
POST /train-logistic-binary
POST /train-logistic-multiclass
POST /train-dummy
POST /predict
GET /health

Run:
uvicorn backend.main:app --reload
"""

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path

# Ensure backend is in the path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils import train_utils
from backend.utils.predict_utils import predict_from_store
from backend.models import store


app = FastAPI(title="Samudrika - ML Integration Backend")

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------- Pydantic Models --------------------
class PredictRequest(BaseModel):
    model_name: str
    input: List[float]


# -------------------- Health Check --------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "available_models": list(store.MODEL_STORE.keys()),  # type: ignore[attr-defined]
        "note": "Train models using /train-* endpoints."
    }


# ==========================================================
#                 TRAINING ENDPOINTS
# ==========================================================

# ***** LINEAR REGRESSION *****
@app.post("/train-linear")
async def train_linear(file: Optional[UploadFile] = None):
    try:
        if file:
            df = train_utils.load_csv_from_upload(file)
        else:
            df = train_utils.generate_dummy_regression(n_samples=200, n_features=1)

        result = train_utils.train_linear_regression(df)

        store.MODEL_STORE["linear"] = {  # type: ignore[attr-defined]
            "model": result["model_obj"],
            "n_features": result["n_features"]
        }

        return {
            "model": "linear_regression",
            "status": "trained",
            "mse": result["mse"],
            "accuracy": result["accuracy"],
            "coefficients": result["coefficients"],
            "sample_predictions": result["sample_predictions"],
            "graph": result["graph"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ***** MULTIPLE LINEAR REGRESSION *****
@app.post("/train-multiple-linear")
async def train_multiple_linear(file: Optional[UploadFile] = None):
    try:
        if file:
            df = train_utils.load_csv_from_upload(file)
        else:
            df = train_utils.generate_dummy_regression(n_samples=300, n_features=3)

        result = train_utils.train_multiple_linear_regression(df)

        store.MODEL_STORE["multiple"] = {  # type: ignore[attr-defined]
            "model": result["model_obj"],
            "n_features": result["n_features"]
        }

        return {
            "model": "multiple_linear_regression",
            "status": "trained",
            "mse": result["mse"],
            "accuracy": result["accuracy"],
            "coefficients": result["coefficients"],
            "sample_predictions": result["sample_predictions"],
            "graph": result["graph"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ***** BINARY LOGISTIC REGRESSION *****
@app.post("/train-logistic-binary")
async def train_logistic_binary(file: Optional[UploadFile] = None):
    try:
        if file:
            df = train_utils.load_csv_from_upload(file)
        else:
            df = train_utils.generate_dummy_classification(
                n_samples=250, n_features=2, n_classes=2
            )

        result = train_utils.train_logistic_classification(df, multiclass=False)

        store.MODEL_STORE["logistic_binary"] = {  # type: ignore[attr-defined]
            "model": result["model_obj"],
            "n_features": result["n_features"]
        }

        return {
            "model": "logistic_regression_binary",
            "status": "trained",
            "mse": result["mse"],
            "accuracy": result["accuracy"],
            "coefficients": result["coefficients"],
            "sample_predictions": result["sample_predictions"],
            "graph": result["graph"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ***** MULTICLASS LOGISTIC REGRESSION *****
@app.post("/train-logistic-multiclass")
async def train_logistic_multiclass(file: Optional[UploadFile] = None):
    try:
        if file:
            df = train_utils.load_csv_from_upload(file)
        else:
            df = train_utils.generate_dummy_classification(
                n_samples=300, n_features=4, n_classes=3
            )

        result = train_utils.train_logistic_classification(df, multiclass=True)

        store.MODEL_STORE["logistic_multi"] = {  # type: ignore[attr-defined]
            "model": result["model_obj"],
            "n_features": result["n_features"]
        }

        return {
            "model": "logistic_regression_multiclass",
            "status": "trained",
            "mse": result["mse"],
            "accuracy": result["accuracy"],
            "coefficients": result["coefficients"],
            "sample_predictions": result["sample_predictions"],
            "graph": result["graph"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ***** DUMMY MODEL *****
@app.post("/train-dummy")
async def train_dummy(file: Optional[UploadFile] = None):
    try:
        if file:
            df = train_utils.load_csv_from_upload(file)
        else:
            df = train_utils.generate_dummy_regression(n_samples=200, n_features=1)

        result = train_utils.train_dummy_model(df)

        store.MODEL_STORE["dummy"] = {  # type: ignore[attr-defined]
            "model": result["model_obj"],
            "n_features": result["n_features"]
        }

        return {
            "model": "dummy_model",
            "status": "trained",
            "mse": result["mse"],
            "accuracy": result["accuracy"],
            "coefficients": result["coefficients"],
            "sample_predictions": result["sample_predictions"],
            "graph": result["graph"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
#                    PREDICTION ENDPOINT
# ==========================================================
@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        prediction = predict_from_store(req.model_name, req.input)
        return {"prediction": prediction, "success": True}

    except HTTPException as he:
        raise he

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
