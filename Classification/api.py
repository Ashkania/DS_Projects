"""Minimal FastAPI inference service.

Serves predictions for ONE project per running instance, selected via the
PROJECT_NAME environment variable -- mirroring the --project-name CLI
argument in pipeline.py. To serve both stellar and irrigation, run two
instances of this exact same file:

    PROJECT_NAME=stellar    uvicorn api:app --port 8001
    PROJECT_NAME=irrigation uvicorn api:app --port 8002

This file never hardcodes a project name itself -- consistent with how
pipeline.py stays generic and is parameterized at the CLI instead.
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

from feature_engineering import load_feature_engineer

PROJECT_NAME = os.environ.get('PROJECT_NAME')
if not PROJECT_NAME:
    raise RuntimeError(
        'PROJECT_NAME environment variable is required, e.g. '
        'PROJECT_NAME=stellar uvicorn api:app'
    )

MODEL_DIR = f'artifacts/{PROJECT_NAME}'

# Loaded once at startup, not per-request.
model = joblib.load(os.path.join(MODEL_DIR, 'model.joblib'))
preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.joblib'))
le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
feature_module = joblib.load(os.path.join(MODEL_DIR, 'feature_module.joblib'))
feature_engineer = load_feature_engineer(feature_module) if feature_module else None

app = FastAPI(title=f'{PROJECT_NAME} classifier')


class PredictRequest(BaseModel):
    # Raw feature names/values are project-specific and not validated by
    # field name here -- the preprocessor itself raises a clear error if
    # an expected column is missing or mistyped.
    features: Dict[str, float]


@app.get('/health')
def health():
    return {'status': 'ok', 'project': PROJECT_NAME}


@app.post('/predict')
def predict(request: PredictRequest):
    row = pd.DataFrame([request.features])

    try:
        if feature_engineer is not None:
            # transform() is built for (train, test) pairs; at inference
            # time we only have one row, so pass it as both and keep just
            # the first. The two frames don't reference each other inside
            # transform(), so this is safe -- just slightly wasteful.
            row, _ = feature_engineer.transform(row, row.copy())
        processed = preprocessor.transform(row)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f'Invalid input: {e}')

    pred = model.predict(processed)
    label = le.inverse_transform(pred)[0]

    return {'prediction': label}