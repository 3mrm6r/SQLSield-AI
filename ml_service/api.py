import sys
import os
import re
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from src import config
from src.predict import load_pipeline, predict_with_confidence

app = FastAPI(title="SQLi ML Inference API", version="1.0.0")

# Preload model pipeline at startup
pipeline = load_pipeline()


class PredictRequest(BaseModel):
    query: str


class PredictResponse(BaseModel):
    query: str
    label: str
    confidence: float
    is_sql_injection: bool


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": pipeline is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    label, conf = predict_with_confidence(payload.query, pipeline=pipeline)
    if label is None:
        raise HTTPException(status_code=500, detail="Prediction failed")

    return PredictResponse(
        query=payload.query,
        label=label,
        confidence=conf if conf is not None else 0.0,
        is_sql_injection=(label == "SQL Injection"),
    )


def _parse_metrics_txt(path: Path):
    if not path.is_file():
        return None
    content = path.read_text(encoding="utf-8")
    data = {
        "raw_text": content,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
        "fpr": None,
        "fnr": None,
        "tn": None,
        "fp": None,
        "fn": None,
        "tp": None,
        "test_samples": None,
        "total_time": None,
        "avg_query_time_ms": None,
    }

    for line in content.splitlines():
        line = line.strip()
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip().lower()
            v = v.strip()
            try:
                if k == "accuracy":
                    data["accuracy"] = float(v)
                elif k == "precision":
                    data["precision"] = float(v)
                elif k == "recall":
                    data["recall"] = float(v)
                elif k == "f1-score":
                    data["f1"] = float(v)
                elif k == "roc-auc":
                    data["roc_auc"] = float(v)
                elif k == "false positive rate":
                    data["fpr"] = float(v)
                elif k == "false negative rate":
                    data["fnr"] = float(v)
                elif "true negatives" in k or "(tn)" in k:
                    data["tn"] = int(v)
                elif "false positives" in k or "(fp)" in k:
                    data["fp"] = int(v)
                elif "false negatives" in k or "(fn)" in k:
                    data["fn"] = int(v)
                elif "true positives" in k or "(tp)" in k:
                    data["tp"] = int(v)
                elif k == "test samples":
                    data["test_samples"] = int(v)
                elif k == "total prediction time":
                    data["total_time"] = v
                elif k == "avg time per query":
                    data["avg_query_time_ms"] = v
            except ValueError:
                pass

    return data


def _parse_feature_importance(path: Path):
    if not path.is_file():
        return []
    content = path.read_text(encoding="utf-8")
    features = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("=") or line.startswith("Top"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) == 2:
            token, score = parts
            try:
                features.append({"token": token, "importance": float(score)})
            except ValueError:
                pass
    return features


@app.get("/metrics")
def get_model_metrics():
    has_csv = config.COMPARISON_TABLE_CSV.is_file()
    has_txt = config.METRICS_REPORT.is_file()

    if not has_csv and not has_txt:
        return {
            "available": False,
            "message": "No reports generated yet. Run python -m src.train and python -m src.evaluate first."
        }

    models = []
    if has_csv:
        df = pd.read_csv(config.COMPARISON_TABLE_CSV)
        models = df.to_dict(orient="records")

    evaluation = _parse_metrics_txt(config.METRICS_REPORT) if has_txt else None
    features = _parse_feature_importance(config.FEATURE_IMPORTANCE_TXT) if config.FEATURE_IMPORTANCE_TXT.is_file() else []

    return {
        "available": True,
        "best_model": "Random Forest",
        "models": models,
        "evaluation": evaluation,
        "top_features": features,
    }