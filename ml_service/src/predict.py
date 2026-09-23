"""Load trained pipeline and predict labels for text inputs."""

from pathlib import Path

import joblib

from src import config
from src.utils import sqli_tokenizer

LABEL_NORMAL = "Normal"
LABEL_SQLI = "SQL Injection"


def load_pipeline(path=None):
    """
    Load the saved sklearn Pipeline from disk.
    Returns None if the file does not exist.
    """
    p = Path(path) if path is not None else config.MODEL_PIPELINE
    if not p.is_file():
        return None
    return joblib.load(p)


def _label_name(y):
    return LABEL_SQLI if int(y) == 1 else LABEL_NORMAL


def predict_with_confidence(text, pipeline=None):
    """
    Predict label and confidence for a single input string.
    Returns (label, confidence) where:
        - label      : 'Normal' or 'SQL Injection'
        - confidence : float between 0 and 1 for the predicted class
    Returns (None, None) if input is empty or model is not loaded.
    """
    if text is None or (isinstance(text, str) and not text.strip()):
        return None, None

    pipe = pipeline if pipeline is not None else load_pipeline()
    if pipe is None:
        return None, None

    t = str(text).strip()
    y = int(pipe.predict([t])[0])
    label = _label_name(y)

    confidence = None
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba([t])[0]
        confidence = float(proba[y])

    return label, confidence