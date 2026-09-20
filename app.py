"""Flask backend: loads both models, exposes /predict."""

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RAW_COLUMNS, engineer

MODEL_DIR = Path("models")
app = Flask(__name__)

_rf = None
_knn = None
_meta = None


def load_models():
    """Load once and cache. Raises if train.py hasn't been run yet."""
    global _rf, _knn, _meta
    if _rf is None:
        _rf = joblib.load(MODEL_DIR / "random_forest.joblib")
        _knn = joblib.load(MODEL_DIR / "knn_text.joblib")
        _meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    return _rf, _knn, _meta


def posting_to_frame(posting: dict) -> pd.DataFrame:
    """One submitted job posting (dict) -> one-row engineered DataFrame."""
    row = {col: posting.get(col, "") for col in RAW_COLUMNS}
    for flag in ("telecommuting", "has_company_logo", "has_questions"):
        row[flag] = int(posting.get(flag, 0) or 0)
    return engineer(pd.DataFrame([row]))


@app.route("/health")
def health():
    try:
        load_models()
        return jsonify({"status": "ok"})
    except FileNotFoundError:
        return jsonify({"status": "no models — run train.py"}), 503


@app.route("/", methods=["GET"])
def index():
    """Serves the main web interface."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    rf, knn, meta = load_models()

    posting = request.get_json(silent=True) or {}
    if not posting.get("description", "").strip():
        return jsonify({"error": "description is required"}), 400

    frame = posting_to_frame(posting)

    rf_prob = float(rf.predict_proba(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[0, 1])
    knn_prob = float(knn.predict_proba(frame["text_blob"])[0, 1])
    score = meta["rf_weight"] * rf_prob + meta["knn_weight"] * knn_prob
    is_fake = score >= meta["threshold"]

    return jsonify({
        "verdict": "fake" if is_fake else "genuine",
        "score": round(score, 4),
        "random_forest_score": round(rf_prob, 4),
        "knn_score": round(knn_prob, 4),
        "threshold": meta["threshold"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)