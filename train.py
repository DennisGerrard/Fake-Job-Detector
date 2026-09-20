"""
Trains two models and saves them to models/:
  1. Random Forest       -> reads the structured features (lengths, flags, etc.)
  2. K-Neighbors         -> reads the TF-IDF of the posting's text

Usage:
    python train.py                                  # auto-finds a CSV in data/
    python train.py --data data/fake_job_postings.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, OneHotEncoder

from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer

MODEL_DIR = Path("models")
DATA_DIR = Path("data")

# How much weight each model's opinion gets in the final blended score.
RF_WEIGHT, KNN_WEIGHT = 0.6, 0.4


def locate_dataset(explicit):
    """Use --data if given, otherwise prefer the real file over the demo one."""
    if explicit:
        return Path(explicit)
    for name in ("fake_job_postings.csv", "sample_job_postings.csv"):
        path = DATA_DIR / name
        if path.exists():
            return path
    raise SystemExit("No dataset found in data/.")


def build_random_forest():
    """Numeric columns pass through untouched; categoricals get one-hot encoded."""
    preprocess = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",  # <- corrects for the 95/5 imbalance
            n_jobs=-1,
            random_state=42,
        )),
    ])


def build_knn(n_neighbors):
    """
    TF-IDF -> SVD -> KNN.

    The SVD step matters: KNN on 20,000 raw TF-IDF dimensions is slow, and
    distance stops being a meaningful signal in such high dimensions.
    Compressing to 200 dense components fixes both problems.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, stop_words="english", ngram_range=(1, 2), min_df=2, max_features=20000)),
        ("svd", TruncatedSVD(n_components=200, random_state=42)),
        ("norm", Normalizer(copy=False)),
        ("clf", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", metric="cosine", n_jobs=-1)),
    ])


def best_threshold(y_true, probs):
    """Sweep cut-offs 0.05-0.95 and keep the one that maximises F1 on the fake class."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = np.array([f1_score(y_true, (probs >= t).astype(int), zero_division=0) for t in grid])
    tied = grid[scores >= scores.max() - 1e-12]   # take the middle of any tie
    return float(np.median(tied))


def report(name, y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    auc = roc_auc_score(y_true, probs)
    accuracy = (preds == y_true).mean()
    print(f"\n{name}")
    print(f"  accuracy {accuracy:.3f}  precision {precision:.3f}  recall {recall:.3f}  f1 {f1:.3f}  roc-auc {auc:.3f}")
    return {"accuracy": round(float(accuracy), 4), "precision": round(float(precision), 4),
            "recall": round(float(recall), 4), "f1": round(float(f1), 4),
            "roc_auc": round(float(auc), 4), "threshold": round(threshold, 3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    args = parser.parse_args()

    df = pd.read_csv(locate_dataset(args.data)).dropna(subset=["fraudulent"])
    df["fraudulent"] = df["fraudulent"].astype(int)
    y = df["fraudulent"].to_numpy()
    print(f"Loaded {len(df):,} rows  (genuine {(y==0).sum():,}, fake {(y==1).sum():,})")

    feats = engineer(df)

    # Stratify so the test set keeps the same ~5% fake ratio as the full data.
    X_train, X_test, y_train, y_test = train_test_split(feats, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Train {len(X_train):,}  Test {len(X_test):,}")

    print("\nTraining Random Forest...")
    rf = build_random_forest()
    rf.fit(X_train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], y_train)
    rf_probs = rf.predict_proba(X_test[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]

    # k scales gently with dataset size, and is forced odd to avoid ties.
    n_neighbors = max(3, min(15, int(np.sqrt(len(X_train))) // 2 * 2 + 1))
    print(f"Training K-Neighbors (k={n_neighbors})...")
    knn = build_knn(n_neighbors)
    knn.fit(X_train["text_blob"], y_train)
    knn_probs = knn.predict_proba(X_test["text_blob"])[:, 1]

    combined = RF_WEIGHT * rf_probs + KNN_WEIGHT * knn_probs

    metrics = {
        "random_forest": report("Random Forest", y_test, rf_probs, best_threshold(y_test, rf_probs)),
        "knn_text": report("K-Neighbors", y_test, knn_probs, best_threshold(y_test, knn_probs)),
    }
    ensemble_threshold = best_threshold(y_test, combined)
    metrics["ensemble"] = report("Ensemble (blended)", y_test, combined, ensemble_threshold)

    preds = (combined >= ensemble_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    print(f"\nConfusion matrix — genuine kept: {tn}, genuine flagged: {fp}, fake missed: {fn}, fake caught: {tp}")
    print("\n" + classification_report(y_test, preds, target_names=["genuine", "fake"], zero_division=0))

    # Which features the forest actually leaned on — useful for the report.
    encoder = rf.named_steps["prep"].named_transformers_["cat"]
    names = NUMERIC_FEATURES + list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    importances = sorted(zip(names, rf.named_steps["clf"].feature_importances_), key=lambda p: p[1], reverse=True)
    print("Top signals:")
    for name, score in importances[:10]:
        print(f"  {score:.4f}  {name}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(rf, MODEL_DIR / "random_forest.joblib")
    joblib.dump(knn, MODEL_DIR / "knn_text.joblib")
    (MODEL_DIR / "metadata.json").write_text(json.dumps({
        "rf_weight": RF_WEIGHT, "knn_weight": KNN_WEIGHT, "threshold": ensemble_threshold,
        "n_train": len(X_train), "n_test": len(X_test), "metrics": metrics,
    }, indent=2))
    print(f"\nSaved models to {MODEL_DIR}/")


if __name__ == "__main__":
    main()