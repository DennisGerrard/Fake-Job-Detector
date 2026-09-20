"""
Feature engineering — used by BOTH train.py and app.py later.

Why one shared file: whatever columns the model is trained on, it must be fed
the exact same columns, built the exact same way, at prediction time. If we
wrote this logic twice (once for training, once for the Flask app) they would
slowly drift apart and break predictions silently.
"""

import re
import numpy as np
import pandas as pd

# Numeric columns the Random Forest will read directly.
NUMERIC_FEATURES = [
    "telecommuting", "has_company_logo", "has_questions", "has_salary_range",
    "len_company_profile", "len_description", "len_requirements", "len_benefits",
    "n_words_description", "n_exclamations", "n_money_symbols",
    "uppercase_ratio", "digit_ratio", "has_email", "has_url", "has_phone",
    "n_red_flag_words",
]

# Categorical columns — the Random Forest needs these one-hot encoded (done in train.py).
CATEGORICAL_FEATURES = ["employment_type", "required_experience", "required_education"]

# Free-text columns that get glued together and handed to the KNN / TF-IDF model.
TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]

# Every raw column a posting needs before we can engineer features from it.
RAW_COLUMNS = TEXT_COLUMNS + CATEGORICAL_FEATURES + [
    "location", "salary_range", "telecommuting", "has_company_logo", "has_questions",
]

# Phrases that show up disproportionately often in scam postings.
RED_FLAG_WORDS = [
    "no experience", "work from home", "earn", "easy money", "urgent",
    "immediate start", "quick money", "unlimited income", "be your own boss",
    "wire transfer", "western union", "processing fee", "registration fee",
    "start today", "weekly pay", "daily pay", "guaranteed income",
]

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
URL_RE = re.compile(r"https?://|www\.")
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")


def _ratio(count, total):
    """Safe division — avoids a crash on empty strings."""
    return count / total if total else 0.0


def build_text_column(df):
    """Glue the free-text fields into one lowercase blob, for TF-IDF."""
    combined = df[TEXT_COLUMNS[0]].fillna("").astype(str)
    for col in TEXT_COLUMNS[1:]:
        combined = combined + " " + df[col].fillna("").astype(str)
    return combined.str.replace(r"\s+", " ", regex=True).str.strip().str.lower()


def engineer(df):
    """Add every engineered column the models need, on top of the raw ones.
    Returns ONE DataFrame — not a tuple."""
    out = df.copy()

    for col in RAW_COLUMNS:
        if col not in out.columns:
            out[col] = 0 if col in ("telecommuting", "has_company_logo", "has_questions") else ""

    for col in ("telecommuting", "has_company_logo", "has_questions"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].fillna("").astype(str).str.strip().replace("", "not specified")

    out["has_salary_range"] = out["salary_range"].fillna("").astype(str).str.strip().ne("").astype(int)

    for col in ("company_profile", "description", "requirements", "benefits"):
        out[f"len_{col}"] = out[col].fillna("").astype(str).str.len()

    desc = out["description"].fillna("").astype(str)
    out["n_words_description"] = desc.str.split().str.len().fillna(0).astype(int)

    raw_blob = out["title"].fillna("").astype(str) + " " + desc + " " + out["requirements"].fillna("").astype(str)

    out["n_exclamations"] = raw_blob.str.count("!")
    out["n_money_symbols"] = raw_blob.str.count(r"[$£€]")

    out["uppercase_ratio"] = [
        _ratio(sum(1 for c in t if c.isupper()), sum(1 for c in t if c.isalpha())) for t in raw_blob
    ]
    out["digit_ratio"] = [_ratio(sum(1 for c in t if c.isdigit()), len(t)) for t in raw_blob]

    out["has_email"] = raw_blob.apply(lambda t: int(bool(EMAIL_RE.search(t))))
    out["has_url"] = raw_blob.apply(lambda t: int(bool(URL_RE.search(t))))
    out["has_phone"] = raw_blob.apply(lambda t: int(bool(PHONE_RE.search(t))))

    text_blob = build_text_column(out)
    out["n_red_flag_words"] = text_blob.apply(lambda t: sum(1 for w in RED_FLAG_WORDS if w in t))
    out["text_blob"] = text_blob

    return out
