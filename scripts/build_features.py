"""
build_features.py — Feature engineering pipeline for nocapchicken.

Reads raw inspection + review data, merges them via fuzzy matching,
and produces a clean feature matrix ready for modeling.

Usage:
    python scripts/build_features.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Minimum fuzzy match score (0–100) to accept a name+address link
MATCH_THRESHOLD = 70


def build_features() -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Returns:
        DataFrame with one row per matched establishment, ready for modeling.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    inspections = _load_inspections()
    yelp = _load_reviews("yelp_reviews.csv", prefix="yelp")
    google = _load_reviews("google_reviews.csv", prefix="google")

    merged = _merge(inspections, yelp, google)
    merged = _engineer_features(merged)
    merged = _encode_target(merged)

    out_path = PROCESSED_DIR / "features.csv"
    merged.to_csv(out_path, index=False)
    logger.info("Feature matrix written to %s (%d rows, %d cols)", out_path, *merged.shape)
    return merged


def _load_inspections() -> pd.DataFrame:
    """Load and clean raw inspection records."""
    df = pd.read_csv(RAW_DIR / "inspections.csv")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    df["grade"] = df["grade"].str.strip().str.upper()
    return df


def _load_reviews(filename: str, prefix: str) -> pd.DataFrame:
    """Load a reviews file and filter to high-confidence matches."""
    path = RAW_DIR / filename
    if not path.exists():
        logger.warning("%s not found, skipping", path)
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df[df["match_score"] >= MATCH_THRESHOLD].copy()
    # Rename platform-specific columns to avoid collisions on merge
    df = df.rename(columns={c: c for c in df.columns})
    return df


def _merge(inspections: pd.DataFrame, yelp: pd.DataFrame, google: pd.DataFrame) -> pd.DataFrame:
    """Left-join inspections with yelp and google on inspection_id."""
    df = inspections.copy()

    if not yelp.empty:
        df = df.merge(yelp.drop(columns=["establishment_name"], errors="ignore"),
                      left_index=True, right_on="inspection_id", how="left")

    if not google.empty:
        df = df.merge(google.drop(columns=["establishment_name"], errors="ignore"),
                      left_index=True, right_on="inspection_id", how="left", suffixes=("", "_google"))

    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features from raw columns."""
    # Platform agreement: difference between Yelp and Google ratings
    if "yelp_rating" in df.columns and "google_rating" in df.columns:
        df["rating_delta"] = (df["yelp_rating"] - df["google_rating"]).abs()

    # Combined review text (for NLP models)
    text_cols = [c for c in ["yelp_reviews", "google_reviews"] if c in df.columns]
    if text_cols:
        df["combined_reviews"] = df[text_cols].fillna("").agg(" ".join, axis=1).str.strip()

    # Review volume signal
    for col in ["yelp_review_count", "google_review_count"]:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].fillna(0))

    return df


def _encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the target variable.

    Primary target: grade (A/B/C) as a 3-class classification problem.
    Secondary target: score as a continuous regression target.
    """
    valid_grades = ["A", "B", "C"]
    df = df[df["grade"].isin(valid_grades)].copy()

    le = LabelEncoder()
    df["grade_encoded"] = le.fit_transform(df["grade"])

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_features()
