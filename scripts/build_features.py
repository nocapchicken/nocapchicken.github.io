# AI-assisted (Claude Code, claude.ai) — https://claude.ai
"""Merges inspection + review data via fuzzy matching into a feature matrix."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Minimum fuzzy match score (0–100) to accept a name+address link
MATCH_THRESHOLD = 70


def merge_inspection_years(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Deduplicates on (state_id, inspection_date) and writes inspections.csv."""
    year_files = sorted(raw_dir.glob("inspections_*.csv"))
    if not year_files:
        raise FileNotFoundError(
            f"No inspections_{{year}}.csv files found in {raw_dir}. "
            "Run python3 setup.py first."
        )

    df = pd.concat([pd.read_csv(f) for f in year_files], ignore_index=True)
    df = df.drop_duplicates(subset=["state_id", "inspection_date"])
    df = df.sort_values(["inspection_date", "establishment_name"]).reset_index(drop=True)

    out_path = raw_dir / "inspections.csv"
    df.to_csv(out_path, index=False)
    logger.info(
        "Merged %d year file(s) → inspections.csv (%d rows)",
        len(year_files), len(df),
    )
    return df


def build_features() -> pd.DataFrame:
    """Write data/processed/features.csv."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    merge_inspection_years(RAW_DIR)

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
    df = pd.read_csv(RAW_DIR / "inspections.csv")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    df["grade"] = df["grade"].str.strip().str.upper()
    return df


def _load_reviews(filename: str, prefix: str) -> pd.DataFrame:
    """Load a reviews file and filter to high-confidence matches."""
    path = RAW_DIR / filename
    if not path.exists() or not path.read_text().strip():
        logger.warning("%s not found or empty, skipping", path)
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df[df["match_score"] >= MATCH_THRESHOLD].copy()
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
    # Rating delta: platform agreement signal
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
    """Filter to valid grades (A/B/C) and add label-encoded target column."""
    valid_grades = ["A", "B", "C"]
    df = df[df["grade"].isin(valid_grades)].copy()

    le = LabelEncoder()
    df["grade_encoded"] = le.fit_transform(df["grade"])

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_features()
