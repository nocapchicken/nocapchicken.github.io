# AI-assisted (Claude Code, claude.ai) — https://claude.ai
# External libraries: scikit-learn (BSD-3), rapidfuzz (MIT)
"""Merges inspection + review data via fuzzy matching into a feature matrix."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# Minimum fuzzy match score (0–100) to accept a name+address link
MATCH_THRESHOLD = 50


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
    """Write data/processed/features.csv from raw inspections + reviews."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    merge_inspection_years(RAW_DIR)

    inspections = _load_inspections()
    google = _load_reviews("google_reviews.csv")

    merged = _merge(inspections, google)
    merged = _engineer_features(merged)
    merged = _encode_target(merged)

    out_path = PROCESSED_DIR / "features.csv"
    merged.to_csv(out_path, index=False)
    logger.info("Feature matrix written to %s (%d rows, %d cols)", out_path, *merged.shape)
    return merged


def _load_inspections() -> pd.DataFrame:
    """Read merged inspections, coerce score to numeric, normalize grade."""
    df = pd.read_csv(RAW_DIR / "inspections.csv")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    df["grade"] = df["grade"].str.strip().str.upper()
    return df


def _load_reviews(filename: str) -> pd.DataFrame:
    """Filter to matches above MATCH_THRESHOLD."""
    path = RAW_DIR / filename
    if not path.exists() or not path.read_text().strip():
        logger.warning("%s not found or empty, skipping", path)
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["match_score"] = pd.to_numeric(df["match_score"], errors="coerce")
    df = df[df["match_score"] >= MATCH_THRESHOLD].copy()
    return df


def _merge(inspections: pd.DataFrame, google: pd.DataFrame) -> pd.DataFrame:
    """Left-join inspections with google on state_id (stable restaurant identifier)."""
    df = inspections.copy()

    if not google.empty:
        google_cols = google.drop(columns=["establishment_name", "inspection_id"], errors="ignore")
        df = df.merge(google_cols, on="state_id", how="left", suffixes=("", "_google"))

    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric, temporal, and text-derived features from inspection + review data."""
    # Establishment type as a numeric code (e.g. "1 - Restaurant" → 1)
    df["establishment_type_code"] = (
        df["establishment_type"].str.split(" - ").str[0].astype(float, errors="ignore")
    )

    # County as-is — inspection culture varies by county
    df["county_code"] = pd.to_numeric(df["county_code"], errors="coerce")

    # Temporal features — seasonal patterns and inspection year
    inspection_dt = pd.to_datetime(df["inspection_date"], errors="coerce")
    df["inspection_month"] = inspection_dt.dt.month
    df["inspection_year"] = inspection_dt.dt.year

    # Combined review text (for NLP models)
    if "google_reviews" in df.columns:
        df["combined_reviews"] = df["google_reviews"].fillna("").str.strip()

    # Review volume signal (sparse — only present for Google-matched rows)
    if "google_review_count" in df.columns:
        df["google_review_count_log"] = np.log1p(df["google_review_count"].fillna(0))

    # Text-derived features for RF (gives structured models access to review content)
    if "combined_reviews" in df.columns:
        reviews = df["combined_reviews"].fillna("")
        df["review_word_count"] = reviews.str.split().str.len().fillna(0).astype(int)
        df["review_avg_word_len"] = (
            reviews.str.replace(r"[^\w\s]", "", regex=True)
            .str.split()
            .apply(lambda words: np.mean([len(w) for w in words]) if words else 0)
        )

        # Safety-adjacent keywords that may correlate with inspection outcomes
        safety_terms = r"\b(dirty|filthy|sick|roach|bug|rat|mouse|health|violation|gross|smell|mold|expired|undercooked|raw|contaminated)\b"
        df["safety_keyword_count"] = reviews.str.lower().str.count(safety_terms)

        # Negative sentiment proxy: count of strongly negative phrases
        negative_terms = r"\b(worst|terrible|horrible|disgusting|awful|never again|food poisoning|threw up|diarrhea)\b"
        df["negative_phrase_count"] = reviews.str.lower().str.count(negative_terms)

    return df


def _encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to valid grades (A/B/C) and add label-encoded target column.

    Saves the fitted LabelEncoder to models/grade_encoder.pkl so inference.py
    can verify its GRADE_LABELS mapping matches the training encoding.
    """
    valid_grades = ["A", "B", "C"]
    df = df[df["grade"].isin(valid_grades)].copy()

    le = LabelEncoder()
    df["grade_encoded"] = le.fit_transform(df["grade"])

    # Persist so inference.py can cross-check; alphabetical order gives A=0, B=1, C=2
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(le, MODELS_DIR / "grade_encoder.pkl")
    logger.info("Grade encoder saved: %s", dict(zip(le.classes_, le.transform(le.classes_))))

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_features()
