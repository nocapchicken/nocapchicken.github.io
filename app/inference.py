# AI-assisted (Claude Code, claude.ai) — https://claude.ai
# External libraries: scikit-learn (BSD-3), SHAP (MIT), rapidfuzz (MIT), googlemaps (Apache-2.0)
"""Trained artifacts load on first request (APP1)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from rapidfuzz import fuzz, utils as fuzz_utils

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"

# Binary classification: 0 = safe (A), 1 = flagged (B or C).
# Matches the binary reframing in scripts/model.py load_data(binary=True).
GRADE_LABELS = {0: "A", 1: "Flagged"}
GRADE_COLORS = {"A": "green", "Flagged": "red"}


def _log_grade_mapping() -> None:
    """Log the binary classification mapping for debuggability."""
    logger.info("Using binary classification: 0=A (safe), 1=Flagged (B or C)")


@dataclass
class PredictionResult:
    """All data surfaced to the frontend for a single restaurant lookup."""
    restaurant_name: str
    location: str
    predicted_grade: str
    grade_color: str
    confidence: float                      # 0–1
    google_rating: float | None
    google_review_count: int | None
    top_shap_features: list[dict]          # [{"feature": str, "impact": float}]
    divergence_warning: bool               # reviews look good but model flags restaurant
    actual_grade: str | None = None        # latest inspection grade from DHHS records
    actual_score: float | None = None      # latest inspection score (0-100)
    sample_reviews: list[str] = field(default_factory=list)
    error: str | None = None


@lru_cache(maxsize=1)
def _load_rf_model():
    """Load the trained Random Forest and log the grade mapping on first call."""
    path = MODELS_DIR / "random_forest.pkl"
    if not path.exists():
        logger.warning("Random Forest model not found at %s", path)
        return None
    model = joblib.load(path)
    _log_grade_mapping()
    return model


@lru_cache(maxsize=1)
def _load_feature_names() -> list[str]:
    path = MODELS_DIR / "rf_feature_names.pkl"
    if not path.exists():
        logger.warning("rf_feature_names.pkl not found — falling back to hardcoded features")
        return [
            "google_rating", "google_review_count_log",
            "review_word_count", "review_avg_word_len",
            "safety_keyword_count", "negative_phrase_count",
        ]
    return joblib.load(path)


@lru_cache(maxsize=1)
def _load_explainer():
    import shap
    model = _load_rf_model()
    if model is None:
        return None
    return shap.TreeExplainer(model)


@lru_cache(maxsize=1)
def _load_local_reviews() -> dict[str, dict]:
    """Load google_reviews.csv into a lookup keyed by lowercase google_name.

    Each value has: rating, review_count, reviews (list[str]), establishment_name.
    """
    import csv

    path = ROOT / "data" / "raw" / "google_reviews.csv"
    if not path.exists():
        logger.warning("Local reviews file not found at %s", path)
        return {}

    lookup: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            google_name = (row.get("google_name") or "").strip()
            if not google_name:
                continue
            key = google_name.lower()
            # Keep the entry with the highest review count per name
            review_count = float(row.get("google_review_count") or 0)
            if key in lookup and lookup[key]["review_count"] >= review_count:
                continue
            raw_reviews = row.get("google_reviews", "")
            reviews = [r.strip() for r in raw_reviews.split("|||") if r.strip()]
            rating_str = row.get("google_rating", "")
            try:
                rating = float(rating_str) if rating_str else None
            except ValueError:
                continue  # corrupted row (column shift)
            lookup[key] = {
                "rating": rating,
                "review_count": int(review_count),
                "reviews": reviews,
                "establishment_name": row.get("establishment_name", ""),
                "google_name": google_name,
                "state_id": (row.get("state_id") or "").strip(),
            }
    logger.info("Loaded %d local review entries", len(lookup))
    return lookup


@lru_cache(maxsize=1)
def _load_local_locations() -> dict[str, str]:
    """Build state_id -> 'City, NC' lookup from inspection files."""
    import csv
    import glob

    locations: dict[str, str] = {}
    for path in sorted(glob.glob(str(ROOT / "data" / "raw" / "inspections_*.csv"))):
        with open(path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                sid = row.get("state_id", "").strip()
                if sid and sid not in locations:
                    city = (row.get("city") or "").strip().title()
                    if city:
                        locations[sid] = f"{city}, NC"
    logger.info("Loaded %d location entries from inspection files", len(locations))
    return locations


@lru_cache(maxsize=1)
def _load_latest_grades() -> dict[str, dict]:
    """Build state_id -> {grade, score} for the most recent inspection."""
    import csv
    import glob

    grades: dict[str, dict] = {}
    for path in sorted(glob.glob(str(ROOT / "data" / "raw" / "inspections_*.csv"))):
        with open(path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                sid = row.get("state_id", "").strip()
                grade = (row.get("grade") or "").strip()
                date = (row.get("inspection_date") or "").strip()
                if not sid or not grade or grade.lower() == "nan":
                    continue
                prev_date = grades.get(sid, {}).get("date", "")
                if date >= prev_date:
                    try:
                        score = float(row.get("score") or 0)
                    except ValueError:
                        score = 0.0
                    grades[sid] = {"grade": grade, "score": score, "date": date}
    logger.info("Loaded %d latest grade entries from inspection files", len(grades))
    return grades


def _fetch_local(name: str) -> dict:
    """Look up restaurant from local CSV data using fuzzy matching."""
    lookup = _load_local_reviews()
    if not lookup:
        return {}

    # Try exact match first
    key = name.lower().strip()
    if key in lookup:
        entry = lookup[key]
    else:
        # Fuzzy match against all google_name keys
        best_key, best_score = None, 0
        for candidate_key in lookup:
            score = fuzz.token_sort_ratio(
                name, candidate_key, processor=fuzz_utils.default_process,
            )
            if score > best_score:
                best_score = score
                best_key = candidate_key
        if best_score < 65 or best_key is None:
            return {}
        entry = lookup[best_key]

    # Resolve location and actual grade via state_id
    location = ""
    actual_grade = None
    actual_score = None
    sid = entry.get("state_id", "")
    if sid:
        locations = _load_local_locations()
        location = locations.get(sid, "")
        grade_info = _load_latest_grades().get(sid, {})
        actual_grade = grade_info.get("grade")
        actual_score = grade_info.get("score")

    return {
        "rating": entry["rating"],
        "review_count": entry["review_count"],
        "reviews": entry["reviews"],
        "location": location,
        "actual_grade": actual_grade,
        "actual_score": actual_score,
    }


def _fetch_google_api(name: str) -> dict:
    """Live Google Places API fallback for restaurants not in local data."""
    api_key = os.getenv("GOOGLE_PLACES_API_KEY", "")
    if not api_key:
        return {}

    try:
        import googlemaps
        gmaps = googlemaps.Client(key=api_key)

        query = f"{name} restaurant NC"
        result = gmaps.find_place(query, input_type="textquery", fields=["place_id", "name"])
        candidates = result.get("candidates", [])
        if not candidates:
            return {}

        match_score = fuzz.token_sort_ratio(
            name, candidates[0].get("name", ""),
            processor=fuzz_utils.default_process,
        )
        if match_score < 65:
            return {}

        details = gmaps.place(
            candidates[0]["place_id"],
            fields=["name", "rating", "user_ratings_total", "reviews", "formatted_address"]
        )
        place = details.get("result", {})
        reviews = [rev["text"] for rev in place.get("reviews", [])]

        raw_address = place.get("formatted_address", "")
        parts = [p.strip() for p in raw_address.split(",")]
        location = ", ".join(parts[1:3]).strip() if len(parts) >= 3 else raw_address

        return {
            "rating": place.get("rating"),
            "review_count": place.get("user_ratings_total"),
            "reviews": reviews,
            "location": location,
        }
    except Exception as exc:
        logger.warning("Google API lookup failed: %s", exc)
        return {}


def _fetch_google(name: str) -> dict:
    """Look up restaurant data: local CSV first, live API as fallback."""
    result = _fetch_local(name)
    if result and result.get("rating") is not None:
        return result
    return _fetch_google_api(name)


@dataclass
class _FeatureData:
    """Feature vector plus the raw values needed by PredictionResult."""
    X: np.ndarray
    col_names: list[str]
    google_rating: float | None


def _build_feature_vector(google: dict) -> _FeatureData:
    """Align to the columns the RF was trained on.

    Must stay in sync with scripts/build_features.py _engineer_features().
    Text-derived features (safety keywords, negative phrases, word stats)
    are computed at inference time from the same review text.
    """
    import re

    google_rating = google.get("rating")
    google_count = google.get("review_count", 0) or 0
    reviews_text = " ".join(google.get("reviews", []))

    # Text-derived features (patterns imported from scripts/feature_constants.py)
    from scripts.feature_constants import NEGATIVE_PATTERN, SAFETY_PATTERN

    words = reviews_text.split() if reviews_text else []
    word_count = len(words)
    avg_word_len = (
        np.mean([len(w) for w in re.sub(r"[^\w\s]", "", reviews_text).split()])
        if words else 0.0
    )
    safety_count = len(re.findall(SAFETY_PATTERN, reviews_text.lower()))
    negative_count = len(re.findall(NEGATIVE_PATTERN, reviews_text.lower()))

    available = {
        "google_rating": google_rating or 0.0,
        "google_review_count_log": np.log1p(google_count),
        "review_word_count": float(word_count),
        "review_avg_word_len": avg_word_len,
        "safety_keyword_count": float(safety_count),
        "negative_phrase_count": float(negative_count),
    }

    feature_names = _load_feature_names()
    missing = [n for n in feature_names if n not in available]
    if missing:
        raise ValueError(
            f"Training/inference feature mismatch: {missing} expected by the model "
            "but not computed at inference time. Update _build_feature_vector()."
        )
    feature_values = [available[name] for name in feature_names]

    import pandas as pd
    X = pd.DataFrame([feature_values], columns=feature_names)

    return _FeatureData(
        X=X.values,
        col_names=feature_names,
        google_rating=google_rating,
    )


def _compute_shap(X: np.ndarray, col_names: list[str], pred_class: int) -> list[dict]:
    """Return top 3 SHAP feature impacts for the prediction."""
    explainer = _load_explainer()
    if explainer is None:
        return []

    try:
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            # Old SHAP API: list of arrays, one per class
            vals = shap_vals[pred_class][0]
        elif shap_vals.ndim == 3:
            # New SHAP API: (samples, features, classes)
            vals = shap_vals[0, :, pred_class]
        else:
            vals = shap_vals[0]

        impacts = [
            {"feature": name, "impact": float(val)}
            for name, val in zip(col_names, vals)
        ]
        impacts.sort(key=lambda item: abs(item["impact"]), reverse=True)
        return impacts[:3]
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        return []


def suggest_restaurants(name: str) -> list[str]:
    """Return up to 5 restaurant name suggestions from local data."""
    if len(name) < 2:
        return []

    lookup = _load_local_reviews()
    query = name.lower()
    scored = []
    for key, entry in lookup.items():
        if query in key:
            # Substring matches rank higher
            scored.append((100, key))
        else:
            score = fuzz.token_sort_ratio(
                name, key, processor=fuzz_utils.default_process,
            )
            if score >= 50:
                scored.append((score, key))

    scored.sort(key=lambda item: item[0], reverse=True)
    # Return the google_name in original casing (title-cased from the key)
    return [lookup[key]["google_name"] for _, key in scored[:5]]


def _unavailable(restaurant_name: str, error: str) -> PredictionResult:
    return PredictionResult(
        restaurant_name=restaurant_name,
        location="",
        predicted_grade="?",
        grade_color="gray",
        confidence=0.0,
        google_rating=None,
        google_review_count=None,
        top_shap_features=[],
        divergence_warning=False,
        error=error,
    )


def predict(restaurant_name: str) -> PredictionResult:
    """Fetch Google data, run RF inference, and return a PredictionResult."""
    model = _load_rf_model()
    if model is None:
        return _unavailable(restaurant_name, "Model not loaded — run python scripts/model.py first.")

    google = _fetch_google(restaurant_name)

    if google.get("rating") is None:
        return _unavailable(restaurant_name, "No review data found for this restaurant — prediction unavailable.")

    feat = _build_feature_vector(google)
    proba = model.predict_proba(feat.X)[0]
    pred_class = int(np.argmax(proba))
    confidence = float(proba[pred_class])
    predicted_grade = GRADE_LABELS.get(pred_class, "?")

    shap_features = _compute_shap(feat.X, feat.col_names, pred_class)

    divergence_warning = (
        predicted_grade == "Flagged"
        and feat.google_rating is not None
        and feat.google_rating >= 4.0
    )

    return PredictionResult(
        restaurant_name=restaurant_name,
        location=google.get("location", ""),
        predicted_grade=predicted_grade,
        grade_color=GRADE_COLORS.get(predicted_grade, "gray"),
        confidence=confidence,
        google_rating=feat.google_rating,
        google_review_count=google.get("review_count"),
        top_shap_features=shap_features,
        divergence_warning=divergence_warning,
        actual_grade=google.get("actual_grade"),
        actual_score=google.get("actual_score"),
        sample_reviews=google.get("reviews", [])[:3],
    )
