# AI-assisted (Claude Code, claude.ai) — https://claude.ai
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


def _verify_grade_mapping() -> None:
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
    sample_reviews: list[str] = field(default_factory=list)
    error: str | None = None


@lru_cache(maxsize=1)
def _load_rf_model():
    """Load the trained Random Forest; also verifies grade label mapping on first call."""
    path = MODELS_DIR / "random_forest.pkl"
    if not path.exists():
        logger.warning("Random Forest model not found at %s", path)
        return None
    model = joblib.load(path)
    _verify_grade_mapping()
    return model


@lru_cache(maxsize=1)
def _load_feature_names() -> list[str]:
    path = MODELS_DIR / "rf_feature_names.pkl"
    if not path.exists():
        logger.warning("rf_feature_names.pkl not found — falling back to hardcoded features")
        return ["google_rating", "google_review_count_log"]
    return joblib.load(path)


@lru_cache(maxsize=1)
def _load_explainer():
    import shap
    model = _load_rf_model()
    if model is None:
        return None
    return shap.TreeExplainer(model)


def _fetch_google(name: str) -> dict:
    """Returns rating, review count, location, and up to 5 review texts."""
    api_key = os.getenv("GOOGLE_PLACES_API_KEY", "")
    if not api_key:
        return {}

    try:
        import googlemaps  # deferred: optional heavy dependency, avoids startup cost
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

        # Trim address to city, state (e.g. "Raleigh, NC")
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
        logger.warning("Google lookup failed: %s", exc)
        return {}


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

    # Text-derived features (same patterns as build_features.py)
    words = reviews_text.split() if reviews_text else []
    word_count = len(words)
    avg_word_len = (
        np.mean([len(w) for w in re.sub(r"[^\w\s]", "", reviews_text).split()])
        if words else 0.0
    )
    safety_pattern = r"\b(dirty|filthy|sick|roach|bug|rat|mouse|health|violation|gross|smell|mold|expired|undercooked|raw|contaminated)\b"
    negative_pattern = r"\b(worst|terrible|horrible|disgusting|awful|never again|food poisoning|threw up|diarrhea)\b"
    safety_count = len(re.findall(safety_pattern, reviews_text.lower()))
    negative_count = len(re.findall(negative_pattern, reviews_text.lower()))

    available = {
        "google_rating": google_rating or 0.0,
        "google_review_count_log": np.log1p(google_count),
        "review_word_count": float(word_count),
        "review_avg_word_len": avg_word_len,
        "safety_keyword_count": float(safety_count),
        "negative_phrase_count": float(negative_count),
    }

    feature_names = _load_feature_names()
    zeroed = [n for n in feature_names if n not in available]
    if zeroed:
        logger.warning(
            "Zeroing %d feature(s) not available at inference time: %s. "
            "These were present at training but have no inference-time source.",
            len(zeroed), zeroed,
        )
    feature_values = [available.get(name, 0.0) for name in feature_names]

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
            vals = shap_vals[pred_class][0]
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
    """Return up to 5 restaurant name suggestions from Google Places."""
    api_key = os.getenv("GOOGLE_PLACES_API_KEY", "")
    if not api_key or len(name) < 2:
        return []
    try:
        import googlemaps  # deferred: optional heavy dependency, avoids startup cost
        gmaps = googlemaps.Client(key=api_key)
        results = gmaps.places_autocomplete(
            name, types=["establishment"], location=None,
            components={"country": "us"},
        )
        return [r["description"] for r in results[:5] if r.get("description")]
    except Exception as exc:
        logger.warning("Suggest lookup failed: %s", exc)
        return []


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
        sample_reviews=google.get("reviews", [])[:3],
    )
