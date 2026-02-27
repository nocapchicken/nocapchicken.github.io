# AI-assisted (Claude Code, claude.ai) — https://claude.ai
"""Loads trained artifacts at startup. No training happens here (APP1)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import requests
import shap
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"

GRADE_LABELS = {0: "A", 1: "B", 2: "C"}
GRADE_COLORS = {"A": "green", "B": "yellow", "C": "red"}


@dataclass
class PredictionResult:
    """All data surfaced to the frontend for a single restaurant lookup."""
    restaurant_name: str
    location: str
    predicted_grade: str
    grade_color: str
    confidence: float                      # 0–1
    yelp_rating: float | None
    yelp_review_count: int | None
    google_rating: float | None
    google_review_count: int | None
    rating_delta: float | None             # |yelp - google|; divergence signal
    top_shap_features: list[dict]          # [{"feature": str, "impact": float}]
    divergence_warning: bool               # reviews look good but model predicts C
    sample_reviews: list[str] = field(default_factory=list)
    error: str | None = None


@lru_cache(maxsize=1)
def _load_rf_model():
    path = MODELS_DIR / "random_forest.pkl"
    if not path.exists():
        logger.warning("Random Forest model not found at %s", path)
        return None
    return joblib.load(path)


@lru_cache(maxsize=1)
def _load_explainer():
    model = _load_rf_model()
    if model is None:
        return None
    return shap.TreeExplainer(model)


def _fetch_yelp(name: str, city: str) -> dict:
    """Fetch Yelp business metadata and up to 3 reviews via RapidAPI."""
    api_key = os.getenv("RAPIDAPI_KEY", "")
    if not api_key:
        return {}

    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "yelp-business-api.p.rapidapi.com",
    }

    try:
        search = requests.get(
            "https://yelp-business-api.p.rapidapi.com/search",
            headers=headers,
            params={"term": name, "location": f"{city}, NC", "limit": "1"},
            timeout=8,
        )
        businesses = search.json().get("businesses", [])
        if not businesses:
            return {}

        biz = businesses[0]
        match_score = fuzz.token_sort_ratio(name, biz.get("name", ""))
        if match_score < 65:
            return {}

        rev_resp = requests.get(
            "https://yelp-business-api.p.rapidapi.com/reviews",
            headers=headers,
            params={"business_id": biz["id"]},
            timeout=8,
        )
        reviews = [r["text"] for r in rev_resp.json().get("reviews", [])[:3]]

        return {
            "rating": biz.get("rating"),
            "review_count": biz.get("review_count"),
            "reviews": reviews,
        }
    except Exception as exc:
        logger.warning("Yelp lookup failed: %s", exc)
        return {}


def _fetch_google(name: str, city: str) -> dict:
    """Fetch Google Places rating and up to 5 reviews."""
    api_key = os.getenv("GOOGLE_PLACES_API_KEY", "")
    if not api_key:
        return {}

    try:
        import googlemaps
        gmaps = googlemaps.Client(key=api_key)

        query = f"{name} restaurant {city} NC"
        result = gmaps.find_place(query, input_type="textquery", fields=["place_id", "name"])
        candidates = result.get("candidates", [])
        if not candidates:
            return {}

        match_score = fuzz.token_sort_ratio(name, candidates[0].get("name", ""))
        if match_score < 65:
            return {}

        details = gmaps.place(
            candidates[0]["place_id"],
            fields=["name", "rating", "user_ratings_total", "reviews"]
        )
        r = details.get("result", {})
        reviews = [rev["text"] for rev in r.get("reviews", [])]

        return {
            "rating": r.get("rating"),
            "review_count": r.get("user_ratings_total"),
            "reviews": reviews,
        }
    except Exception as exc:
        logger.warning("Google lookup failed: %s", exc)
        return {}


def _build_feature_vector(yelp: dict, google: dict) -> tuple[np.ndarray, list[str]]:
    """Must match the columns produced by build_features.py."""
    yelp_rating = yelp.get("rating")
    google_rating = google.get("rating")
    yelp_count = yelp.get("review_count", 0) or 0
    google_count = google.get("review_count", 0) or 0

    rating_delta = None
    if yelp_rating is not None and google_rating is not None:
        rating_delta = abs(yelp_rating - google_rating)

    features = {
        "yelp_rating": yelp_rating or 0.0,
        "google_rating": google_rating or 0.0,
        "rating_delta": rating_delta or 0.0,
        "yelp_review_count_log": np.log1p(yelp_count),
        "google_review_count_log": np.log1p(google_count),
    }

    X = np.array(list(features.values())).reshape(1, -1)
    col_names = list(features.keys())
    return X, col_names


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


def predict(restaurant_name: str, city: str) -> PredictionResult:
    """Run end-to-end inference for a restaurant."""
    model = _load_rf_model()
    if model is None:
        return PredictionResult(
            restaurant_name=restaurant_name,
            location=city,
            predicted_grade="?",
            grade_color="gray",
            confidence=0.0,
            yelp_rating=None,
            yelp_review_count=None,
            google_rating=None,
            google_review_count=None,
            rating_delta=None,
            top_shap_features=[],
            divergence_warning=False,
            error="Model not loaded — run python scripts/model.py first.",
        )

    yelp = _fetch_yelp(restaurant_name, city)
    google = _fetch_google(restaurant_name, city)

    X, col_names = _build_feature_vector(yelp, google)
    proba = model.predict_proba(X)[0]
    pred_class = int(np.argmax(proba))
    confidence = float(proba[pred_class])
    predicted_grade = GRADE_LABELS.get(pred_class, "?")
    grade_color = GRADE_COLORS.get(predicted_grade, "gray")

    shap_features = _compute_shap(X, col_names, pred_class)

    yelp_rating = yelp.get("rating")
    google_rating = google.get("rating")

    rating_delta = (
        round(abs(yelp_rating - google_rating), 2)
        if yelp_rating is not None and google_rating is not None
        else None
    )

    # Divergence flag: high platform ratings but model predicts C
    platform_ratings = [r for r in (yelp_rating, google_rating) if r is not None]
    avg_platform_rating = np.mean(platform_ratings) if platform_ratings else None
    divergence_warning = (
        predicted_grade == "C"
        and avg_platform_rating is not None
        and avg_platform_rating >= 4.0
    )

    return PredictionResult(
        restaurant_name=restaurant_name,
        location=city,
        predicted_grade=predicted_grade,
        grade_color=grade_color,
        confidence=confidence,
        yelp_rating=yelp_rating,
        yelp_review_count=yelp.get("review_count"),
        google_rating=google_rating,
        google_review_count=google.get("review_count"),
        rating_delta=rating_delta,
        top_shap_features=shap_features,
        divergence_warning=divergence_warning,
        sample_reviews=(yelp.get("reviews", []) + google.get("reviews", []))[:3],
    )
