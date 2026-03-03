# AI-assisted (Claude Code, claude.ai) — https://claude.ai
"""Local inference using Tennessee restaurant data (no API calls)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "nashville"

GRADE_LABELS = {0: "A", 1: "B", 2: "C"}
GRADE_COLORS = {"A": "green", "B": "yellow", "C": "red"}


def _score_to_grade(score: float) -> str:
    """Convert TN inspection score (0-100) to letter grade."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    return "C"


@dataclass
class PredictionResult:
    """All data surfaced to the frontend for a single restaurant lookup."""
    restaurant_name: str
    location: str
    predicted_grade: str
    grade_color: str
    confidence: float
    yelp_rating: float | None
    yelp_review_count: int | None
    google_rating: float | None
    google_review_count: int | None
    rating_delta: float | None
    top_shap_features: list[dict]
    divergence_warning: bool
    sample_reviews: list[str] = field(default_factory=list)
    actual_grade: str | None = None
    actual_score: float | None = None
    error: str | None = None


@lru_cache(maxsize=1)
def _load_rf_model():
    """Load random forest model."""
    path = MODELS_DIR / "random_forest.pkl"
    if not path.exists():
        logger.warning("Random Forest model not found at %s", path)
        return None
    return joblib.load(path)


@lru_cache(maxsize=1)
def _load_feature_names() -> list[str]:
    """Load feature names used by the model."""
    path = MODELS_DIR / "rf_feature_names.pkl"
    if not path.exists():
        logger.warning("rf_feature_names.pkl not found — using defaults")
        return ["google_rating", "google_review_count_log"]
    return joblib.load(path)


@lru_cache(maxsize=1)
def _load_explainer():
    """Load SHAP explainer for model."""
    import shap
    model = _load_rf_model()
    if model is None:
        return None
    return shap.TreeExplainer(model)


@lru_cache(maxsize=1)
def _load_reviews_df() -> pd.DataFrame:
    """Load Tennessee reviews data."""
    path = DATA_DIR / "reviews" / "tn_restaurant_reviews.csv"
    if not path.exists():
        logger.warning("Reviews data not found at %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


@lru_cache(maxsize=1)
def _load_inspections_df() -> pd.DataFrame:
    """Load Tennessee inspections data."""
    path = DATA_DIR / "inspections" / "tn_inspections_raw.csv"
    if not path.exists():
        logger.warning("Inspections data not found at %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


@lru_cache(maxsize=1)
def _get_unique_restaurants() -> list[dict]:
    """Get unique restaurants from reviews data."""
    df = _load_reviews_df()
    if df.empty:
        return []

    unique = df.drop_duplicates(subset=['business_id'])[
        ['business_id', 'business_name', 'city', 'address', 'business_avg_stars']
    ].to_dict('records')
    return unique


def _find_restaurant(name: str, city: str) -> dict | None:
    """Find best matching restaurant using fuzzy matching."""
    restaurants = _get_unique_restaurants()
    if not restaurants:
        return None

    city_lower = city.lower().strip()
    city_filtered = [
        r for r in restaurants
        if str(r.get('city', '')).lower().strip() == city_lower
    ]

    search_list = city_filtered if city_filtered else restaurants
    if not search_list:
        return None

    names = [r['business_name'] for r in search_list]
    match = process.extractOne(name, names, scorer=fuzz.token_sort_ratio)

    if match and match[1] >= 50:
        idx = names.index(match[0])
        return search_list[idx]

    return None


def _get_restaurant_reviews(business_id: str) -> list[str]:
    """Get review texts for a restaurant."""
    df = _load_reviews_df()
    if df.empty:
        return []

    reviews = df[df['business_id'] == business_id]['review_text'].dropna().tolist()
    return reviews[:5]


def _get_restaurant_stats(business_id: str) -> dict:
    """Compute review statistics for a restaurant."""
    df = _load_reviews_df()
    if df.empty:
        return {}

    biz_df = df[df['business_id'] == business_id]
    if biz_df.empty:
        return {}

    avg_stars = biz_df['business_avg_stars'].iloc[0]
    review_count = len(biz_df)

    return {
        'rating': float(avg_stars) if pd.notna(avg_stars) else None,
        'review_count': int(review_count),
    }


def _find_inspection(name: str, city: str) -> dict | None:
    """Find most recent inspection for a restaurant."""
    df = _load_inspections_df()
    if df.empty:
        return None

    city_lower = city.lower().strip()
    city_df = df[df['city'].str.lower().str.strip() == city_lower]

    if city_df.empty:
        city_df = df

    names = city_df['establishmentName'].tolist()
    match = process.extractOne(name, names, scorer=fuzz.token_sort_ratio)

    if match and match[1] >= 50:
        matched_rows = city_df[city_df['establishmentName'] == match[0]]
        if not matched_rows.empty:
            matched_rows = matched_rows.sort_values('inspectionDate', ascending=False)
            row = matched_rows.iloc[0]
            return {
                'name': row['establishmentName'],
                'score': row['score'],
                'date': row['inspectionDate'],
                'city': row['city'],
            }

    return None


def _build_feature_vector(stats: dict) -> tuple[pd.DataFrame, list[str], float | None]:
    """Build feature vector from restaurant stats."""
    rating = stats.get('rating')
    review_count = stats.get('review_count', 0) or 0

    available = {
        'yelp_rating': rating or 0.0,
        'google_rating': rating or 0.0,
        'rating_delta': 0.0,
        'yelp_review_count_log': np.log1p(review_count),
        'google_review_count_log': np.log1p(review_count),
    }

    feature_names = _load_feature_names()
    feature_values = [available.get(name, 0.0) for name in feature_names]

    # Return as DataFrame to preserve feature names
    df = pd.DataFrame([feature_values], columns=feature_names)
    return df, feature_names, rating


def _compute_shap(feature_df: pd.DataFrame, col_names: list[str], pred_class: int) -> list[dict]:
    """Compute SHAP values for prediction."""
    explainer = _load_explainer()
    if explainer is None:
        return []

    try:
        shap_vals = explainer.shap_values(feature_df)
        if isinstance(shap_vals, list):
            vals = np.array(shap_vals[pred_class]).flatten()
        else:
            vals = np.array(shap_vals).flatten()

        impacts = [
            {"feature": name, "impact": float(val)}
            for name, val in zip(col_names, vals)
        ]
        impacts.sort(key=lambda item: abs(item["impact"]), reverse=True)
        return impacts[:3]
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        return []


def suggest_restaurants(name: str, city: str) -> list[str]:
    """Return up to 5 restaurant name suggestions from local data."""
    if len(name) < 2:
        return []

    restaurants = _get_unique_restaurants()
    if not restaurants:
        return []

    city_lower = city.lower().strip() if city else ""
    if city_lower:
        filtered = [r for r in restaurants if str(r.get('city', '')).lower().strip() == city_lower]
        if filtered:
            restaurants = filtered

    names = [r['business_name'] for r in restaurants]
    matches = process.extract(name, names, scorer=fuzz.token_sort_ratio, limit=5)

    return [m[0] for m in matches if m[1] >= 30]


def predict(restaurant_name: str, city: str) -> PredictionResult:
    """Predict food safety grade for a Tennessee restaurant."""
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

    restaurant = _find_restaurant(restaurant_name, city)
    if restaurant is None:
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
            error="Restaurant not found in Tennessee database. Try a different spelling or city.",
        )

    business_id = restaurant['business_id']
    stats = _get_restaurant_stats(business_id)
    reviews = _get_restaurant_reviews(business_id)

    if stats.get('rating') is None:
        return PredictionResult(
            restaurant_name=restaurant['business_name'],
            location=restaurant.get('city', city),
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
            error="No review data found for this restaurant.",
        )

    feature_vector, feature_names, rating = _build_feature_vector(stats)

    proba = model.predict_proba(feature_vector)[0]
    pred_class = int(np.argmax(proba))
    confidence = float(proba[pred_class])
    predicted_grade = GRADE_LABELS.get(pred_class, "?")

    shap_features = _compute_shap(feature_vector, feature_names, pred_class)

    divergence_warning = predicted_grade == "C" and rating is not None and rating >= 4.0

    inspection = _find_inspection(restaurant['business_name'], city)
    actual_grade = None
    actual_score = None
    if inspection:
        actual_score = float(inspection['score'])
        actual_grade = _score_to_grade(actual_score)

    return PredictionResult(
        restaurant_name=restaurant['business_name'],
        location=restaurant.get('city', city),
        predicted_grade=predicted_grade,
        grade_color=GRADE_COLORS.get(predicted_grade, "gray"),
        confidence=confidence,
        yelp_rating=rating,
        yelp_review_count=stats.get('review_count'),
        google_rating=None,
        google_review_count=None,
        rating_delta=None,
        top_shap_features=shap_features,
        divergence_warning=divergence_warning,
        sample_reviews=reviews[:3],
        actual_grade=actual_grade,
        actual_score=actual_score,
    )
