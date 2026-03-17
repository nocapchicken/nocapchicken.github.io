# AI-assisted (Claude Code, claude.ai) — https://claude.ai
"""Export pre-computed inference data for browser-side ONNX inference.

Writes docs/static/data/restaurants.json — a lookup keyed by lowercase
google_name. Each entry contains pre-computed feature values so the browser
never needs raw review text; it feeds the features directly into the ONNX model.

Run after training:
    python scripts/export_browser_data.py
"""

from __future__ import annotations

import csv
import glob
import gzip
import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path

from scripts.feature_constants import NEGATIVE_PATTERN, SAFETY_PATTERN

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "docs" / "static" / "data" / "restaurants.json.gz"
SAMPLE_REVIEW_CHARS = 120  # truncate each sample review to keep JSON small


def _load_locations_and_grades() -> tuple[dict[str, str], dict[str, dict]]:
    """Build state_id → location and state_id → {grade, score} lookups."""
    locations: dict[str, str] = {}
    grades: dict[str, dict] = {}

    for path in sorted(glob.glob(str(RAW_DIR / "inspections_*.csv"))):
        with open(path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                sid = row.get("state_id", "").strip()
                if not sid:
                    continue

                city = (row.get("city") or "").strip().title()
                if city and sid not in locations:
                    locations[sid] = f"{city}, NC"

                grade = (row.get("grade") or "").strip()
                date = (row.get("inspection_date") or "").strip()
                if grade and grade.lower() != "nan":
                    prev_date = grades.get(sid, {}).get("date", "")
                    try:
                        is_newer = datetime.strptime(date, "%m/%d/%Y") >= datetime.strptime(prev_date, "%m/%d/%Y")
                    except ValueError:
                        is_newer = date >= prev_date
                    if is_newer:
                        try:
                            score = float(row.get("score") or 0)
                        except ValueError:
                            score = 0.0
                        grades[sid] = {"grade": grade, "score": score, "date": date}

    return locations, grades


def _compute_text_features(reviews_text: str) -> dict:
    """Compute the same text features as app/inference.py _build_feature_vector."""
    words = reviews_text.split() if reviews_text else []
    word_count = len(words)
    clean_words = re.sub(r"[^\w\s]", "", reviews_text).split() if reviews_text else []
    avg_word_len = (
        sum(len(w) for w in clean_words) / len(clean_words) if clean_words else 0.0
    )
    safety_count = len(re.findall(SAFETY_PATTERN, reviews_text.lower()))
    negative_count = len(re.findall(NEGATIVE_PATTERN, reviews_text.lower()))
    return {
        "review_word_count": float(word_count),
        "review_avg_word_len": round(avg_word_len, 4),
        "safety_keyword_count": float(safety_count),
        "negative_phrase_count": float(negative_count),
    }


def export_restaurants() -> int:
    """Build and write the restaurant lookup JSON. Returns number of entries."""
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    locations, grades = _load_locations_and_grades()
    logger.info("Loaded %d locations, %d grade entries", len(locations), len(grades))

    lookup: dict[str, dict] = {}

    with open(RAW_DIR / "google_reviews.csv", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            google_name = (row.get("google_name") or "").strip()
            rating_str = row.get("google_rating", "")
            reviews_raw = row.get("google_reviews", "")

            if not google_name or not rating_str or not reviews_raw:
                continue

            try:
                rating = float(rating_str)
            except ValueError:
                continue

            review_count_str = row.get("google_review_count") or "0"
            try:
                review_count = int(float(review_count_str))
            except ValueError:
                review_count = 0

            reviews_text = " ".join(r.strip() for r in reviews_raw.split("|||") if r.strip())
            sample_reviews = [
                r.strip()[:SAMPLE_REVIEW_CHARS]
                for r in reviews_raw.split("|||")
                if r.strip()
            ][:3]

            text_feats = _compute_text_features(reviews_text)

            sid = (row.get("state_id") or "").strip()
            location = locations.get(sid, "")
            grade_info = grades.get(sid, {})

            key = google_name.lower()
            existing = lookup.get(key)
            if existing and existing["review_count"] >= review_count:
                continue

            lookup[key] = {
                "name": google_name,
                "features": [
                    round(rating, 4),
                    round(math.log1p(review_count), 6),
                    text_feats["review_word_count"],
                    text_feats["review_avg_word_len"],
                    text_feats["safety_keyword_count"],
                    text_feats["negative_phrase_count"],
                ],
                "rating": rating,
                "review_count": review_count,
                "location": location,
                "actual_grade": grade_info.get("grade"),
                "actual_score": grade_info.get("score"),
                "actual_date": grade_info.get("date"),
                "sample_reviews": sample_reviews,
            }

    raw_json = json.dumps(lookup, separators=(",", ":"))
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as gz:
        gz.write(raw_json)

    size_kb = OUT_PATH.stat().st_size / 1024
    logger.info("Wrote %d entries → %s (%.1f KB gzipped)", len(lookup), OUT_PATH, size_kb)
    return len(lookup)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    export_restaurants()
