"""
make_dataset.py — Data acquisition for nocapchicken.

Three sources:
  1. NC DHHS public inspection records (scraped from cdpehs.com)
  2. Yelp Fusion API (business metadata + up to 3 reviews)
  3. Google Places API (rating, review count, up to 5 reviews)

The linking step uses fuzzy name + address matching (rapidfuzz) to join
inspection records to Yelp/Google listings — this is the novel data
contribution of the project.
"""

from __future__ import annotations

import time
import logging
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process
from tqdm import tqdm

logger = logging.getLogger(__name__)

# NC county codes 1–100 (DHHS uses sequential integers)
NC_COUNTY_CODES = list(range(1, 101))
INSPECTION_BASE_URL = "https://public.cdpehs.com/NCENVPBL"


# ---------------------------------------------------------------------------
# NC Inspection Records
# ---------------------------------------------------------------------------

def collect_inspections(output_dir: Path, county_codes: list[int] = NC_COUNTY_CODES) -> pd.DataFrame:
    """
    Scrape NC DHHS public inspection records for all specified counties.

    Args:
        output_dir: Directory to write inspections.csv
        county_codes: List of NC county integer codes to collect

    Returns:
        DataFrame of raw inspection records
    """
    records = []

    for county_code in tqdm(county_codes, desc="Counties"):
        try:
            rows = _scrape_county(county_code)
            records.extend(rows)
            time.sleep(0.5)  # polite crawl delay
        except Exception as exc:
            logger.warning("County %d failed: %s", county_code, exc)

    df = pd.DataFrame(records)
    out_path = output_dir / "inspections.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d inspection records to %s", len(df), out_path)
    return df


def _scrape_county(county_code: int) -> list[dict]:
    """Scrape one county's inspection listing page."""
    url = f"{INSPECTION_BASE_URL}/ShowESTABLISHMENTTablePage.aspx?ESTTST_CTY={county_code}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    rows = []

    # Table structure varies slightly by county — adapt selector as needed
    for row in soup.select("table tr")[1:]:  # skip header
        cols = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cols) >= 5:
            rows.append({
                "county_code": county_code,
                "establishment_name": cols[0],
                "address": cols[1],
                "city": cols[2],
                "score": cols[3],
                "grade": cols[4],
                "inspection_date": cols[5] if len(cols) > 5 else None,
            })

    return rows


# ---------------------------------------------------------------------------
# Yelp Fusion API
# ---------------------------------------------------------------------------

def collect_yelp_reviews(
    api_key: str,
    inspections_path: Path,
    output_dir: Path,
    max_per_business: int = 3,
) -> pd.DataFrame:
    """
    Match inspection records to Yelp businesses and fetch review data.

    Args:
        api_key: Yelp Fusion API key
        inspections_path: Path to inspections.csv
        output_dir: Directory to write yelp_reviews.csv
        max_per_business: Max reviews per business (Yelp free tier cap)

    Returns:
        DataFrame with Yelp business metadata and reviews
    """
    inspections = pd.read_csv(inspections_path)
    headers = {"Authorization": f"Bearer {api_key}"}
    results = []

    for _, row in tqdm(inspections.iterrows(), total=len(inspections), desc="Yelp"):
        business = _yelp_search(row["establishment_name"], row["address"], row["city"], headers)
        if business is None:
            continue

        reviews = _yelp_reviews(business["id"], headers, limit=max_per_business)
        results.append({
            "inspection_id": row.name,
            "establishment_name": row["establishment_name"],
            "yelp_id": business["id"],
            "yelp_name": business["name"],
            "yelp_rating": business.get("rating"),
            "yelp_review_count": business.get("review_count"),
            "yelp_price": business.get("price"),
            "yelp_reviews": " ||| ".join(r["text"] for r in reviews),
            "match_score": _fuzzy_match_score(row["establishment_name"], business["name"]),
        })
        time.sleep(0.25)

    df = pd.DataFrame(results)
    out_path = output_dir / "yelp_reviews.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d Yelp records to %s", len(df), out_path)
    return df


def _yelp_search(name: str, address: str, city: str, headers: dict) -> dict | None:
    """Search Yelp for a single business by name and location."""
    params = {"term": name, "location": f"{address}, {city}, NC", "limit": 1}
    resp = requests.get(
        "https://api.yelp.com/v3/businesses/search",
        headers=headers,
        params=params,
        timeout=10,
    )
    if resp.status_code != 200:
        return None
    businesses = resp.json().get("businesses", [])
    return businesses[0] if businesses else None


def _yelp_reviews(business_id: str, headers: dict, limit: int = 3) -> list[dict]:
    """Fetch reviews for a Yelp business."""
    resp = requests.get(
        f"https://api.yelp.com/v3/businesses/{business_id}/reviews",
        headers=headers,
        params={"limit": limit},
        timeout=10,
    )
    if resp.status_code != 200:
        return []
    return resp.json().get("reviews", [])


# ---------------------------------------------------------------------------
# Google Places API
# ---------------------------------------------------------------------------

def collect_google_reviews(
    api_key: str,
    inspections_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Match inspection records to Google Places and fetch review data.

    Args:
        api_key: Google Places API key
        inspections_path: Path to inspections.csv
        output_dir: Directory to write google_reviews.csv

    Returns:
        DataFrame with Google Places metadata and reviews
    """
    import googlemaps

    gmaps = googlemaps.Client(key=api_key)
    inspections = pd.read_csv(inspections_path)
    results = []

    for _, row in tqdm(inspections.iterrows(), total=len(inspections), desc="Google"):
        query = f"{row['establishment_name']} {row['address']} {row['city']} NC"
        try:
            place = _google_search(gmaps, query)
            if place is None:
                continue

            details = gmaps.place(place["place_id"], fields=["name", "rating", "user_ratings_total", "reviews"])
            detail_result = details.get("result", {})

            reviews_text = " ||| ".join(
                r["text"] for r in detail_result.get("reviews", [])
            )
            results.append({
                "inspection_id": row.name,
                "establishment_name": row["establishment_name"],
                "google_place_id": place["place_id"],
                "google_name": detail_result.get("name"),
                "google_rating": detail_result.get("rating"),
                "google_review_count": detail_result.get("user_ratings_total"),
                "google_reviews": reviews_text,
                "match_score": _fuzzy_match_score(row["establishment_name"], detail_result.get("name", "")),
            })
        except Exception as exc:
            logger.warning("Google lookup failed for '%s': %s", row["establishment_name"], exc)

        time.sleep(0.1)

    df = pd.DataFrame(results)
    out_path = output_dir / "google_reviews.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d Google records to %s", len(df), out_path)
    return df


def _google_search(gmaps, query: str) -> dict | None:
    """Search Google Places for a single establishment."""
    result = gmaps.find_place(query, input_type="textquery", fields=["place_id", "name"])
    candidates = result.get("candidates", [])
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _fuzzy_match_score(name_a: str, name_b: str) -> float:
    """Return token sort ratio between two business names (0–100)."""
    return fuzz.token_sort_ratio(name_a or "", name_b or "")
