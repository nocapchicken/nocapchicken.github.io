"""
make_dataset.py — Data acquisition for nocapchicken.

Three sources:
  1. NC DHHS public inspection records (scraped from cdpehs.com)
  2. Yelp via RapidAPI proxy (business metadata + reviews)
  3. Google Places API (rating, review count, reviews)

The linking step uses fuzzy name + address matching (rapidfuzz) to join
inspection records to Yelp/Google listings — this is the novel data
contribution of the project.

NC DHHS page structure (confirmed via inspection):
  URL: /NCENVPBL/ESTABLISHMENT/ShowESTABLISHMENTTablePage.aspx?ESTTST_CTY={code}
  Data rows: <tr> with exactly 10 <td> cells where cells[0].text == "Violation Details"
  Columns:   [violation_link, date, name, address_full, state_id,
              est_type, score, grade, inspector_id, report_link]
  Pagination: ASP.NET PostBack via __VIEWSTATE; default 10 records/page.
"""

from __future__ import annotations

import re
import time
import logging
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from tqdm import tqdm

logger = logging.getLogger(__name__)

# NC county codes 1–100 (DHHS uses sequential integers)
NC_COUNTY_CODES = list(range(1, 101))

BASE_URL = "https://public.cdpehs.com/NCENVPBL/ESTABLISHMENT/ShowESTABLISHMENTTablePage.aspx"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Only collect restaurant types to keep dataset focused
RESTAURANT_TYPE_CODES = {"1", "2", "3", "4", "14", "15"}  # restaurants + food stands + mobile


# ---------------------------------------------------------------------------
# NC Inspection Records
# ---------------------------------------------------------------------------

def collect_inspections(output_dir: Path, county_codes: list[int] = NC_COUNTY_CODES) -> pd.DataFrame:
    """
    Scrape NC DHHS public inspection records for all specified counties.

    Paginates through all pages for each county using ASP.NET PostBack.
    Filters to restaurant/food establishment types only.

    Args:
        output_dir: Directory to write inspections.csv
        county_codes: List of NC county integer codes to collect

    Returns:
        DataFrame of raw inspection records
    """
    records = []

    for county_code in tqdm(county_codes, desc="Counties"):
        try:
            rows = _scrape_county_all_pages(county_code)
            records.extend(rows)
            time.sleep(0.75)  # polite crawl delay between counties
        except Exception as exc:
            logger.warning("County %d failed: %s", county_code, exc)

    df = pd.DataFrame(records)
    out_path = output_dir / "inspections.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d inspection records to %s", len(df), out_path)
    return df


def _scrape_county_all_pages(county_code: int) -> list[dict]:
    """Scrape all pages for one county via ASP.NET PostBack pagination."""
    session = requests.Session()
    session.headers.update(HEADERS)

    url = f"{BASE_URL}?ESTTST_CTY={county_code}"
    resp = session.get(url, timeout=15)
    resp.raise_for_status()

    all_rows = []
    page = 1

    while True:
        soup = BeautifulSoup(resp.text, "lxml")
        rows = _parse_data_rows(soup, county_code)
        all_rows.extend(rows)

        # Check if a "Next" page link exists
        next_payload = _get_next_page_payload(soup)
        if next_payload is None:
            break

        page += 1
        resp = session.post(url, data=next_payload, timeout=15)
        resp.raise_for_status()
        time.sleep(0.3)

    logger.debug("County %d: %d records across %d page(s)", county_code, len(all_rows), page)
    return all_rows


def _parse_data_rows(soup: BeautifulSoup, county_code: int) -> list[dict]:
    """
    Extract data rows from a parsed inspection listing page.

    Each data row is a <tr> with 10 <td> cells where the first cell
    contains the text "Violation Details".

    Column order (0-indexed):
      0  violation link  (contains ESTABLISHMENT and INSPECTION ids in href)
      1  inspection_date
      2  premises_name
      3  address_full    (street + city + state + zip concatenated)
      4  state_id
      5  establishment_type
      6  final_score
      7  grade
      8  inspector_id
      9  report link
    """
    rows = []

    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) != 10:
            continue

        texts = [td.get_text(separator=" ", strip=True) for td in tds]
        if texts[0] != "Violation Details":
            continue

        # Filter to food/restaurant types only
        est_type_raw = texts[5]
        est_type_code = est_type_raw.split(" - ")[0].strip()
        if est_type_code not in RESTAURANT_TYPE_CODES:
            continue

        # Extract ESTABLISHMENT and INSPECTION ids from the violation link href
        violation_href = tds[0].find("a", href=True)
        establishment_id, inspection_id = _parse_violation_href(
            violation_href["href"] if violation_href else ""
        )

        # Parse the concatenated address field: "123 MAIN STCITY, NC 27601"
        address_raw = texts[3].replace("\xa0", " ")
        street, city, zipcode = _split_address(address_raw)

        score_raw = texts[6].strip()
        try:
            score = float(score_raw)
        except ValueError:
            score = None

        rows.append({
            "county_code": county_code,
            "establishment_id": establishment_id,
            "inspection_id": inspection_id,
            "inspection_date": texts[1],
            "establishment_name": texts[2],
            "street_address": street,
            "city": city,
            "zip": zipcode,
            "state_id": texts[4],
            "establishment_type": est_type_raw,
            "score": score,
            "grade": texts[7],
            "inspector_id": texts[8],
        })

    return rows


def _parse_violation_href(href: str) -> tuple[str, str]:
    """Extract ESTABLISHMENT and INSPECTION ids from a violation detail URL."""
    est = re.search(r"ESTABLISHMENT=(\d+)", href)
    ins = re.search(r"INSPECTION=(\d+)", href)
    return (est.group(1) if est else ""), (ins.group(1) if ins else "")


# Longest suffix forms first so STREET matches before ST, BOULEVARD before BLVD, etc.
_STREET_SUFFIX_RE = re.compile(
    r"BOULEVARD|STREET|AVENUE|PARKWAY|HIGHWAY|CIRCLE|COURT|PLACE|TRAIL|DRIVE|ROAD"
    r"|BLVD|PKWY|HWY|CIR|PL|TRL|AVE|LANE|LN|WAY|RD|DR|CT|ST"
    r"(?:\s+(?:UNIT|STE|SUITE|APT|BLDG|#)\s*[\w-]+)?"
)


def _split_address(address_full: str) -> tuple[str, str, str]:
    """
    Split the concatenated address string into street, city, and zip.

    The DHHS site omits the space between street and city, e.g.:
      "101 N SCOTSWOOD BLVDHILLSBOROUGH, NC 27278"
      "313 E MAIN STCARRBORO, NC 27510"
      "200 W FRANKLIN STREET UNIT 130CHAPEL HILL, NC 27516"

    Strategy:
      1. Split on ", NC " to isolate (street+city) from zip.
      2. Find the last street-suffix token in the combined string
         (no word boundary needed — city follows the suffix directly).
      3. Everything after the suffix end is the city.
    """
    parts = re.split(r",\s*NC\s+", address_full, maxsplit=1)
    if len(parts) != 2:
        return address_full, "", ""

    street_city, zipcode = parts[0].strip(), parts[1].strip()

    matches = list(_STREET_SUFFIX_RE.finditer(street_city))
    if matches:
        last = matches[-1]
        return street_city[: last.end()].strip(), street_city[last.end() :].strip(), zipcode

    return street_city, "", zipcode


def _get_next_page_payload(soup: BeautifulSoup) -> dict | None:
    """
    Build the POST payload to navigate to the next page.

    ASP.NET WebForms pagination uses __doPostBack() with a target
    control ID. Returns None if no next-page link exists.
    """
    # Find a pager link whose text is ">" or contains "Next"
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        if text in (">", "Next", "»") and "__doPostBack" in href:
            # Extract the event target from javascript:__doPostBack('target','')
            m = re.search(r"__doPostBack\('([^']+)'", href)
            if not m:
                continue
            event_target = m.group(1)
            return _build_postback_payload(soup, event_target)

    return None


def _build_postback_payload(soup: BeautifulSoup, event_target: str) -> dict:
    """Collect ASP.NET hidden fields + pagination event target into a POST dict."""
    payload = {"__EVENTTARGET": event_target, "__EVENTARGUMENT": ""}

    for hidden in soup.find_all("input", type="hidden"):
        name = hidden.get("name", "")
        value = hidden.get("value", "")
        if name:
            payload[name] = value

    return payload


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
    Match inspection records to Yelp businesses and fetch review data
    via the RapidAPI Yelp Business API proxy.

    Args:
        api_key: RapidAPI key (RAPIDAPI_KEY in .env)
        inspections_path: Path to inspections.csv
        output_dir: Directory to write yelp_reviews.csv
        max_per_business: Max reviews to store per business

    Returns:
        DataFrame with Yelp business metadata and reviews
    """
    inspections = pd.read_csv(inspections_path)
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "yelp-business-api.p.rapidapi.com",
    }
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
    """Search Yelp for a single business by name and location (RapidAPI)."""
    params = {"term": name, "location": f"{address}, {city}, NC", "limit": "1"}
    resp = requests.get(
        "https://yelp-business-api.p.rapidapi.com/search",
        headers=headers,
        params=params,
        timeout=10,
    )
    if resp.status_code != 200:
        return None
    businesses = resp.json().get("businesses", [])
    return businesses[0] if businesses else None


def _yelp_reviews(business_id: str, headers: dict, limit: int = 3) -> list[dict]:
    """Fetch reviews for a Yelp business (RapidAPI)."""
    resp = requests.get(
        "https://yelp-business-api.p.rapidapi.com/reviews",
        headers=headers,
        params={"business_id": business_id},
        timeout=10,
    )
    if resp.status_code != 200:
        return []
    reviews = resp.json().get("reviews", [])
    return reviews[:limit]


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
