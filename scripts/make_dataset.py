"""
make_dataset.py — Data acquisition for nocapchicken.

Data sources:
  1. NC DHHS public inspection records
       NC Department of Health and Human Services, Environmental Health Section.
       Public record under NC Public Records Law (G.S. § 132-1).
       Accessed via the CDP public inspection portal:
       https://public.cdpehs.com/NCENVPBL/ESTABLISHMENT/ShowESTABLISHMENTTablePage.aspx
       Portal software © Custom Data Processing, Inc. — data is public government record.

  2. Yelp business data via RapidAPI Yelp Business API proxy
       https://rapidapi.com/oneapi/api/yelp-business-api

  3. Google Places API
       https://developers.google.com/maps/documentation/places/web-service/overview

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

def collect_inspections(
    output_dir: Path,
    county_codes: list[int] = NC_COUNTY_CODES,
    years: list[int] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Scrape NC DHHS public inspection records, one file per year.

    Each year is saved as inspections_{year}.csv. Years whose file already
    exists are skipped unless force=True. The current year is always
    re-fetched (data is still accumulating).

    After collection, all year files are merged into inspections.csv.

    Args:
        output_dir: Directory to write per-year files and inspections.csv
        county_codes: List of NC county integer codes to collect
        years: Years to collect. Defaults to 2020 through current year.
        force: Re-scrape all years, including ones already on disk.

    Returns:
        Merged DataFrame of all collected inspection records
    """
    import datetime
    if years is None:
        years = list(range(2020, datetime.date.today().year + 1))

    for year in years:
        year_path = output_dir / f"inspections_{year}.csv"
        if _csv_has_rows(year_path) and not force:
            is_current = year == datetime.date.today().year
            if is_current:
                last_modified = datetime.date.fromtimestamp(year_path.stat().st_mtime)
                if last_modified >= datetime.date.today():
                    logger.info("inspections_%d.csv already fetched today — skipping.", year)
                    continue
                logger.info("inspections_%d.csv is stale (last fetched %s) — re-fetching.", year, last_modified)
            else:
                logger.info("inspections_%d.csv exists — skipping.", year)
                continue

        logger.info("Fetching %d inspection records...", year)
        records = []
        date_from = f"01/01/{year}"
        date_to = f"12/31/{year}"

        for county_code in tqdm(county_codes, desc=f"{year}"):
            try:
                rows = _scrape_county_bulk(county_code, date_from=date_from, date_to=date_to)
                records.extend(rows)
                time.sleep(0.5)
            except Exception as exc:
                logger.warning("County %d / %d failed: %s", county_code, year, exc)

        df_year = pd.DataFrame(records)
        df_year.to_csv(year_path, index=False)
        logger.info("  → %d records written to %s", len(df_year), year_path)

    logger.info("Collection done. Run build_features.py to merge year files.")
    # Return whatever year files exist on disk so callers have something useful
    year_files = sorted(output_dir.glob("inspections_*.csv"))
    if not year_files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in year_files], ignore_index=True)


def _scrape_county_bulk(county_code: int, date_from: str = "", date_to: str = "") -> list[dict]:
    """
    Fetch all inspection records for one county via the CSV export button.

    The site's CSV export returns all matching records regardless of page size,
    avoiding the need to paginate. Address fields arrive pre-split (city, zip
    are separate columns) so no regex parsing is required.

    Args:
        county_code: NC DHHS county integer code
        date_from: Inspection date lower bound (MM/DD/YYYY) — always set per year
        date_to: Inspection date upper bound (MM/DD/YYYY)
    """
    import io

    session = requests.Session()
    session.headers.update(HEADERS)

    url = f"{BASE_URL}?ESTTST_CTY={county_code}"

    # Step 1: GET the page to harvest ASP.NET hidden fields
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # Step 2: POST clicking the CSV export button with date filters applied
    payload = _build_postback_payload(soup, event_target="")
    payload.update({
        "ctl00$PageContent$INSPECTION_DATEFromFilter": date_from,
        "ctl00$PageContent$INSPECTION_DATEToFilter": date_to,
        "ctl00$PageContent$CSVButton1.x": "1",
        "ctl00$PageContent$CSVButton1.y": "1",
    })

    resp = session.post(url, data=payload, timeout=60)
    resp.raise_for_status()

    if "text/plain" not in resp.headers.get("Content-Type", ""):
        logger.warning("County %d: unexpected Content-Type %s — no CSV returned", county_code, resp.headers.get("Content-Type"))
        return []

    # Strip UTF-8 BOM if present and parse
    df = pd.read_csv(io.StringIO(resp.text.lstrip("\ufeff")))
    if df.empty:
        return []

    # Filter to food/restaurant establishment types
    df = df[df["Establishment Type"].str.split(" - ").str[0].isin(RESTAURANT_TYPE_CODES)]

    # Normalise to our internal schema
    records = []
    for _, row in df.iterrows():
        addr1 = str(row.get("Premise Address 1") or "").strip()
        addr2 = row.get("Premise Address 2")
        addr2 = "" if pd.isna(addr2) else str(addr2).strip()
        street = f"{addr1} {addr2}".strip() if addr2 else addr1

        score_raw = row.get("Final Score")
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = None

        records.append({
            "county_code": county_code,
            "establishment_id": "",
            "inspection_id": "",
            "inspection_date": str(row.get("Inspection Date", "")).strip(),
            "establishment_name": str(row.get("Premises Name", "")).strip(),
            "street_address": street,
            "city": str(row.get("Premise City", "")).strip(),
            "zip": str(row.get("Premise ZIP", "")).strip(),
            "state_id": str(row.get("State ID#", "")).strip(),
            "establishment_type": str(row.get("Establishment Type", "")).strip(),
            "score": score,
            "grade": str(row.get("Grade", "")).strip(),
            "inspector_id": str(row.get("Inspector ID", "")).strip(),
        })

    logger.debug("County %d: %d records", county_code, len(records))
    return records


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


def _csv_has_rows(path: Path) -> bool:
    """Return True only if the file exists and contains at least a header + one data row."""
    if not path.exists():
        return False
    with open(path) as f:
        lines = [l for l in f if l.strip()]
    return len(lines) >= 2


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
    force: bool = False,
) -> pd.DataFrame:
    """
    Match inspection records to Yelp businesses and fetch review data
    via the RapidAPI Yelp Business API proxy.

    Args:
        api_key: RapidAPI key (RAPIDAPI_KEY in .env)
        inspections_path: Path to inspections.csv
        output_dir: Directory to write yelp_reviews.csv
        max_per_business: Max reviews to store per business
        force: Re-fetch even if yelp_reviews.csv already exists

    Returns:
        DataFrame with Yelp business metadata and reviews
    """
    out_path = output_dir / "yelp_reviews.csv"
    if _csv_has_rows(out_path) and not force:
        logger.info("yelp_reviews.csv already exists — skipping. Use force=True to re-fetch.")
        return pd.read_csv(out_path)

    inspections = pd.read_csv(inspections_path)
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "yelp-business-api.p.rapidapi.com",
    }
    results = []

    for _, row in tqdm(inspections.iterrows(), total=len(inspections), desc="Yelp"):
        business = _yelp_search(row["establishment_name"], row["street_address"], row["city"], headers)
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
        logger.warning("Yelp search HTTP %d for '%s': %s", resp.status_code, name, resp.text[:200])
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
    force: bool = False,
) -> pd.DataFrame:
    """
    Match inspection records to Google Places and fetch review data.

    Args:
        api_key: Google Places API key
        inspections_path: Path to inspections.csv
        output_dir: Directory to write google_reviews.csv
        force: Re-fetch even if google_reviews.csv already exists

    Returns:
        DataFrame with Google Places metadata and reviews
    """
    out_path = output_dir / "google_reviews.csv"
    if _csv_has_rows(out_path) and not force:
        logger.info("google_reviews.csv already exists — skipping. Use force=True to re-fetch.")
        return pd.read_csv(out_path)

    import googlemaps

    gmaps = googlemaps.Client(key=api_key)
    inspections = pd.read_csv(inspections_path)
    results = []

    for _, row in tqdm(inspections.iterrows(), total=len(inspections), desc="Google"):
        query = f"{row['establishment_name']} {row['street_address']} {row['city']} NC"
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
