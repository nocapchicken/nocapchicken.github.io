# AI-assisted (Claude Code, claude.ai) — https://claude.ai
"""
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

import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from tqdm import tqdm

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

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


def collect_inspections(
    output_dir: Path,
    county_codes: list[int] = NC_COUNTY_CODES,
    years: list[int] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Completed years are skipped unless force=True. Current year re-fetches if stale."""
    import datetime
    if years is None:
        years = list(range(2020, datetime.date.today().year + 1))

    today = datetime.date.today()

    for year in years:
        year_path = output_dir / f"inspections_{year}.csv"

        if _csv_has_rows(year_path) and not force:
            if year != today.year:
                logger.info("inspections_%d.csv exists — skipping.", year)
                continue
            last_modified = datetime.date.fromtimestamp(year_path.stat().st_mtime)
            if last_modified >= today:
                logger.info("inspections_%d.csv already fetched today — skipping.", year)
                continue
            logger.info("inspections_%d.csv is stale (last fetched %s) — re-fetching.", year, last_modified)

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
    year_files = sorted(output_dir.glob("inspections_*.csv"))
    if not year_files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in year_files], ignore_index=True)


def _scrape_county_bulk(county_code: int, date_from: str = "", date_to: str = "") -> list[dict]:
    """Fetch all inspection records for one county via the site's CSV export."""
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


def _csv_has_rows(path: Path) -> bool:
    if not path.exists():
        return False
    with open(path) as fh:
        non_blank = 0
        for line in fh:
            if line.strip():
                non_blank += 1
                if non_blank >= 2:
                    return True
    return False


def _build_postback_payload(soup: BeautifulSoup, event_target: str) -> dict:
    """Collect ASP.NET hidden fields + pagination event target into a POST dict."""
    payload = {"__EVENTTARGET": event_target, "__EVENTARGUMENT": ""}

    for hidden in soup.find_all("input", type="hidden"):
        name = hidden.get("name", "")
        value = hidden.get("value", "")
        if name:
            payload[name] = value

    return payload


def collect_yelp_reviews(
    api_key: str,
    inspections_path: Path,
    output_dir: Path,
    max_per_business: int = 3,
    force: bool = False,
) -> pd.DataFrame:
    """Fuzzy-match inspections to Yelp businesses and fetch reviews."""
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
            "state_id": row["state_id"],
            "establishment_name": row["establishment_name"],
            "yelp_id": business["id"],
            "yelp_name": business["name"],
            "yelp_rating": business.get("rating"),
            "yelp_review_count": business.get("review_count"),
            "yelp_price": business.get("price"),
            "yelp_reviews": " ||| ".join(r["text"] for r in reviews),
            "match_score": fuzz.token_sort_ratio(row["establishment_name"], business["name"]),
        })
        time.sleep(0.25)

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d Yelp records to %s", len(df), out_path)
    return df


def _yelp_search(name: str, address: str, city: str, headers: dict) -> dict | None:
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


GOOGLE_SAVE_INTERVAL = 50  # flush to disk every N new records


def collect_google_reviews(
    api_key: str,
    inspections_path: Path,
    output_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    """Fuzzy-match inspections to Google Places and fetch reviews.

    Resumes from where a previous run left off — already-fetched state_ids
    are skipped. Appends to google_reviews.csv every GOOGLE_SAVE_INTERVAL
    records so progress is not lost on interruption or quota exhaustion.
    Use force=True to discard existing results and start fresh.
    """
    import googlemaps

    out_path = output_dir / "google_reviews.csv"

    if force and out_path.exists():
        out_path.unlink()
        logger.info("force=True — removed existing google_reviews.csv")

    # Load already-fetched state_ids so we can skip them
    if _csv_has_rows(out_path):
        existing = pd.read_csv(out_path)
        fetched_ids = set(existing["state_id"].astype(str))
        logger.info("Resuming — %d state_ids already fetched, skipping them.", len(fetched_ids))
    else:
        existing = pd.DataFrame()
        fetched_ids = set()

    inspections = pd.read_csv(inspections_path)
    remaining = inspections[~inspections["state_id"].astype(str).isin(fetched_ids)]
    logger.info("%d restaurants remaining to fetch.", len(remaining))

    if remaining.empty:
        logger.info("All restaurants already fetched.")
        return existing

    gmaps = googlemaps.Client(key=api_key)
    new_results = []

    for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Google"):
        query = f"{row['establishment_name']} {row['street_address']} {row['city']} NC"
        try:
            candidates = gmaps.find_place(
                query, input_type="textquery", fields=["place_id", "name"]
            ).get("candidates", [])
            if not candidates:
                continue
            place = candidates[0]

            details = gmaps.place(
                place["place_id"],
                fields=["name", "rating", "user_ratings_total", "reviews"],
            )
            detail_result = details.get("result", {})

            reviews_text = " ||| ".join(r["text"] for r in detail_result.get("reviews", []))
            new_results.append({
                "state_id": row["state_id"],
                "establishment_name": row["establishment_name"],
                "google_place_id": place["place_id"],
                "google_name": detail_result.get("name"),
                "google_rating": detail_result.get("rating"),
                "google_review_count": detail_result.get("user_ratings_total"),
                "google_reviews": reviews_text,
                "match_score": fuzz.token_sort_ratio(
                    row["establishment_name"], detail_result.get("name", "")
                ),
            })
        except Exception as exc:
            logger.warning("Google lookup failed for '%s': %s", row["establishment_name"], exc)

        time.sleep(0.1)

        if len(new_results) % GOOGLE_SAVE_INTERVAL == 0:
            _append_google_results(new_results, out_path)
            new_results = []

    # Final flush
    if new_results:
        _append_google_results(new_results, out_path)

    result = pd.read_csv(out_path) if _csv_has_rows(out_path) else pd.DataFrame()
    logger.info("google_reviews.csv now has %d total records.", len(result))
    return result


def _append_google_results(records: list[dict], out_path: Path) -> None:
    """Append records to google_reviews.csv, writing header only on first write."""
    df = pd.DataFrame(records)
    df.to_csv(out_path, mode="a", header=not out_path.exists() or out_path.stat().st_size == 0, index=False)
    logger.info("Flushed %d records to %s", len(df), out_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Collect NC inspection + review data")
    parser.add_argument("--google-key", metavar="KEY", help="Google Maps API key")
    parser.add_argument("--yelp-key", metavar="KEY", help="RapidAPI key for Yelp Business API")
    parser.add_argument("--inspections-only", action="store_true", help="Only scrape inspections, skip reviews")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if output files already exist")
    args = parser.parse_args()

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    inspections_path = raw_dir / "inspections.csv"

    collect_inspections(raw_dir, force=args.force)

    if args.inspections_only:
        return

    if args.google_key:
        collect_google_reviews(args.google_key, inspections_path, raw_dir, force=args.force)
    else:
        logger.info("--google-key not provided — skipping Google review collection")

    if args.yelp_key:
        collect_yelp_reviews(args.yelp_key, inspections_path, raw_dir, force=args.force)
    else:
        logger.info("--yelp-key not provided — skipping Yelp review collection")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
