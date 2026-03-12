# AI-assisted (Claude Code, claude.ai) — https://claude.ai
"""
Google Maps review scraper using browser-use + Gemini 2.5 Flash.

An AI agent navigates Google Maps, extracts ratings and reviews,
and stores results in SQLite. Much more resilient to DOM changes
than hard-coded selectors since the LLM interprets the page visually.
"""

import argparse
import asyncio
import csv
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel

from browser_use import Agent, BrowserProfile, ChatGoogle

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_CSV = Path(__file__).parent / "restaurants.csv"
OUTPUT_DIR = Path(__file__).parent / "output"
RESTAURANTS_OUT = OUTPUT_DIR / "restaurants.csv"
REVIEWS_OUT = OUTPUT_DIR / "reviews.csv"

DELAY_BETWEEN_RESTAURANTS = (3.0, 5.0)


# ── Pydantic models for structured output ───────────────────────────────────


class Review(BaseModel):
    """A single Google Maps review."""

    stars: Optional[float] = None
    date: Optional[str] = None
    text: Optional[str] = None
    food_score: Optional[float] = None
    service_score: Optional[float] = None
    atmosphere_score: Optional[float] = None


class RestaurantData(BaseModel):
    """Structured scrape result for one restaurant."""

    google_name: Optional[str] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    reviews: list[Review] = []


# ── CSV storage layer ───────────────────────────────────────────────────────


def init_output():
    """Create output directory and CSV files with headers if they don't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not RESTAURANTS_OUT.exists():
        with open(RESTAURANTS_OUT, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "id", "input_name", "input_address", "google_name",
                "rating", "review_count", "scraped_at",
            ])

    if not REVIEWS_OUT.exists():
        with open(REVIEWS_OUT, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "id", "restaurant_id", "stars", "date", "text",
                "food_score", "service_score", "atmosphere_score",
            ])


def _next_restaurant_id():
    """Get the next restaurant ID by counting existing rows."""
    if not RESTAURANTS_OUT.exists():
        return 1
    with open(RESTAURANTS_OUT, newline="", encoding="utf-8") as fh:
        return sum(1 for _ in fh)  # header + data rows → next id


def _next_review_id():
    """Get the next review ID by counting existing rows."""
    if not REVIEWS_OUT.exists():
        return 1
    with open(REVIEWS_OUT, newline="", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def is_scraped(name, address):
    """Return True if this restaurant already has a row in the output CSV."""
    if not RESTAURANTS_OUT.exists():
        return False
    with open(RESTAURANTS_OUT, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["input_name"] == name and row["input_address"] == address:
                return True
    return False


def _ensure_trailing_newline(filepath):
    """Ensure file ends with a newline to prevent row concatenation on append."""
    if not filepath.exists():
        return
    with open(filepath, "rb") as fh:
        fh.seek(0, 2)
        if fh.tell() == 0:
            return
        fh.seek(-1, 2)
        if fh.read(1) != b"\n":
            with open(filepath, "a", encoding="utf-8") as append_fh:
                append_fh.write("\n")


def save_to_csv(name, address, data):
    """Append restaurant and its reviews to the output CSVs."""
    restaurant_id = _next_restaurant_id()

    _ensure_trailing_newline(RESTAURANTS_OUT)
    with open(RESTAURANTS_OUT, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            restaurant_id,
            name,
            address,
            data.google_name,
            data.rating,
            data.review_count,
            datetime.now(timezone.utc).isoformat(),
        ])

    if data.reviews:
        review_id = _next_review_id()
        _ensure_trailing_newline(REVIEWS_OUT)
        with open(REVIEWS_OUT, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            for review in data.reviews:
                writer.writerow([
                    review_id,
                    restaurant_id,
                    review.stars,
                    review.date,
                    review.text,
                    review.food_score,
                    review.service_score,
                    review.atmosphere_score,
                ])
                review_id += 1

    return restaurant_id


# ── CSV loader ──────────────────────────────────────────────────────────────


def load_restaurants():
    """Read restaurant names and addresses from the CSV file."""
    restaurants = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            restaurants.append((row["name"].strip(), row["address"].strip()))
    return restaurants


# ── Agent task builder ──────────────────────────────────────────────────────


def build_task_prompt(name, address):
    """Build the instruction prompt for the browser-use agent.

    Uses a two-phase strategy: vision for navigation/scrolling, then
    JavaScript DOM extraction for review data (star ratings from aria-labels
    are far more reliable than trying to count star icons visually).
    """
    js_extract = _review_extraction_js()
    return f"""Go to Google (https://www.google.com) and search for "{name} {address} Durham NC".

PHASE 1 — NAVIGATE & SCROLL (use vision):

From the knowledge panel / business listing on the results page, extract:
1. The place name as shown by Google (google_name)
2. The overall star rating, e.g. 4.5 (rating)
3. The total number of reviews (review_count)

Then click the reviews link (e.g. "X Google reviews") to open the reviews popup.

Scroll the reviews popup EXACTLY 5 times. After each scroll, wait 1 second.
After the 5th scroll, STOP scrolling immediately — do NOT scroll again under
any circumstances.

IMPORTANT: Do NOT scroll more than 5 times. Over-scrolling triggers Google's
sign-in wall which disrupts extraction. 5 scrolls is enough for 15-20 reviews.

PHASE 2 — EXPAND TRUNCATED REVIEWS:

After scrolling, use the `evaluate` action to run this JavaScript to click all
"More" buttons and expand every truncated review:

document.querySelectorAll('a.MtCSLb[role="button"]').forEach(function(a){{ a.click() }})

Wait 2 seconds after running that script so the DOM updates.

PHASE 3 — EXTRACT REVIEWS VIA JAVASCRIPT (do NOT try to read stars visually):

**Do NOT include sub-rating text (e.g. "Food: 5/5 | Service: 5/5 | Atmosphere: 5/5")
in the review `text` field.** The JavaScript extraction handles these separately as
`food_score`, `service_score`, and `atmosphere_score`.

Now use the `evaluate` action to run this exact JavaScript in the page.
Do NOT modify it:

{js_extract}

The script returns a JSON array of review objects. If the result is empty or
starts with "Error", try this fallback JS instead:

(function(){{try{{var r=[];var seen=new Set();document.querySelectorAll('[role="img"][aria-label*="Rated"]').forEach(function(starsEl){{var container=starsEl.closest('div[class]');if(!container)return;var sm=starsEl.getAttribute('aria-label').match(/([\\d.]+)/);var stars=sm?parseFloat(sm[1]):null;var d=container.querySelector('span[class]');var date=d?d.textContent.trim():null;var t=container.querySelector('div[class]>span')||container.querySelector('[data-expandable-section]');var text=t?t.textContent.trim():null;if(text)text=text.replace(/\\s*…\\s*More\\s*$/,'').replace(/\\s*More\\s*$/,'').trim();var sr={{}};var subEl=container.querySelector('div.zMjRQd');if(subEl){{var st=subEl.textContent;var sms=st.matchAll(/(Food|Service|Atmosphere):\\s*(\\d)\\s*\\/\\s*5/g);for(var sm of sms)sr[sm[1].toLowerCase()]=parseInt(sm[2]);if(text)text=text.replace(st,'').trim();}}var key=(text||'')+'|'+stars;if(seen.has(key))return;seen.add(key);if(stars!==null||text)r.push({{stars:stars,date:date,text:text,food_score:sr.food||null,service_score:sr.service||null,atmosphere_score:sr.atmosphere||null}})}});return JSON.stringify(r)}}catch(e){{return 'Error: '+e.message}}}})()

Parse the JSON array from the evaluate result and use it as the reviews list.

OUTPUT: Call done with structured data matching this schema:
  google_name: str | null
  rating: float | null
  review_count: int | null
  reviews: list of {{stars: float | null, date: str | null, text: str | null, food_score: float | null, service_score: float | null, atmosphere_score: float | null}}

Even if you collected fewer than 20 reviews, return what you have."""


def _review_extraction_js():
    """Return the primary JavaScript snippet for extracting reviews from the DOM.

    Uses div.bwb7ce as the review container (NOT [data-review-id], which is on
    a child image-carousel div). Star ratings come from aria-label attributes on
    div.dHX2k[role="img"] elements — far more reliable than vision-based counting.
    """
    return """(function(){
  try {
    var reviews = [];
    var seen = new Set();
    document.querySelectorAll('div.bwb7ce').forEach(function(el) {
      var starsEl = el.querySelector('div.dHX2k[role="img"]');
      var starsMatch = starsEl ? starsEl.getAttribute('aria-label').match(/([\\d.]+)/) : null;
      var stars = starsMatch ? parseFloat(starsMatch[1]) : null;
      var dateEl = el.querySelector('span.y3Ibjb');
      var date = dateEl ? dateEl.textContent.trim() : null;
      var textEl = el.querySelector('div.OA1nbd');
      var text = textEl ? textEl.textContent.trim() : null;
      if (text) text = text.replace(/\\s*…\\s*More\\s*$/, '').replace(/\\s*More\\s*$/, '').trim();
      var subRatings = {};
      var subEl = el.querySelector('div.zMjRQd');
      if (subEl) {
        var st = subEl.textContent;
        var matches = st.matchAll(/(Food|Service|Atmosphere):\s*(\d)\s*\/\s*5/g);
        for (var sm of matches) subRatings[sm[1].toLowerCase()] = parseInt(sm[2]);
        if (text) text = text.replace(st, '').trim();
      }
      var key = (text || '') + '|' + stars;
      if (seen.has(key)) return;
      seen.add(key);
      if (stars !== null || text) reviews.push({
        stars: stars, date: date, text: text,
        food_score: subRatings.food || null,
        service_score: subRatings.service || null,
        atmosphere_score: subRatings.atmosphere || null
      });
    });
    return JSON.stringify(reviews);
  } catch(e) { return 'Error: ' + e.message; }
})()"""


# ── Main scraper loop ───────────────────────────────────────────────────────


async def scrape_restaurant(name, address, headed=False):
    """Run a browser-use agent to scrape one restaurant from Google Maps."""
    llm = ChatGoogle(model="gemini-2.5-flash")

    browser_profile = BrowserProfile(
        headless=not headed,
        minimum_wait_page_load_time=1.0,
        wait_between_actions=0.5,
    )

    agent = Agent(
        task=build_task_prompt(name, address),
        llm=llm,
        browser_profile=browser_profile,
        use_vision=True,
        max_steps=30,
        max_failures=5,
        output_model_schema=RestaurantData,
        calculate_cost=True,
    )

    history = await agent.run()

    # Prefer the typed structured_output; fall back to manual JSON parsing
    if history.structured_output is not None:
        data = history.structured_output
    else:
        raw_result = history.final_result()
        if not raw_result:
            raise ValueError("Agent returned no result")
        data = RestaurantData.model_validate_json(raw_result)

    original_count = len(data.reviews)
    data.reviews = _filter_garbage_reviews(data.reviews)
    if len(data.reviews) < original_count:
        logger.info(
            '"%s": filtered %d garbage reviews (%d -> %d)',
            name, original_count - len(data.reviews),
            original_count, len(data.reviews),
        )
    _warn_if_stars_missing(data, name)
    return data, history.usage


def _filter_garbage_reviews(reviews):
    """Remove reviews that are UI artifacts or have no meaningful content."""
    garbage_patterns = ["Price per person:", "Service: Meal type:"]
    filtered = []
    for review in reviews:
        if not review.text and review.stars is None:
            continue
        if review.text and any(review.text.strip() == pat for pat in garbage_patterns):
            continue
        if review.text and len(review.text.strip()) < 3 and review.stars is None:
            continue
        filtered.append(review)
    return filtered


def _warn_if_stars_missing(data, restaurant_name):
    """Log a warning if more than half of extracted reviews lack a star rating."""
    if not data.reviews:
        return
    missing = sum(1 for r in data.reviews if r.stars is None)
    ratio = missing / len(data.reviews)
    if ratio > 0.5:
        logger.warning(
            '"%s": %d/%d reviews (%.0f%%) have no star rating — '
            "JS extraction may have failed",
            restaurant_name, missing, len(data.reviews), ratio * 100,
        )


async def run(headed=False, limit=None):
    """Main entry point: iterate restaurants, scrape each, save to DB."""
    init_output()
    restaurants = load_restaurants()
    total = len(restaurants)

    if limit is not None:
        restaurants = restaurants[:limit]
        logger.info("Limiting to first %d of %d restaurants", limit, total)

    logger.info("Loaded %d restaurants from %s", len(restaurants), INPUT_CSV.name)

    scraped = 0
    skipped = 0
    failed = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_cost = 0.0

    for idx, (name, address) in enumerate(restaurants, start=1):
        if is_scraped(name, address):
            logger.info("[%d/%d] Skipping \"%s\" (already scraped)", idx, len(restaurants), name)
            skipped += 1
            continue

        logger.info("[%d/%d] Scraping \"%s\"...", idx, len(restaurants), name)
        try:
            data, usage = await scrape_restaurant(name, address, headed=headed)
            save_to_csv(name, address, data)
            scraped += 1
            logger.info(
                "  -> %s | rating=%s | reviews=%d",
                data.google_name or name,
                data.rating or "N/A",
                len(data.reviews),
            )
            if usage:
                logger.info(
                    "  -> tokens: prompt=%d compl=%d total=%d | cost=$%.4f (%d LLM calls)",
                    usage.total_prompt_tokens,
                    usage.total_completion_tokens,
                    usage.total_tokens,
                    usage.total_cost or 0.0,
                    usage.entry_count,
                )
                total_prompt_tokens += usage.total_prompt_tokens
                total_completion_tokens += usage.total_completion_tokens
                total_tokens += usage.total_tokens
                total_cost += usage.total_cost or 0.0
        except Exception as exc:
            failed += 1
            logger.error("  -> FAILED: %s", exc)

        delay = random.uniform(*DELAY_BETWEEN_RESTAURANTS)
        await asyncio.sleep(delay)

    logger.info(
        "Done. scraped=%d skipped=%d failed=%d total=%d",
        scraped, skipped, failed, len(restaurants),
    )
    if total_tokens > 0:
        logger.info(
            "Token totals: prompt=%d compl=%d total=%d | cost=$%.4f",
            total_prompt_tokens, total_completion_tokens, total_tokens, total_cost,
        )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Scrape Google Maps reviews")
    parser.add_argument(
        "--headed", action="store_true",
        help="Run browser in headed mode (visible window) for debugging",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only scrape the first N restaurants",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(headed=args.headed, limit=args.limit))
