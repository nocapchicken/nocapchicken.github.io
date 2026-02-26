"""
Setup script: acquire raw data from NC inspection records, Yelp, and Google Places.
Run once to populate data/raw/ before training.

Usage:
    python setup.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on the path so scripts can be imported
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from scripts.make_dataset import collect_inspections, collect_yelp_reviews, collect_google_reviews


def main() -> None:
    """Orchestrate full data acquisition pipeline."""
    load_dotenv()

    yelp_key = os.getenv("RAPIDAPI_KEY")
    google_key = os.getenv("GOOGLE_PLACES_API_KEY")

    if not yelp_key or not google_key:
        print("ERROR: Missing API keys. Copy .env.example to .env and fill in your keys.")
        sys.exit(1)

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1/3: Collecting NC inspection records...")
    collect_inspections(output_dir=raw_dir)

    print("Step 2/3: Fetching Yelp review data...")
    collect_yelp_reviews(api_key=yelp_key, inspections_path=raw_dir / "inspections.csv", output_dir=raw_dir)

    print("Step 3/3: Fetching Google Places review data...")
    collect_google_reviews(api_key=google_key, inspections_path=raw_dir / "inspections.csv", output_dir=raw_dir)

    print("\nData collection complete. Files written to data/raw/")
    print("Next step: python scripts/build_features.py")


if __name__ == "__main__":
    main()
