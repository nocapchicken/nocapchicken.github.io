"""
Setup script: acquire raw data from NC inspection records, Yelp, and Google Places.

Each step is skipped if its output file already exists unless --force is passed.

Usage:
    python3 setup.py                        # skip steps whose output already exists
    python3 setup.py --force                # re-run all steps
    python3 setup.py --from 01/01/2022      # inspections from 2022 onward
    python3 setup.py --from 01/01/2022 --to 12/31/2024
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from scripts.make_dataset import collect_inspections, collect_yelp_reviews, collect_google_reviews


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="nocapchicken data acquisition pipeline")
    p.add_argument("--force", action="store_true", help="Re-run all steps even if output exists")
    p.add_argument("--from", dest="date_from", default="01/01/2020", metavar="MM/DD/YYYY",
                   help="Inspection date lower bound (default: 01/01/2020)")
    p.add_argument("--to", dest="date_to", default="", metavar="MM/DD/YYYY",
                   help="Inspection date upper bound (default: no limit)")
    return p.parse_args()


def main() -> None:
    """Orchestrate full data acquisition pipeline."""
    args = parse_args()
    load_dotenv()

    yelp_key = os.getenv("RAPIDAPI_KEY")
    google_key = os.getenv("GOOGLE_PLACES_API_KEY")

    if not yelp_key or not google_key:
        print("ERROR: Missing API keys. Copy .env.example to .env and fill in your keys.")
        sys.exit(1)

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Step 1/3: Collecting NC inspection records (from={args.date_from or 'any'} to={args.date_to or 'any'})...")
    collect_inspections(
        output_dir=raw_dir,
        date_from=args.date_from,
        date_to=args.date_to,
        force=args.force,
    )

    print("Step 2/3: Fetching Yelp review data...")
    collect_yelp_reviews(
        api_key=yelp_key,
        inspections_path=raw_dir / "inspections.csv",
        output_dir=raw_dir,
        force=args.force,
    )

    print("Step 3/3: Fetching Google Places review data...")
    collect_google_reviews(
        api_key=google_key,
        inspections_path=raw_dir / "inspections.csv",
        output_dir=raw_dir,
        force=args.force,
    )

    print("\nData collection complete. Files written to data/raw/")
    print("Next step: python3 scripts/build_features.py")


if __name__ == "__main__":
    main()
