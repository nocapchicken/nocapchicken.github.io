# AI-assisted (Claude Code, claude.ai) — https://claude.ai
"""
setup.py — Data acquisition pipeline for nocapchicken.

Usage:
    python3 setup.py                        # collect 2020-now, skip existing years
    python3 setup.py --years 2023 2024      # only collect specific years
    python3 setup.py --force                # re-run all steps including existing years
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from scripts.make_dataset import collect_inspections, collect_google_reviews


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the data acquisition pipeline."""
    current_year = datetime.date.today().year
    p = argparse.ArgumentParser(description="nocapchicken data acquisition pipeline")
    p.add_argument("--force", action="store_true", help="Re-run all steps even if output exists")
    p.add_argument("--years", nargs="+", type=int, metavar="YYYY",
                   help=f"Years to collect (default: 2020–{current_year})")
    return p.parse_args()


def main() -> None:
    """Run the full data acquisition pipeline: inspections → Google reviews."""
    args = parse_args()
    load_dotenv()

    google_key = os.getenv("GOOGLE_PLACES_API_KEY")

    if not google_key:
        print("ERROR: Missing GOOGLE_PLACES_API_KEY. Copy .env.example to .env and fill in your key.")
        sys.exit(1)

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    years_label = ", ".join(str(y) for y in args.years) if args.years else "2020-now"
    print(f"Step 1/2: Collecting NC inspection records ({years_label})...")
    collect_inspections(output_dir=raw_dir, years=args.years, force=args.force)

    print("Step 2/2: Fetching Google Places review data...")
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
