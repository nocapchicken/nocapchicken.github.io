# Google Maps Agentic Scraper (Archived)

Browser-automation scraper that used Gemini + browser-use to extract
restaurant inspection-related signals from Google Maps listings.

## Why archived

A teammate obtained structured data directly from the Google Places API,
which gave us cleaner, more reliable fields (ratings, review counts,
price level) without the fragility of browser automation. This scraper
is preserved here to document the exploration effort.

## What's here

| File | Purpose |
|------|---------|
| `agent_scraper.py` | Gemini-powered browser agent that navigates Google Maps |
| `requirements.txt` | Python dependencies (`browser-use`, `langchain-google-genai`) |
| `restaurants.csv` | Target restaurant list used as input |

## Not used by the main pipeline

Nothing in `extras/` is imported or called by `scripts/`, `app/`, or any
other production code.
