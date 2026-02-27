# nocapchicken

**Do crowdsourced reviews actually reflect food safety?**

This project investigates the relationship between NC restaurant health inspection scores and public review platform data (Yelp + Google Places). We collect, link, and model inspection records against review sentiment to surface restaurants where public perception diverges from ground-truth food safety outcomes.

## Project Structure

```
├── README.md               <- You are here
├── requirements.txt        <- Python dependencies
├── setup.py                <- Data acquisition and environment setup
├── main.py                 <- Entry point: run inference / launch app
├── scripts/
│   ├── make_dataset.py     <- Scrape NC inspection records + fetch review data
│   ├── build_features.py   <- Feature engineering pipeline
│   └── model.py            <- Train models and generate predictions
├── models/                 <- Serialized trained models
├── data/
│   ├── raw/                <- Raw inspection + review data (not committed)
│   ├── processed/          <- Cleaned, merged, feature-engineered data
│   └── outputs/            <- Predictions, evaluation results, plots
├── notebooks/              <- Exploratory notebooks (not used for grading)
└── .github/
    └── PULL_REQUEST_TEMPLATE/
        └── pull_request_template.md
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect data (NC inspections + Yelp + Google Places)
python setup.py

# 3. Train models
python scripts/model.py

# 4. Launch the app
python main.py
```

## Models

Three approaches are implemented and evaluated:

| Model | Location |
|---|---|
| Naive baseline (majority class / mean predictor) | `scripts/model.py` |
| Classical ML (Random Forest + SHAP explainability) | `scripts/model.py` |
| Deep learning (DistilBERT fine-tuned on review text) | `scripts/model.py` |

## Live App

Deployed at: _link TBD_

## Data Sources

| Source | Description | License |
|---|---|---|
| [NC DHHS Environmental Health](https://public.cdpehs.com/NCENVPBL/ESTABLISHMENT/ShowESTABLISHMENTTablePage.aspx) | Restaurant inspection scores, grades, and violation records for all 100 NC counties. Collected by NC DHHS and published as public record under NC Public Records Law (G.S. § 132-1). Portal software © Custom Data Processing, Inc. | Public government record |
| [Yelp Business API (RapidAPI)](https://rapidapi.com/oneapi/api/yelp-business-api) | Business ratings, review counts, and review text matched to inspection records via fuzzy name+address linking. | Yelp Terms of Service |
| [Google Places API](https://developers.google.com/maps/documentation/places/web-service/overview) | Business ratings, review counts, and review text as a second independent platform signal. | Google Terms of Service |

## Ethics

See the written report for a full ethics statement covering data provenance, consent, and potential harms from misclassification.
