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

- [NC DHHS Public Inspection Records](https://public.cdpehs.com/NCENVPBL/INSPECTION_VIOLATION)
- [Yelp Fusion API](https://docs.developer.yelp.com/docs/fusion-intro)
- [Google Places API](https://developers.google.com/maps/documentation/places/web-service/overview)

## Ethics

See the written report for a full ethics statement covering data provenance, consent, and potential harms from misclassification.
