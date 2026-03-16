# nocapchicken

[![Live App](https://img.shields.io/badge/app-live%20on%20Render-blue)](https://nocapchicken-github-io.onrender.com)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-green)](https://nocapchicken.github.io)
[![Python 3.11](https://img.shields.io/badge/python-3.11-yellow)](https://www.python.org/)

**Do crowdsourced reviews actually reflect food safety?**

This project investigates the relationship between NC restaurant health inspection scores and public review platform data (Google Places). We collect, link, and model inspection records against review sentiment to surface restaurants where public perception diverges from ground-truth food safety outcomes.

## Project Structure

```
├── main.py                 <- Entry point: run inference / launch app
├── setup.py                <- Data acquisition and environment setup
├── requirements.txt        <- Python dependencies (full pipeline)
├── requirements-app.txt    <- Python dependencies (app only, for Render)
├── render.yaml             <- Render deployment config
├── report.md               <- Written report
├── Makefile                <- Common tasks (setup, train, run)
├── scripts/
│   ├── make_dataset.py     <- Fetch NC inspection records + Google Places data
│   ├── build_features.py   <- Feature engineering pipeline
│   └── model.py            <- Train models and generate predictions
├── app/                    <- Flask web app (inference only)
├── models/                 <- Serialized trained models (not committed)
├── data/
│   ├── raw/                <- Raw inspection + review data (not committed)
│   ├── processed/          <- Cleaned, merged, feature-engineered data
│   └── outputs/            <- Predictions, evaluation results, plots
├── docs/                   <- GitHub Pages static site
├── notebooks/              <- Exploratory notebooks (not used for grading)
└── extras/                 <- Archived utilities and experiments
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect data (NC inspections + Google Places)
python setup.py

# 3. Train models
python scripts/model.py

# 4. Launch the app
python main.py
```

## Models

Binary classification: **A (safe)** vs **Flagged (B or C inspection grade)**.
231,160 inspections across ~33K restaurants. 3,354 flagged (3,249 B, 105 C), 68:1 imbalance.

| Model | Location | Description |
|---|---|---|
| Naive baseline | `scripts/model.py` | Majority-class predictor (performance floor) |
| Random Forest + SHAP | `scripts/model.py` | Tabular + text-derived features, class_weight=balanced |
| DistilBERT | `scripts/model.py` | Fine-tuned on review text for binary sequence classification |

## Deployment

Two services work together:

| Layer | URL | What it serves |
|---|---|---|
| **Frontend** | [nocapchicken.github.io](https://nocapchicken.github.io) | Static landing page (GitHub Pages, `docs/`) |
| **Backend** | [nocapchicken-github-io.onrender.com](https://nocapchicken-github-io.onrender.com) | Flask API: inference, SHAP explanations, Google Places lookup (Render) |

The frontend calls the Render backend for predictions. The backend is inference-only; no training happens at runtime.

## Data Sources

| Source | Description | License |
|---|---|---|
| [NC DHHS Environmental Health](https://public.cdpehs.com/NCENVPBL/ESTABLISHMENT/ShowESTABLISHMENTTablePage.aspx) | Restaurant inspection scores, grades, and violation records for all 100 NC counties. Collected by NC DHHS and published as public record under NC Public Records Law (G.S. § 132-1). Portal software © Custom Data Processing, Inc. | Public government record |
| [Google Places API](https://developers.google.com/maps/documentation/places/web-service/overview) | Business ratings, review counts, and review text matched to inspection records via fuzzy name+address linking. | Google Terms of Service |

## Ethics

See the written report for a full ethics statement covering data provenance, consent, and potential harms from misclassification.
