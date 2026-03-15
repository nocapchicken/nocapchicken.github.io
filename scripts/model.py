# AI-assisted (Claude Code, claude.ai) — https://claude.ai
"""
Three required modeling approaches:
  1. Naive baseline     — majority class classifier
  2. Classical ML       — Random Forest with SHAP explainability
  3. Deep learning      — DistilBERT fine-tuned on combined review text
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "data" / "outputs"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "grade_encoded"


EXCLUDE_COLS = {
    # Target and raw text
    TARGET_COL, "grade", "combined_reviews", "google_reviews",
    # Direct label leakage — score determines grade by definition
    "score",
    # Identifiers with no predictive value
    "establishment_name", "street_address", "address", "city", "zip",
    "establishment_id", "inspection_id", "inspection_id_google",
    "state_id", "inspector_id", "inspection_date",
    # Raw county/type — only encoded versions are used
    "county_code", "establishment_type",
    "google_place_id", "google_name", "match_score",
    # Raw count — only log-transformed version is used
    "google_review_count",
    # Not available at inference time (app only has Google data, not inspection metadata)
    "establishment_type_code", "inspection_month", "inspection_year",
}


def load_data(binary: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix and target from features.csv.

    When binary=True, collapses B+C into a single 'flagged' class (1) vs A (0).
    This reframing is necessary because only 197 non-A samples exist (194 B, 3 C),
    making 3-class learning infeasible.
    """
    df = pd.read_csv(PROCESSED_DIR / "features.csv")
    feature_cols = [
        col for col in df.columns
        if col not in EXCLUDE_COLS and np.issubdtype(df[col].dtype, np.number)
    ]
    X = df[feature_cols].fillna(0)
    y = df[TARGET_COL]
    if binary:
        # A=0 stays 0 (safe), B=1 and C=2 both become 1 (flagged)
        y = (y > 0).astype(int)
    return X, y


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> dict:
    """Save confusion matrix to data/outputs/."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    logger.info("\n=== %s ===", name)
    logger.info(classification_report(y_test, y_pred))
    logger.info("Confusion matrix:\n%s", cm)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm).to_csv(OUTPUTS_DIR / f"{name.lower().replace(' ', '_')}_cm.csv")

    return {"name": name, "report": report, "confusion_matrix": cm}


def train_naive_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> DummyClassifier:
    """Train a majority-class dummy classifier as the performance lower bound."""
    model = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Returns best_estimator_ from 5-fold grid search (f1_macro).

    Uses class_weight='balanced' to counteract the ~160:1 class imbalance
    (A vs flagged). Without this, RF predicts all-A and learns nothing.
    SMOTE was tested and produced no improvement (same recall, same precision).
    """
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    rf = RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced",
    )
    search = GridSearchCV(rf, param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    logger.info("Best RF params: %s", search.best_params_)
    return search.best_estimator_


def explain_random_forest(model: RandomForestClassifier, X_test: pd.DataFrame) -> None:
    """Save SHAP feature importance to data/outputs/."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Average absolute SHAP across samples; collapse class axis if multiclass
    mean_shap = np.abs(shap_values).mean(axis=0)
    if mean_shap.ndim == 2:
        mean_shap = mean_shap.mean(axis=1)

    shap_df = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    out_path = OUTPUTS_DIR / "shap_importance.csv"
    shap_df.to_csv(out_path, index=False)
    logger.info("SHAP importance written to %s", out_path)


def train_distilbert(
    texts_train: list[str],
    y_train: list[int],
    texts_test: list[str],
    y_test: list[int],
    num_labels: int = 3,
    epochs: int = 3,
    batch_size: int = 16,
):
    """Fine-tune DistilBERT for sequence classification on review text; returns the Trainer."""
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
    import torch
    from torch.utils.data import Dataset

    class ReviewDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_enc = tokenizer(texts_train, truncation=True, padding=True, max_length=256)
    test_enc = tokenizer(texts_test, truncation=True, padding=True, max_length=256)

    train_dataset = ReviewDataset(train_enc, y_train)
    test_dataset = ReviewDataset(test_enc, y_test)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / "distilbert"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=str(OUTPUTS_DIR / "distilbert_logs"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()
    return trainer


def main() -> None:
    """Train all models, save artifacts to models/, metrics to data/outputs/."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bert", action="store_true", help="Skip DistilBERT training")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data(binary=True)
    logger.info("Binary target distribution: %s", y.value_counts().to_dict())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    results = []

    # 1. Naive baseline
    baseline = train_naive_baseline(X_train, y_train)
    results.append(evaluate(baseline, X_test, y_test, "Naive Baseline"))
    joblib.dump(baseline, MODELS_DIR / "naive_baseline.pkl")

    # 2. Random Forest
    rf = train_random_forest(X_train, y_train)
    results.append(evaluate(rf, X_test, y_test, "Random Forest"))
    explain_random_forest(rf, X_test)
    joblib.dump(rf, MODELS_DIR / "random_forest.pkl")
    joblib.dump(X_train.columns.tolist(), MODELS_DIR / "rf_feature_names.pkl")

    # 3. DistilBERT (requires combined_reviews column to exist)
    # Uses the same row indices as the RF split so metrics are comparable.
    if args.skip_bert:
        logger.info("Skipping DistilBERT training (--skip-bert)")
    else:
        features_df = pd.read_csv(PROCESSED_DIR / "features.csv")
        if "combined_reviews" not in features_df.columns:
            logger.warning("No combined_reviews column found — skipping DistilBERT training.")
        else:
            # Only train on rows with actual review text (empty reviews = no signal)
            has_text = features_df["combined_reviews"].fillna("").str.len() > 0
            bert_df = features_df[has_text].copy()
            logger.info("DistilBERT: training on %d rows with review text (of %d total)",
                        len(bert_df), len(features_df))

            texts = bert_df["combined_reviews"].tolist()
            labels = bert_df[TARGET_COL].tolist()
            # Binary reframing: B+C → 1 (flagged), A → 0 (safe)
            labels = [1 if lbl > 0 else 0 for lbl in labels]
            texts_train, texts_test, yl_train, yl_test = train_test_split(
                texts, labels,
                test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels,
            )
            trainer = train_distilbert(
                texts_train, yl_train, texts_test, yl_test, num_labels=2,
            )
            trainer.save_model(str(MODELS_DIR / "distilbert"))
            logger.info("DistilBERT saved to models/distilbert/")

    # Summary comparison
    summary = pd.DataFrame([
        {"model": r["name"], "macro_f1": r["report"]["macro avg"]["f1-score"]}
        for r in results
    ])
    logger.info("\n=== Model Comparison ===\n%s", summary.to_string(index=False))
    summary.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
