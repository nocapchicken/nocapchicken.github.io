# AI-assisted (Claude Code, claude.ai) — https://claude.ai
"""
model.py — Train and evaluate all three required modeling approaches.

Models:
  1. Naive baseline     — majority class classifier
  2. Classical ML       — Random Forest with SHAP explainability
  3. Deep learning      — DistilBERT fine-tuned on combined review text

Usage:
    python scripts/model.py
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
    TARGET_COL, "grade", "combined_reviews", "establishment_name",
    "address", "yelp_reviews", "google_reviews",
}


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the processed feature matrix, returning (X, y)."""
    df = pd.read_csv(PROCESSED_DIR / "features.csv")
    feature_cols = [
        col for col in df.columns
        if col not in EXCLUDE_COLS and np.issubdtype(df[col].dtype, np.number)
    ]
    X = df[feature_cols].fillna(0)
    y = df[TARGET_COL]
    return X, y


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> dict:
    """Log classification metrics and save confusion matrix to disk."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    logger.info("\n=== %s ===", name)
    logger.info(classification_report(y_test, y_pred))
    logger.info("Confusion matrix:\n%s", cm)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm).to_csv(OUTPUTS_DIR / f"{name.lower().replace(' ', '_')}_cm.csv")

    return {"name": name, "report": report, "confusion_matrix": cm}


# ---------------------------------------------------------------------------
# 1. Naive baseline
# ---------------------------------------------------------------------------

def train_naive_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> DummyClassifier:
    """Majority-class baseline that all other models must beat."""
    model = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# 2. Classical ML — Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Random Forest with grid-searched hyperparameters."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    search = GridSearchCV(rf, param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    logger.info("Best RF params: %s", search.best_params_)
    return search.best_estimator_


def explain_random_forest(model: RandomForestClassifier, X_test: pd.DataFrame) -> None:
    """Compute and save SHAP feature importance to data/outputs/."""
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


# ---------------------------------------------------------------------------
# 3. Deep learning — DistilBERT
# ---------------------------------------------------------------------------

def train_distilbert(
    texts_train: list[str],
    y_train: list[int],
    texts_test: list[str],
    y_test: list[int],
    num_labels: int = 3,
    epochs: int = 3,
    batch_size: int = 16,
):
    """Fine-tune DistilBERT on combined review text for grade classification."""
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
            item["labels"] = torch.tensor(self.labels[idx])
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
        evaluation_strategy="epoch",
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


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Train all three models, evaluate, and save artifacts."""
    logging.basicConfig(level=logging.INFO)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
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

    # 3. DistilBERT (requires combined_reviews column to exist)
    features_df = pd.read_csv(PROCESSED_DIR / "features.csv")
    if "combined_reviews" in features_df.columns:
        texts = features_df["combined_reviews"].fillna("").tolist()
        labels = features_df[TARGET_COL].tolist()
        texts_train, texts_test, yl_train, yl_test = train_test_split(
            texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        trainer = train_distilbert(texts_train, yl_train, texts_test, yl_test)
        trainer.save_model(str(MODELS_DIR / "distilbert"))
        logger.info("DistilBERT saved to models/distilbert/")
    else:
        logger.warning("No combined_reviews column found — skipping DistilBERT training.")

    # Summary comparison
    summary = pd.DataFrame([
        {"model": r["name"], "macro_f1": r["report"]["macro avg"]["f1-score"]}
        for r in results
    ])
    logger.info("\n=== Model Comparison ===\n%s", summary.to_string(index=False))
    summary.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
