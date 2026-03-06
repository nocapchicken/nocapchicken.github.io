# AI-assisted (Claude Code, claude.ai) — https://claude.ai
"""Run DistilBERT at each score threshold (same as RF sweep) on GPU.

Thresholds represent the user's comfort level:
  < 98 = "Lenient"  (A- cutoff — anything below 98 is flagged)
  < 97 = "Moderate" (A  cutoff)
  < 96 = "Strict"   (A+ cutoff)
  < 95 = "Very Strict"
  < 94 = "Ultra Strict"

At each threshold we train a binary classifier: safe (score >= threshold)
vs at-risk (score < threshold), then report sensitivity, specificity,
F1, and ROC AUC — directly comparable to the RF threshold sweep.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUTS_DIR = ROOT / "data" / "outputs"
RANDOM_STATE = 42
THRESHOLDS = [94, 95, 96, 97, 98]


def load_restaurant_data() -> pd.DataFrame:
    """Load per-restaurant dataset with reviews and mean scores."""
    inspections = pd.read_csv(RAW_DIR / "inspections.csv")
    inspections["score"] = pd.to_numeric(inspections["score"], errors="coerce")
    inspections = inspections.dropna(subset=["score"])
    inspections["grade"] = inspections["grade"].str.strip().str.upper()

    google = pd.read_csv(RAW_DIR / "google_reviews.csv")
    google["match_score"] = pd.to_numeric(google["match_score"], errors="coerce")
    google = google[google["match_score"] >= 50]

    merged = inspections.merge(
        google[["state_id", "google_rating", "google_review_count", "google_reviews"]],
        on="state_id", how="inner",
    )

    rest = merged.groupby("state_id").agg(
        mean_score=("score", "mean"),
        latest_grade=("grade", "last"),
        n_inspections=("score", "count"),
        google_rating=("google_rating", "first"),
        google_review_count=("google_review_count", "first"),
        combined_reviews=("google_reviews", "first"),
    ).reset_index()

    rest = rest[rest["latest_grade"].isin(["A", "B", "C"])].copy()
    rest["combined_reviews"] = rest["combined_reviews"].fillna("").astype(str)
    rest = rest[rest["combined_reviews"].str.strip().ne("")].copy()
    return rest


def train_bert_binary(
    tr_texts: list[str],
    y_tr: list[int],
    te_texts: list[str],
    y_te: list[int],
    device: torch.device,
    tokenizer: DistilBertTokenizerFast,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
) -> dict:
    """Train binary DistilBERT, return predictions and probabilities."""

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

    tr_enc = tokenizer(tr_texts, truncation=True, padding=True, max_length=256)
    te_enc = tokenizer(te_texts, truncation=True, padding=True, max_length=256)
    tr_ds = ReviewDataset(tr_enc, y_tr)
    te_ds = ReviewDataset(te_enc, y_te)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    ).to(device)

    # Class-weighted loss for imbalance
    counts = np.bincount(y_tr, minlength=2).astype(float)
    weights = len(y_tr) / (2 * counts + 1e-9)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(device)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

    # Train
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = loss_fn(out.logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info("  Epoch %d/%d — loss: %.4f", epoch + 1, epochs, total_loss / len(loader))

    # Evaluate
    model.eval()
    te_loader = DataLoader(te_ds, batch_size=batch_size)
    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in te_loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            probs = torch.softmax(out.logits, dim=-1)
            all_preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())  # P(at-risk)

    return {"preds": all_preds, "probs": all_probs}


def compute_metrics(y_true: list[int], y_pred: list[int], y_prob: list[int], threshold: int) -> dict:
    """Compute sensitivity, specificity, F1, ROC AUC for one threshold."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return {
        "threshold": threshold,
        "n_safe": int(tn + fp),
        "n_at_risk": int(tp + fn),
        "sensitivity": round(sensitivity, 3),
        "specificity": round(specificity, 3),
        "f1": round(f1, 3),
        "roc_auc": round(auc, 3),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def main():
    """Run BERT threshold sweep matching the RF experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("CUDA: %s | Device: %s", torch.cuda.is_available(),
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

    rest = load_restaurant_data()
    logger.info("Restaurants with reviews: %d", len(rest))

    # Load tokenizer once (reused across thresholds)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    texts = rest["combined_reviews"].tolist()
    scores = rest["mean_score"].values

    results = []

    for thresh in THRESHOLDS:
        # Binary target: 1 = at-risk (score < threshold), 0 = safe
        labels = (scores < thresh).astype(int).tolist()
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        logger.info("\n===== THRESHOLD < %d =====  (at-risk: %d, safe: %d)", thresh, n_pos, n_neg)

        if n_pos < 5:
            logger.warning("  Too few at-risk samples (%d), skipping", n_pos)
            continue

        # 60/20/20 stratified split
        tr_txt, temp_txt, y_tr, y_temp = train_test_split(
            texts, labels, test_size=0.4, random_state=RANDOM_STATE, stratify=labels
        )
        va_txt, te_txt, y_va, y_te = train_test_split(
            temp_txt, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
        )

        logger.info("  Train: %d | Val: %d | Test: %d", len(y_tr), len(y_va), len(y_te))
        logger.info("  Test at-risk: %d / %d", sum(y_te), len(y_te))

        bert_out = train_bert_binary(tr_txt, y_tr, te_txt, y_te, device, tokenizer, epochs=3)
        metrics = compute_metrics(y_te, bert_out["preds"], bert_out["probs"], thresh)
        results.append(metrics)

        print(f"\n--- Threshold < {thresh} ---")
        print(classification_report(
            y_te, bert_out["preds"], labels=[0, 1],
            target_names=["Safe", "At-Risk"], zero_division=0
        ))
        print(f"  Sensitivity: {metrics['sensitivity']:.3f} | "
              f"Specificity: {metrics['specificity']:.3f} | "
              f"F1: {metrics['f1']:.3f} | "
              f"ROC AUC: {metrics['roc_auc']:.3f}")

    # Summary table
    print("\n\n========== BERT THRESHOLD SWEEP SUMMARY ==========")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))

    # Save for notebook comparison
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUTS_DIR / "bert_threshold_sweep.csv", index=False)
    logger.info("Results saved to %s", OUTPUTS_DIR / "bert_threshold_sweep.csv")


if __name__ == "__main__":
    main()
