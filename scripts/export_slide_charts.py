# AI-assisted (Claude Code, claude.ai) — https://claude.ai
"""Export key charts as PNGs for the final presentation slides."""

import os

import matplotlib.pyplot as plt
import numpy as np


SLIDES_DIR = os.path.expanduser(
    "~/Github/jonasneves/agora/slides/nocapchicken/img"
)


def export_model_comparison(out_dir):
    """Bar chart of all 4 binary models' test macro F1."""
    models = [
        ("Naive Baseline", 0.496),
        ("RF (structured)", 0.568),
        ("RF (struct+tfidf)", 0.570),
        ("DistilBERT", 0.580),
    ]
    names = [m[0] for m in models]
    scores = [m[1] for m in models]
    colors = ["#aaaaaa", "#66c2a5", "#8da0cb", "#fc8d62"]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(names, scores, color=colors, edgecolor="white", height=0.55)
    ax.axvline(0.50, color="#999999", ls="--", lw=1, alpha=0.6)
    ax.text(0.501, -0.6, "baseline = 0.50", fontsize=8, color="#666666")

    for bar, score in zip(bars, scores):
        ax.text(
            score + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}", va="center", fontsize=11, fontweight="bold",
        )

    ax.set_xlim(0.4, 0.65)
    ax.set_xlabel("Macro F1 (binary: A vs flagged)", fontsize=10)
    ax.set_title("Test Set — All Models", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "model_comparison.png"),
        dpi=200, bbox_inches="tight", transparent=True,
    )
    plt.close()
    print("  model_comparison.png")


def export_threshold_sweep(out_dir):
    """Sensitivity vs specificity across score thresholds."""
    # From notebook Part 4 results
    thresholds = [90, 91, 92, 93, 94, 95, 96, 97, 98]
    sensitivity = [0.638, 0.650, 0.638, 0.695, 0.733, 0.767, 0.787, 0.821, 0.847]
    specificity = [0.963, 0.960, 0.963, 0.958, 0.948, 0.937, 0.922, 0.897, 0.857]
    macro_f1 = [0.611, 0.620, 0.611, 0.650, 0.676, 0.701, 0.716, 0.744, 0.761]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        thresholds, sensitivity, "o-",
        label="Sensitivity (catch at-risk)", color="#e41a1c", lw=2.5, ms=7,
    )
    ax.plot(
        thresholds, specificity, "s-",
        label="Specificity (clear safe)", color="#377eb8", lw=2.5, ms=7,
    )
    ax.plot(
        thresholds, macro_f1, "D--",
        label="Macro F1", color="#4daf4a", lw=1.5, ms=5, alpha=0.8,
    )
    ax.fill_between(
        thresholds, sensitivity, specificity, alpha=0.08, color="gray",
    )

    best_idx = np.argmax(macro_f1)
    ax.axvline(
        thresholds[best_idx], color="#4daf4a", ls=":", lw=1.5, alpha=0.6,
    )
    ax.annotate(
        f"Best F1 = {macro_f1[best_idx]:.3f}",
        xy=(thresholds[best_idx], macro_f1[best_idx]),
        xytext=(thresholds[best_idx] - 3, macro_f1[best_idx] - 0.08),
        fontsize=9, fontweight="bold", color="#4daf4a",
        arrowprops=dict(arrowstyle="->", color="#4daf4a", lw=1.2),
    )

    ax.set_xlabel(
        'Score threshold ("at-risk" if inspection score < this)', fontsize=10,
    )
    ax.set_ylabel("Rate", fontsize=10)
    ax.set_title(
        "Threshold Sweep — Where Do Reviews Predict?", fontsize=12,
        fontweight="bold",
    )
    ax.set_ylim(0.5, 1.02)
    ax.set_xticks(thresholds)
    ax.legend(loc="lower left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "threshold_sweep.png"),
        dpi=200, bbox_inches="tight", transparent=True,
    )
    plt.close()
    print("  threshold_sweep.png")


def export_shap_importance(out_dir):
    """SHAP feature importance for RF model."""
    features = [
        "review_word_count",
        "google_review_count (log)",
        "review_avg_word_len",
        "google_rating",
        "safety_keyword_count",
        "negative_phrase_count",
    ]
    importances = [0.0595, 0.0561, 0.0529, 0.0420, 0.0113, 0.0105]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#8da0cb"] * 4 + ["#fc8d62"] * 2
    ax.barh(features[::-1], importances[::-1], color=colors[::-1], height=0.5)

    for i, (feat, imp) in enumerate(zip(features[::-1], importances[::-1])):
        ax.text(imp + 0.001, i, f"{imp:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Mean |SHAP|", fontsize=10)
    ax.set_title("RF Feature Importance (SHAP)", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "shap_importance.png"),
        dpi=200, bbox_inches="tight", transparent=True,
    )
    plt.close()
    print("  shap_importance.png")


def main():
    """Generate all slide charts."""
    os.makedirs(SLIDES_DIR, exist_ok=True)
    print(f"Exporting to {SLIDES_DIR}")
    export_model_comparison(SLIDES_DIR)
    export_threshold_sweep(SLIDES_DIR)
    export_shap_importance(SLIDES_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
