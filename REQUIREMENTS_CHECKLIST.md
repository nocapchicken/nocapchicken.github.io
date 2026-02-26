# Project Requirements Checklist

> **Purpose:** Track rubric compliance. Used by team members, PR reviewers, and the automated
> Claude Code review workflow (`.github/workflows/claude-review.yml`).
>
> Mark items `[x]` as they are completed. Items marked ⚠️ are graded and will result in
> point deductions or zeros if missed.
>
> **Grader notes from prior cohort submission (83/100):** Deductions came from:
> - `-5` Experiment was conducted but not strongly connected to final modeling decisions
> - `-5` Error analysis mitigation strategies were too vague ("ask for the next paper")
> - `-5` PRs lacked meaningful descriptions — *every PR needs a description, even a short one*
> - `-2` "LGTM" is **not** an acceptable PR review — reviewers must leave substantive comments

---

## 1. Modeling — Three Required Approaches ⚠️

All three must be **implemented, evaluated, and findable** in the repo.

| # | Requirement | Location | Status |
|---|-------------|----------|--------|
| M1 | Naive baseline (majority class / mean predictor) | `scripts/model.py` → `train_naive_baseline()` | ✅ |
| M2 | Classical (non-DL) ML model | `scripts/model.py` → `train_random_forest()` | ✅ |
| M3 | Deep learning model | `scripts/model.py` → `train_distilbert()` | ✅ |
| M4 | All three documented in README with file locations | `README.md` § Models | ✅ |
| M5 | Rationale for each model written up in report | Written report | ⬜ |
| M6 | Selected final model clearly identified (and justified) | Written report | ⬜ |
| M7 | Trained model artifacts present or reproducible | `models/` (currently empty — must run `scripts/model.py`) | ⬜ |

---

## 2. Required Experimentation ⚠️

At least **one focused experiment** must be implemented and written up.

> **Grader warning (-5 risk):** A previous team ran a "frozen vs. fine-tuned embedding"
> experiment but lost 5 points because the results didn't clearly *inform* their final modeling
> choice. The experiment must answer a question that changes or validates a decision in your system.

- [ ] **EX1** — Experiment is well-motivated: poses a specific question about your system
- [ ] **EX2** — Experimental plan documented (hypothesis, method, metrics)
- [ ] **EX3** — Results reported with numbers/visualizations
- [ ] **EX4** — Interpretation written: what do the results tell you?
- [ ] **EX5** — Experiment **directly informs a modeling or system design decision**
- [ ] **EX6** — Actionable recommendations drawn from experiment

Suggested options (pick one):
- Ablation: model with only Yelp features vs. only Google vs. both → informs which API is worth the cost
- Sensitivity: training set size vs. macro F1 → informs minimum viable data collection
- Robustness: inject ±0.5 star noise into ratings → informs whether to trust sparse review counts
- Preprocessing: fuzzy-match threshold (60/70/80) effect on dataset yield and model score

---

## 3. Interactive Application ⚠️

> **Zero-tolerance**: if the app doesn't run when graded → **0 for this section**.

- [x] **APP1** — Flask app exists and runs inference only (no training in app code)
- [x] **APP2** — Good UX — polished interface, not a bare Streamlit demo
- [ ] **APP3** — Publicly accessible via internet (deployed URL) — *README says TBD*
- [ ] **APP4** — Live for at least **one week** after submission date
- [ ] **APP5** — Deployment URL recorded in `README.md` § Live App

---

## 4. Written Report ⚠️

Format: NeurIPS/ICML-style paper, white paper, or technical report.

### Required Sections

- [ ] **R01** — Problem Statement
- [ ] **R02** — Data Sources (with provenance and access method)
- [ ] **R03** — Related Work (review of prior literature)
- [ ] **R04** — Evaluation Strategy & Metrics (with justification — *"this is critical"*)
- [ ] **R05** — Modeling Approach → Data Processing Pipeline (with rationale per step)
- [ ] **R06** — Hyperparameter Tuning Strategy (GridSearchCV documented)
- [ ] **R07** — Models Evaluated: Naive baseline, Classical ML, Deep learning (with rationale)
- [ ] **R08** — Results: quantitative comparison across all models and metrics
- [ ] **R09** — Results: visualizations and confusion matrices
- [ ] **R10** — Error Analysis: **5 specific mispredictions** identified (restaurant name, true grade, predicted grade)
- [ ] **R11** — Error Analysis: root cause explained for each (data quality? feature gap? class imbalance?)
- [ ] **R12** — Error Analysis: **concrete, specific** mitigation strategies per case — *not* generic statements; grader penalized (-5) vague strategies like "collect more data" without specifics
- [ ] **R13** — Experiment Write-Up (plan → results → interpretation → recommendations)
- [ ] **R14** — Conclusions
- [ ] **R15** — Future Work ("what would you do with another semester?") — grader noted this was "a little hidden" in a prior submission; make it a clearly labelled section
- [ ] **R16** — Commercial Viability Statement
- [ ] **R17** — Ethics Statement

---

## 5. In-Class Pitch (5 min hard stop) ⚠️

- [ ] **P1** — Problem & Motivation slide(s)
- [ ] **P2** — Approach Overview slide(s)
- [ ] **P3** — Live Demo prepared and rehearsed
- [ ] **P4** — Results, Insights, or Key Findings slide(s)
- [ ] **P5** — Presentation stays within 5 minutes

---

## 6. Code & Repository

### Repo Structure

- [x] `README.md` — project description, setup instructions
- [x] `requirements.txt` — all dependencies pinned
- [x] `setup.py` — data acquisition pipeline
- [x] `main.py` — entry point / app launcher
- [x] `scripts/make_dataset.py`
- [x] `scripts/build_features.py`
- [x] `scripts/model.py`
- [x] `models/` — directory exists (artifacts generated after training)
- [x] `data/raw/`, `data/processed/`, `data/outputs/`
- [x] `notebooks/` — directory exists
- [x] `.gitignore`
- [x] `Makefile`
- [ ] **REPO1** — At least one exploration notebook in `notebooks/`

### Code Quality ⚠️

- [x] **CQ1** — All code modularized into classes/functions (no loose executable code)
- [x] **CQ2** — No executable code outside `if __name__ == "__main__"` guards
- [x] **CQ3** — Descriptive variable names throughout
- [x] **CQ4** — Docstrings on all public functions
- [ ] **CQ5** — **AI usage attributed** at top of each file that used AI assistance (link to source required)
- [ ] **CQ6** — External code/libraries attributed at top of relevant files

> Note: Jupyter notebooks are allowed **only** in `notebooks/` and will not be graded directly.

### Git Best Practices ⚠️

> **Grader warning (-5 + -2 risk):** Prior team lost 7 points here:
> - Every PR must have a written description, even a short one
> - "LGTM" is **not** an acceptable review — leave at least one substantive comment per PR

- [x] **GIT1** — Feature branches in use (current: `feature/flask-app`)
- [x] **GIT2** — PR template in `.github/PULL_REQUEST_TEMPLATE/`
- [ ] **GIT3** — All code merged via PRs (no direct commits to `main`)
- [ ] **GIT4** — Every PR has a meaningful Summary (use the PR template — 1 paragraph minimum)
- [ ] **GIT5** — Every PR reviewed with **substantive comments** — not just "LGTM" or a rubber-stamp
- [ ] **GIT6** — `.env` is **never** committed (check `.gitignore`)
- [ ] **GIT7** — Large data files / model binaries are **never** committed

### Project Novelty (choose one)

- [ ] **NOV1** — Working on a dataset/problem with no prior modeling approaches, OR
- [ ] **NOV2** — Clearly explains what is new/novel vs. prior approaches (with citations), and achieves near-SOTA or better explainability

---

## 7. Pre-Submission Checklist

Run through this before final submission:

- [ ] `python setup.py` runs end-to-end without errors
- [ ] `python scripts/build_features.py` produces `data/processed/features.csv`
- [ ] `python scripts/model.py` trains all three models and writes artifacts to `models/`
- [ ] `python main.py` launches the Flask app and it is reachable in browser
- [ ] Live deployment URL is working and accessible without login
- [ ] README deployment link updated from "TBD" to the actual URL
- [ ] Written report submitted in the required format
- [ ] Pitch deck/slides prepared
- [ ] Repo is public (or access granted to grader)

---

## Status Summary

| Category | Done | Total | % |
|----------|------|-------|---|
| Modeling | 4 | 7 | 57% |
| Experimentation | 0 | 5 | 0% |
| App | 2 | 5 | 40% |
| Written Report | 0 | 17 | 0% |
| Pitch | 0 | 5 | 0% |
| Repo / Code Quality | 13 | 16 | 81% |
| Git Best Practices | 2 | 7 | 29% |
| **Total** | **21** | **62** | **34%** |

> Last updated: 2026-02-26
