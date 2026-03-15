# Can Crowdsourced Reviews Predict Food Safety? A Three-Model Investigation of NC Restaurant Inspection Grades

Roshan Gill, Jonas Neves, Dominic Tanzillo

AIPI 540 Deep Learning, Duke University, Spring 2026

---

## 1. Problem Statement

North Carolina's Department of Health and Human Services (DHHS) inspects every licensed food establishment and assigns letter grades (A, B, or C) based on sanitation, temperature control, pest management, and food handling. These grades are public record, but consumers rarely check them. Instead, they rely on crowdsourced review platforms like Google to decide where to eat.

This project asks: **does the language and sentiment of public restaurant reviews contain signal that predicts official food safety inspection outcomes?** If so, a predictive model could surface restaurants where high public ratings mask poor inspection records, a gap we call a "reality gap." If not, the negative result itself is important: it means review platforms give consumers no information about food safety, and the two systems (public perception vs regulatory compliance) operate on completely independent axes.

We frame this as a binary classification problem: **A (safe)** vs **Flagged (B or C inspection grade)**, trained on 31,760 NC DHHS inspection records linked to Google Places review data.

## 2. Data Sources

| Source | Records | Access Method | License |
|--------|---------|---------------|---------|
| NC DHHS Environmental Health | 31,760 inspections (2020-2026) across 100 counties | Scraped from the CDP public inspection portal via ASP.NET CSV export per county per year | Public government record (G.S. 132-1) |
| Google Places API | 17,561 restaurant listings with ratings, review counts, and review text | `googlemaps` Python client, `find_place` + `place` detail requests | Google Terms of Service |

The NC DHHS data provides inspection date, establishment name, address, score (0-100), grade (A/B/C), and inspector ID. Google Places data provides star rating, review count, and up to 5 review texts per listing.

**Linking.** Inspections and Google listings were joined on `state_id` after fuzzy name matching using `rapidfuzz.fuzz.token_sort_ratio` with case normalization. A match threshold of 50 was used. Of 31,760 inspections, 14,529 (45.7%) were successfully linked to a Google listing with review data.

**Class distribution.** The dataset exhibits extreme imbalance: 31,563 A (99.4%), 194 B (0.6%), 3 C (0.01%). We collapsed B and C into a single "Flagged" class (n=197), creating a binary classification problem with a 160:1 imbalance ratio.

## 3. Related Work

Prior work on predicting restaurant health outcomes from online data includes:

- **Kang et al. (2013)** used Yelp review text to predict hygiene violations in Seattle restaurants, reporting over 82% accuracy in identifying severe offenders using unigram features. However, later work showed sampling bias inflated those results, and their dataset had a more balanced violation distribution than NC's heavily A-skewed grading system.

- **Sadilek et al. (2018)**, published in npj Digital Medicine, combined anonymized aggregated location history from opted-in users with illness-related search queries to identify potentially unsafe restaurants in Las Vegas. Their approach relied on indirect behavioral signals rather than review text, achieving meaningful recall on serious violations.

- **Nsoesie et al. (2014)** explored using Yelp data to enhance foodborne illness surveillance, finding that review text could detect outbreak clusters but was less reliable for predicting individual establishment risk.

Our work differs in several ways: (1) we use NC's letter-grade system rather than binary violation detection, (2) our class imbalance is far more extreme (99.4% A), (3) we compare three model architectures from naive baseline through deep learning, and (4) we explicitly test whether review language carries food safety signal at all, rather than assuming it does.

## 4. Evaluation Strategy and Metrics

Metric selection is critical in this domain because the extreme class imbalance makes standard accuracy meaningless. A classifier that predicts A for every restaurant achieves 99.4% accuracy while catching zero unsafe establishments.

We evaluate on two complementary metrics:

**Macro F1.** The unweighted mean of per-class F1 scores. With two classes, this penalizes models that ignore the minority class. A majority-class baseline scores 0.50 macro F1. Any model that fails to improve on this has learned nothing beyond class frequency.

**Flagged recall.** The fraction of truly B/C restaurants that the model correctly identifies. This is the safety-critical metric: a missed B/C grade means a consumer eats at a restaurant with documented sanitation failures, trusting a model that said it was safe. In a deployment context, false negatives are more dangerous than false positives.

We chose not to use accuracy or weighted F1 because both metrics are dominated by the majority class and would report near-perfect scores for a completely uninformative model.

## 5. Data Processing Pipeline

The pipeline consists of four stages:

1. **Collection** (`scripts/make_dataset.py`). NC DHHS inspection records are scraped from the CDP portal via county-level CSV exports for years 2020-2026. Each county's records are fetched using ASP.NET postback with date range filters. Records are filtered to restaurant establishment types (codes 1, 2, 3, 4, 14, 15).

2. **Google linkage** (`scripts/make_dataset.py`). Each inspection record is matched to a Google Places listing using `find_place` with a combined name+address+city query. Match quality is scored using `fuzz.token_sort_ratio` with `processor=default_process` for case normalization. Listings above the match threshold (50) are retained, and up to 5 reviews are fetched via `place` detail requests.

3. **Feature engineering** (`scripts/build_features.py`). Year-level inspection files are merged and deduplicated on `(state_id, inspection_date)`. Google review data is left-joined on `state_id`. Features engineered for the Random Forest include:
   - `google_rating`: Google star rating (1.0-5.0)
   - `google_review_count_log`: log(1 + review count)
   - `review_word_count`: total words in concatenated review text
   - `review_avg_word_len`: mean word length (proxy for review complexity)
   - `safety_keyword_count`: count of safety-adjacent terms (dirty, roach, sick, mold, etc.)
   - `negative_phrase_count`: count of strongly negative phrases (terrible, disgusting, food poisoning, etc.)

4. **Target encoding.** Grades are binarized: A = 0 (safe), B or C = 1 (flagged). A `LabelEncoder` is persisted to `models/grade_encoder.pkl` for reproducibility.

**Data quality fix.** An early version of the pipeline used `rapidfuzz.fuzz.token_sort_ratio` without case normalization. NC DHHS records use ALL-CAPS names; Google uses title case. This caused character-level matching to fail on casing alone ("BOJANGLES" vs "Bojangles" scored 11.1% instead of 100%). After adding `processor=fuzz_utils.default_process`, usable Google matches increased from 432 to 14,868 (a 34x improvement). This fix is documented as the project's primary experiment (Section 9).

## 6. Hyperparameter Tuning Strategy

The Random Forest was tuned via 5-fold `GridSearchCV` optimizing `f1_macro`:

| Hyperparameter | Search Space | Best Value |
|----------------|-------------|------------|
| `n_estimators` | [100, 200] | 100 |
| `max_depth` | [None, 10, 20] | None |
| `min_samples_split` | [2, 5] | 2 |

`class_weight='balanced'` was applied to counteract the 160:1 imbalance. SMOTE oversampling was tested but produced no improvement over balanced class weights alone, so it was not included in the final model.

DistilBERT was trained for 3 epochs with batch size 16, max sequence length 256, `eval_strategy='epoch'`, and `load_best_model_at_end=True` (selected by lowest validation loss). No learning rate sweep was performed due to computational constraints.

## 7. Models Evaluated

### 7.1 Naive Baseline (DummyClassifier)

A majority-class predictor (`strategy='most_frequent'`) that always predicts A. This establishes the performance floor: any useful model must exceed macro F1 = 0.50.

**Rationale.** With 99.4% class A, a model that learns nothing will score extremely well on accuracy (99.4%) but contribute zero predictive value. The naive baseline makes this explicit.

### 7.2 Random Forest with SHAP Explainability

A `RandomForestClassifier` trained on 6 Google-derived features with `class_weight='balanced'`. SHAP `TreeExplainer` provides per-prediction feature attribution, surfaced in the web app.

**Rationale.** Random Forest handles tabular mixed-type features well and is robust to feature scale. The SHAP integration provides interpretable explanations, which is important for a consumer-facing application. The balanced class weights force the model to attend to the minority class rather than defaulting to all-A.

### 7.3 DistilBERT Fine-Tuned on Review Text

`DistilBertForSequenceClassification` fine-tuned on concatenated Google review text (binary, num_labels=2). Trained only on the 14,529 rows with review text, on a T4 GPU via Google Colab.

**Rationale.** If food safety signal exists in review text, a pretrained language model should be able to find it. DistilBERT can capture semantic patterns ("the bathroom was filthy," "saw a roach") that keyword counting would miss. This model tests the upper bound of what NLP can extract from this data.

## 8. Results

### Quantitative Comparison

| Model | Macro F1 | Flagged Precision | Flagged Recall | Accuracy |
|-------|----------|-------------------|----------------|----------|
| Naive Baseline | 0.50 | 0.00 | 0.00 | 99.4% |
| Random Forest (balanced) | 0.50 | 0.00 | 0.00 | 99.4% |
| DistilBERT (binary) | 0.50 | 0.00 | 0.00 | 99.4% |

All three models converge to the same behavior: predict A for every restaurant. Neither structured features nor pretrained language models learned to distinguish safe from flagged establishments using review data.

### SHAP Feature Importance (Random Forest)

| Feature | Mean |SHAP| |
|---------|------------|
| review_avg_word_len | 0.065 |
| google_review_count_log | 0.065 |
| review_word_count | 0.054 |
| google_rating | 0.046 |
| safety_keyword_count | 0.008 |
| negative_phrase_count | 0.008 |

All SHAP values are near zero, confirming that no individual feature carries discriminative signal.

### DistilBERT Training Dynamics

| Epoch | Training Loss | Validation Loss |
|-------|--------------|-----------------|
| 1 | 0.043 | 0.042 |
| 2 | 0.047 | 0.042 |
| 3 | 0.044 | 0.042 |

Validation loss is flat from epoch 1, indicating the model learned to predict the majority class immediately and found no additional signal in subsequent epochs.

### Confusion Matrices

**Naive Baseline (test set, n=6,352):**

|  | Pred Safe | Pred Flagged |
|--|-----------|-------------|
| True Safe | 6,313 | 0 |
| True Flagged | 39 | 0 |

**Random Forest (test set, n=6,352):**

|  | Pred Safe | Pred Flagged |
|--|-----------|-------------|
| True Safe | 6,312 | 1 |
| True Flagged | 39 | 0 |

**DistilBERT (test set, n=2,906):**

|  | Pred Safe | Pred Flagged |
|--|-----------|-------------|
| True Safe | 2,886 | 0 |
| True Flagged | 20 | 0 |

All three confusion matrices are nearly identical to the trivial all-A prediction.

## 9. Error Analysis

We examine 5 specific false negatives: restaurants with grade B or C that the model predicted as safe. All 39 flagged restaurants in the test set were misclassified.

### Case 1: Little Caesar's #84 (Grade B, Score 89.5)

- **Google rating:** 3.8 (522 reviews)
- **Model P(flagged):** 0.150
- **Safety keywords:** 2 ("raw" appears twice in reviews)
- **Review excerpt:** "My pizza and crazy bread were practically raw. I had to try to bake it myself when I got home."
- **Root cause:** Despite reviews mentioning raw food (a safety-relevant signal), the model assigns low probability because the overall feature profile (3.8 stars, 522 reviews, moderate word count) is statistically indistinguishable from an A-grade restaurant.
- **Mitigation:** Weight safety-keyword features more heavily, or train a secondary classifier specifically on review sentences containing safety-adjacent language. The word "raw" in a food context is a strong prior for violations.

### Case 2: Little Caesars New Bern (Grade B, Score 88.0)

- **Google rating:** 3.6 (608 reviews)
- **Model P(flagged):** 0.050
- **Negative phrases:** 2
- **Review excerpt:** "There's a long black hair all across my food... complete waste of my money."
- **Root cause:** Hair contamination is a classic inspection violation, but the model has no feature to capture foreign-object complaints specifically. The negative phrase count (2) is within the normal range for A-grade restaurants.
- **Mitigation:** Add a "contamination" keyword list (hair, foreign object, found something in) as a dedicated feature. Currently, these signals are diluted in the general negative phrase count.

### Case 3: A Taste of Big Bois (Grade B, Score 88.5)

- **Google rating:** 4.0 (75 reviews)
- **Model P(flagged):** 0.040
- **Safety keywords:** 0, **Negative phrases:** 0
- **Review excerpt:** "These 2 meals were sooooo good! Even though he's the only chef, it took less than 20 min..."
- **Root cause:** Reviews are overwhelmingly positive and discuss food quality, not safety. The inspection failure (score 88.5, grade B) reflects back-of-house issues (e.g., temperature logs, sanitizer concentration) that customers cannot observe or write about.
- **Mitigation:** This case is fundamentally unlinkable from review data. The only viable strategy is to incorporate structured inspection history (has the establishment been flagged before?) rather than relying solely on review text.

### Case 4: La Palma (Grade B, Score 86.0)

- **Google rating:** 4.3 (249 reviews)
- **Model P(flagged):** 0.020
- **Review excerpt:** "I have spent the past month dreading taco nights... anytime they fail me I come to my little corner store..."
- **Root cause:** High Google rating (4.3) and entirely positive reviews. The inspection score of 86.0 reflects violations invisible to customers. With 0 safety keywords and 0 negative phrases, the model has no textual signal to work with.
- **Mitigation:** Use inspection date proximity as a feature. If a restaurant's most recent inspection was >12 months ago, flag it for recency bias. However, this requires access to inspection metadata at inference time.

### Case 5: Soma Bistro Curry N Cake (Grade C, Score 78.0)

- **Google rating:** N/A (no Google match)
- **Model P(flagged):** 0.464
- **Root cause:** This is one of only 3 grade-C restaurants in the entire dataset, and it has no Google data at all. The model has zero features to work with. The 0.464 flagged probability is the model's default for restaurants with no review data.
- **Mitigation:** For unmatched restaurants, surface a "no review data available" warning rather than a prediction. Alternatively, broaden the Google matching to include partial name matches or nearby location searches.

### Summary of Root Causes

| Root Cause | Cases | Fundamental? |
|------------|-------|-------------|
| Reviews discuss taste/service, not safety | 3, 4 | Yes |
| Safety-relevant review language not weighted | 1, 2 | Partially addressable |
| No Google review data at all | 5 | Addressable via better matching |

The dominant failure mode (cases 3, 4) is structural: customers write about what they experience (food quality, service speed, atmosphere), while inspectors evaluate what customers cannot see (sanitizer concentration, cold-holding temperatures, pest evidence). This gap is not a modeling failure. It is a data limitation.

## 10. Experiment: Impact of Fuzzy Matching Quality on Model Performance

### Hypothesis

The initial pipeline's case-sensitive fuzzy matching (`fuzz.token_sort_ratio` without processor) artificially deflated match scores between NC DHHS names (ALL CAPS) and Google names (title case). We hypothesized that fixing this bug would dramatically increase Google data coverage and potentially improve model performance.

### Method

1. **Before fix:** Match scores computed without case normalization. "BOJANGLES" vs "Bojangles" = 11.1%.
2. **After fix:** Added `processor=fuzz_utils.default_process` (lowercases + strips non-alphanumeric). Same comparison = 100%.
3. Recomputed all 17,561 match scores. Rebuilt feature matrix. Retrained all models.

### Results

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Rows with match_score >= 50 | 432 | 14,868 |
| Rows with Google rating in features | ~400 | 14,564 |
| Median match score | 20.8 | 74.3 |
| RF Macro F1 | 0.50 | 0.50 |
| DistilBERT Macro F1 | 0.33 | 0.50 |

### Interpretation

The fix achieved a **34x increase in Google data coverage** (432 to 14,868 matched restaurants). However, model performance did not improve. This tells us that data sparsity was not the bottleneck. Even with 14,529 reviews available, the features derived from those reviews do not separate A from B/C restaurants.

### Recommendation

This experiment validates the negative result: the review-grade disconnect is not caused by insufficient data linkage. The features are genuinely non-discriminative. Future work should explore data sources that are closer to the inspection process (violation history, inspection frequency, complaint records) rather than investing further in review coverage.

## 11. Conclusions

We investigated whether crowdsourced Google reviews can predict NC restaurant food safety inspection grades. Using a three-model pipeline (majority-class baseline, Random Forest with SHAP, and DistilBERT fine-tuned on review text), we found that **no model architecture can reliably distinguish safe (grade A) from flagged (grade B/C) restaurants using review data alone.**

The core reason is a domain mismatch: customers write about taste, service, and atmosphere. Inspectors evaluate sanitization, temperature control, and pest management. These two information domains have minimal overlap in natural language.

This negative result is itself a contribution. It demonstrates that consumer review platforms, despite their value for evaluating dining experience, provide no reliable signal about food safety compliance. Public health information and consumer experience are orthogonal, and models that conflate them risk giving false confidence.

## 12. Future Work

Given another semester, we would pursue three directions:

1. **Incorporate inspection history as features.** A restaurant's prior inspection scores, violation counts, and time since last inspection are strong predictors of future outcomes. These are available from NC DHHS and would shift the model from review-based prediction to inspection-trend prediction.

2. **Complaint-driven signals.** NC DHHS accepts public complaints that trigger inspections. Mining complaint text (which does describe safety concerns) rather than review text (which describes dining experience) would provide features with direct domain relevance.

3. **Anomaly detection reframing.** Rather than binary classification, treat the problem as anomaly detection: identify restaurants whose review profile is statistically unusual given their inspection history. This reframing sidesteps the class imbalance problem by not requiring labeled examples of each class.

4. **Cross-state validation.** Test whether the finding generalizes to states with different grading distributions. NYC, for example, has a higher proportion of B and C grades, which might provide enough minority-class signal for models to learn from.

5. **Review-level (not restaurant-level) classification.** Rather than predicting grades from aggregated reviews, classify individual review sentences as safety-relevant or not, then use the proportion of safety-relevant sentences as a feature.

## 13. Commercial Viability

A model that reliably predicts food safety from publicly available data would have clear commercial value: integration into restaurant discovery platforms, insurance underwriting for food service businesses, and public health surveillance tools.

However, our results show that review data alone is insufficient. A commercially viable product would need to combine review analysis with (1) public inspection records, (2) complaint data, and (3) operational signals (e.g., staff turnover, hours of operation changes). The review component would serve as one input among many, not as the sole predictor.

The web application we built (nocapchicken.github.io) demonstrates the UX pattern: search for a restaurant, see its Google rating alongside a model-generated risk assessment with SHAP explanations. The interface is production-ready; the underlying model needs richer data sources to be commercially useful.

## 14. Ethics Statement

**Data provenance.** NC DHHS inspection records are public government records under NC Public Records Law (G.S. 132-1). Google Places data was accessed via the official API under Google's Terms of Service. No personally identifiable information about restaurant patrons was collected or used.

**Potential harms.** A false-positive prediction (labeling a safe restaurant as flagged) could cause reputational harm to a business. A false-negative prediction (labeling an unsafe restaurant as safe) could lead consumers to eat at establishments with sanitation violations. Our current model produces no false positives and 100% false negatives, which is equivalent to providing no prediction at all, but any improved model must carefully calibrate its threshold to minimize consumer harm.

**Limitations of inference.** An inspection grade reflects a point-in-time assessment. Conditions can change between inspections. Users should be reminded that model predictions are not substitutes for official health inspection records.

**Bias considerations.** The fuzzy name matching pipeline may systematically fail to link restaurants with non-English names, leading to lower Google data coverage for certain cuisines. We did not audit for this bias but acknowledge it as a limitation.

---

*AI-assisted (Claude Code, claude.ai) — https://claude.ai*
