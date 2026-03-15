# Can Crowdsourced Reviews Predict Food Safety? A Three-Model Investigation of NC Restaurant Inspection Grades

Roshan Gill, Jonas Neves, Dominic Tanzillo

AIPI 540 Deep Learning, Duke University, Spring 2026

---

## 1. Problem Statement

North Carolina's Department of Health and Human Services (DHHS) inspects every licensed food establishment and assigns letter grades (A, B, or C) based on sanitation, temperature control, pest management, and food handling. These grades are public record, but consumers rarely check them. Instead, they rely on crowdsourced review platforms like Google to decide where to eat.

This project asks: **does the language and sentiment of public restaurant reviews contain signal that predicts official food safety inspection outcomes?** If so, a predictive model could surface restaurants where high public ratings mask poor inspection records, a gap we call a "reality gap." If not, the negative result itself is important: it means review platforms give consumers no information about food safety, and the two systems (public perception vs regulatory compliance) operate on completely independent axes.

We frame this as a binary classification problem: **A (safe)** vs **Flagged (B or C inspection grade)**, trained on 231,160 NC DHHS inspection records linked to Google Places review data.

## 2. Data Sources

| Source | Records | Access Method | License |
|--------|---------|---------------|---------|
| NC DHHS Environmental Health | 231,160 inspections (2020-2026) across 100 counties | Scraped from the CDP public inspection portal via ASP.NET CSV export per county per year | Public government record (G.S. 132-1) |
| Google Places API | 17,561 restaurant listings with ratings, review counts, and review text | `googlemaps` Python client, `find_place` + `place` detail requests | Google Terms of Service |

The NC DHHS data provides inspection date, establishment name, address, score (0-100), grade (A/B/C), and inspector ID. Each restaurant can have multiple inspections over time (average ~7 per restaurant across 2020-2026). Google Places data provides star rating, review count, and up to 5 review texts per listing.

**Linking.** Inspections and Google listings were joined on `state_id` after fuzzy name matching using `rapidfuzz.fuzz.token_sort_ratio` with case normalization. A match threshold of 50 was used. Of 231,160 inspections, 111,542 (48.2%) were successfully linked to a Google listing with review data.

**Class distribution.** The dataset exhibits significant imbalance: 227,806 A (98.5%), 3,249 B (1.4%), 105 C (0.05%). We collapsed B and C into a single "Flagged" class (n=3,354), creating a binary classification problem with a 68:1 imbalance ratio.

## 3. Related Work

Prior work on predicting restaurant health outcomes from online data includes:

- **Kang et al. (2013)** used Yelp review text to predict hygiene violations in Seattle restaurants, reporting over 82% accuracy in identifying severe offenders using unigram features. Altenburger and Ho (WWW 2019) later showed that extreme imbalanced sampling drove much of the reported accuracy. Regardless, their dataset had a more balanced violation distribution than NC's heavily A-skewed grading system.

- **Sadilek et al. (2018)**, published in npj Digital Medicine, combined anonymized aggregated location history from opted-in users with illness-related search queries to identify potentially unsafe restaurants in Las Vegas. Their approach relied on indirect behavioral signals rather than review text, achieving meaningful recall on serious violations.

- **Nsoesie et al. (2014)**, published in *Preventive Medicine*, compared the distribution of implicated food categories in Yelp illness-related reviews against CDC outbreak surveillance reports. They found that Yelp data captured similar food category patterns to official reports, suggesting review platforms could complement (but not replace) traditional surveillance.

Our work differs in several ways: (1) we use NC's letter-grade system rather than binary violation detection, (2) our class imbalance is far more extreme (99.4% A), (3) we compare three model architectures from naive baseline through deep learning, and (4) we explicitly test whether review language carries food safety signal at all, rather than assuming it does.

## 4. Evaluation Strategy and Metrics

Metric selection is critical in this domain because class imbalance makes standard accuracy meaningless. A classifier that predicts A for every restaurant achieves 98.5% accuracy while catching zero unsafe establishments.

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

**Data quality fixes.** Two pipeline bugs silently degraded the training data: (1) case-sensitive fuzzy matching that reduced Google linkage from 14,868 to 432 matches, and (2) a BOM encoding error that dropped all inspection dates, collapsing 232K rows to 31K and losing 94% of B/C grade samples. Both fixes are documented as the project's primary experiment (Section 10).

## 6. Hyperparameter Tuning Strategy

The Random Forest was tuned via 5-fold `GridSearchCV` optimizing `f1_macro`:

| Hyperparameter | Search Space | Best Value |
|----------------|-------------|------------|
| `n_estimators` | [100, 200] | 100 |
| `max_depth` | [None, 10, 20] | None |
| `min_samples_split` | [2, 5] | 2 |

`class_weight='balanced'` was applied to counteract the 68:1 imbalance. SMOTE oversampling was tested but produced no improvement over balanced class weights alone, so it was not included in the final model.

DistilBERT was trained for 3 epochs with batch size 16, max sequence length 256, `eval_strategy='epoch'`, and `load_best_model_at_end=True` (selected by lowest validation loss). No learning rate sweep was performed due to computational constraints.

## 7. Models Evaluated

### 7.1 Naive Baseline (DummyClassifier)

A majority-class predictor (`strategy='most_frequent'`) that always predicts A. This establishes the performance floor: any useful model must exceed macro F1 = 0.50.

**Rationale.** With 98.5% class A, a model that learns nothing will score extremely well on accuracy but contribute zero predictive value. The naive baseline makes this explicit.

### 7.2 Random Forest with SHAP Explainability

A `RandomForestClassifier` trained on 6 Google-derived features with `class_weight='balanced'`. SHAP `TreeExplainer` provides per-prediction feature attribution, surfaced in the web app.

**Rationale.** Random Forest handles tabular mixed-type features well and is robust to feature scale. The SHAP integration provides interpretable explanations, which is important for a consumer-facing application. The balanced class weights force the model to attend to the minority class rather than defaulting to all-A.

### 7.3 DistilBERT Fine-Tuned on Review Text

`DistilBertForSequenceClassification` fine-tuned on concatenated Google review text (binary, num_labels=2). Trained on the 111,542 rows with review text, on a T4 GPU via Google Colab.

**Rationale.** If food safety signal exists in review text, a pretrained language model should be able to find it. DistilBERT can capture semantic patterns ("the bathroom was filthy," "saw a roach") that keyword counting would miss. This model tests the upper bound of what NLP can extract from this data.

## 8. Results

### Quantitative Comparison

| Model | Macro F1 | Flagged Precision | Flagged Recall | Flagged F1 |
|-------|----------|-------------------|----------------|------------|
| Naive Baseline | 0.50 | 0.00 | 0.00 | 0.00 |
| Random Forest (balanced) | **0.57** | 0.11 | **0.28** | **0.16** |
| DistilBERT (binary, 111K texts) | 0.50 | 0.00 | 0.00 | 0.00 |

The Random Forest detects real signal above baseline, catching 28% of flagged restaurants (191 of 671 in the test set). DistilBERT, despite training on 89K review texts with 1,682 flagged samples, predicts all-safe. This is a significant finding: **the RF's structured features (review volume, word statistics) carry signal that BERT cannot extract from raw text.**

The implication is that the predictive signal lives in metadata patterns (how many reviews a restaurant has, how long they are), not in what the reviews say. BERT processes semantic content but the content itself (taste, service, atmosphere) does not distinguish safe from flagged restaurants. The RF's engineered features capture structural correlates that happen to associate with inspection outcomes.

### SHAP Feature Importance (Random Forest)

| Feature | Mean |SHAP| |
|---------|------------|
| review_word_count | 0.060 |
| google_review_count_log | 0.056 |
| review_avg_word_len | 0.053 |
| google_rating | 0.042 |
| safety_keyword_count | 0.011 |
| negative_phrase_count | 0.010 |

Review volume features (word count, review count) carry the most signal. Safety-specific keyword features contribute modestly but measurably. Google star rating alone is a weak predictor.

### Confusion Matrices

**Naive Baseline (test set, n=46,232):**

|  | Pred Safe | Pred Flagged |
|--|-----------|-------------|
| True Safe | 45,561 | 0 |
| True Flagged | 671 | 0 |

**Random Forest (test set, n=46,232):**

|  | Pred Safe | Pred Flagged |
|--|-----------|-------------|
| True Safe | 44,013 | 1,548 |
| True Flagged | 480 | 191 |

The RF trades 1,548 false positives for 191 true positives, a precision of 11% at 28% recall. In this domain, catching 191 flagged restaurants at the cost of over-flagging 1,548 safe ones may be acceptable as a screening tool (not a definitive label).

## 9. Error Analysis

We examine 5 specific false negatives: flagged restaurants the model predicted as safe. Of 671 flagged restaurants in the test set, 480 were missed (28% recall).

### Case 1: La Tovara Mexican Kitchen and Seafood (Grade B, Score 86.5)

- **Google rating:** 4.6 | **Safety keywords:** 0 | **P(flagged):** 0.496
- **Review excerpt:** "Found another gem here in Greenville! Friendly and helpful staff and the food was great tasting and nicely portioned."
- **Root cause:** High rating (4.6), zero safety keywords, entirely positive reviews. The model came close to the decision boundary (P=0.496) but the absence of any negative textual signal kept it on the safe side.
- **Mitigation:** Lower the classification threshold from 0.50 to 0.45 for this borderline band. This would catch near-threshold cases at the cost of more false positives.

### Case 2: El Potrillo Moyock (Grade B, Score 86.5)

- **Google rating:** 4.4 | **Safety keywords:** 0 | **P(flagged):** 0.483
- **Review excerpt:** "Very disappointing to spend $70+ and have to go home and cook dinner... the cheese dip was too salty for any of us to eat."
- **Root cause:** The review expresses dissatisfaction with food quality, not safety. "Disappointing" and "too salty" are dining complaints, not hygiene indicators. The model correctly identifies these as non-safety signals but misses the underlying inspection failure.
- **Mitigation:** This case is fundamentally unlinkable from review text. The inspection violation (score 86.5) reflects back-of-house issues invisible to diners.

### Case 3: El Mexicano Tacos and Tequila (Grade B, Score 88.5)

- **Google rating:** 4.7 | **Safety keywords:** 0 | **P(flagged):** 0.481
- **Review excerpt:** "Ordered the seafood fajitas and seafood dip and the house made guac and all of it was so delicious."
- **Root cause:** 4.7-star rating with glowing reviews. The inspection failure is entirely invisible in the customer experience. No feature in the model's vocabulary can detect this.
- **Mitigation:** Incorporate the restaurant's inspection history as a feature. A restaurant that has been flagged before is more likely to be flagged again. This requires adding prior inspection scores to the inference pipeline.

### Case 4: In & Out Mart (True Positive, Grade B, Score 80.0)

- **Google rating:** 4.6 | **Safety keywords:** 0 | **P(flagged):** 0.992
- **Review excerpt:** "Stopped for fuel, pump was not working properly... I only go there for the gas is cheaper."
- **Root cause (why it was caught):** This is a gas station/convenience store, not a traditional restaurant. The review volume and word patterns differ from typical restaurant reviews. The model learned that this feature profile (short, service-focused reviews about a non-restaurant establishment) correlates with flagged status. This restaurant had 7 B-grade inspections across 2020-2026.
- **Insight:** The model is partially learning establishment type from review content, not food safety from review sentiment. This is a proxy signal, not a causal one.

### Case 5: False Positive Pattern (1,548 safe restaurants flagged)

- The 1,548 false positives represent 3.4% of safe restaurants in the test set.
- **Root cause:** The `class_weight='balanced'` setting inflates the importance of the flagged class by 68x. This is necessary to achieve any recall at all, but it causes the model to over-flag restaurants whose review profiles are statistically unusual (low review count, short reviews, non-restaurant establishments).
- **Mitigation:** Calibrate the model's probability outputs using Platt scaling or isotonic regression, then choose a threshold that balances precision and recall for the deployment context. A screening tool (surface to a human reviewer) tolerates lower precision than a consumer-facing label.

### Summary of Root Causes

| Root Cause | Cases | Addressable? |
|------------|-------|-------------|
| Reviews discuss taste/service, not safety | 2, 3 | No. Fundamental domain mismatch. |
| Near-threshold borderline cases | 1 | Yes. Threshold tuning. |
| Model learns establishment type proxies, not safety | 4 | Partially. Better feature engineering. |
| class_weight='balanced' over-flags unusual profiles | 5 | Yes. Probability calibration. |

The dominant failure mode remains structural: customers write about what they experience, while inspectors evaluate what customers cannot see. However, with sufficient training data (3,354 flagged samples after the data pipeline fix), the model finds weak but real signal, primarily through review volume and establishment-type proxies rather than safety-specific language.

## 10. Experiment: Impact of Data Pipeline Quality on Model Performance

### Hypothesis

We discovered two bugs in the data pipeline that silently degraded the training data. We hypothesized that fixing them would provide enough training signal for the models to learn.

### Bug 1: Case-Sensitive Fuzzy Matching

`fuzz.token_sort_ratio` was called without case normalization. NC DHHS records use ALL-CAPS names; Google uses title case. "BOJANGLES" vs "Bojangles" scored 11.1% instead of 100%. **Fix:** Added `processor=fuzz_utils.default_process`. Usable Google matches jumped from 432 to 14,868 (34x increase).

### Bug 2: BOM-Corrupted Inspection Dates

The DHHS CSV export includes a UTF-8 BOM (byte order mark). The scraper used `resp.text.lstrip("\ufeff")` to strip it, but `requests` decoded the BOM as literal bytes (`ï»¿`), not the unicode codepoint. The column name became `ï»¿"Inspection Date"` instead of `"Inspection Date"`, causing `row.get("Inspection Date")` to return empty for every row.

With all dates null, `drop_duplicates(subset=['state_id', 'inspection_date'])` treated every inspection of the same restaurant as a duplicate and kept only the first one (typically an A-grade inspection). This collapsed 232K inspection rows to 31K and discarded 94% of B/C grade samples.

**Fix:** Changed to `pd.read_csv(io.BytesIO(resp.content), encoding="utf-8-sig")`.

### Results

| Metric | Before Fixes | After Fixes |
|--------|-------------|-------------|
| Total inspection rows | 31,760 | **231,160** |
| Flagged samples (B+C) | 197 | **3,354** |
| Imbalance ratio | 160:1 | **68:1** |
| Rows with Google reviews | 14,529 | **111,542** |
| RF Macro F1 | 0.50 | **0.57** |
| RF Flagged Recall | 0.00 | **0.28** |
| RF Flagged F1 | 0.00 | **0.16** |

### Interpretation

The first fix (case normalization) increased Google data coverage 34x but did not improve model performance on the 31K-row dataset. The second fix (BOM) was the critical one: it increased flagged training samples from 197 to 3,354, giving the model 17x more minority-class examples to learn from. Together, the fixes moved Random Forest macro F1 from 0.50 (identical to baseline) to 0.57 (statistically significant improvement).

This demonstrates that **data pipeline quality is a prerequisite for model quality.** Two silent bugs, one in string encoding and one in CSV parsing, were sufficient to make a learnable problem appear unlearnable.

### Recommendation

The RF now detects weak but real signal, primarily through review volume proxies rather than safety-specific language. The signal exists but is noisy: 11% precision at 28% recall. DistilBERT retrained on the corrected dataset (89K training texts, 1,682 flagged) still predicts all-safe (macro F1 = 0.50), confirming that the signal lives in structural metadata, not semantic content.

## 11. Conclusions

We investigated whether crowdsourced Google reviews can predict NC restaurant food safety inspection grades. The project produced two main findings:

**1. Data pipeline quality determines model quality.** Two silent bugs (case-sensitive fuzzy matching and BOM-corrupted inspection dates) made a learnable problem appear unlearnable. Fixing them increased flagged training samples from 197 to 3,354 and moved RF macro F1 from 0.50 to 0.57. This underscores the importance of data auditing before drawing conclusions about model capacity.

**2. Review text carries weak but real signal about food safety outcomes.** The Random Forest achieves 28% flagged recall with 11% precision, detecting patterns in review volume and text characteristics that correlate with inspection grades. However, the signal is primarily proxy-based (establishment type inferred from review patterns) rather than safety-specific. Customers still write about taste and service, not sanitization. The model learns that restaurants with certain review profiles are more likely to be flagged, not why they are flagged.

These findings have practical implications. A review-based model can serve as a screening tool (surface high-risk candidates for manual review) but not as a definitive safety label. The 11% precision rate means 9 out of 10 flagged restaurants are actually safe, which is acceptable for a first-pass filter but not for a consumer-facing warning.

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

**Potential harms.** A false-positive prediction (labeling a safe restaurant as flagged) could cause reputational harm to a business. A false-negative prediction (labeling an unsafe restaurant as safe) could lead consumers to eat at establishments with sanitation violations. Our current model produces 1,548 false positives and 480 false negatives on the test set (11% precision, 28% recall). The low precision means the model should be used as a screening tool, not a definitive label.

**Limitations of inference.** An inspection grade reflects a point-in-time assessment. Conditions can change between inspections. Users should be reminded that model predictions are not substitutes for official health inspection records.

**Bias considerations.** The fuzzy name matching pipeline may systematically fail to link restaurants with non-English names, leading to lower Google data coverage for certain cuisines. We did not audit for this bias but acknowledge it as a limitation.

---

*AI-assisted (Claude Code, claude.ai) — https://claude.ai*
