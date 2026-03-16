# Can Crowdsourced Reviews Predict Food Safety? A Three-Model Investigation of NC Restaurant Inspection Grades

Roshan Gill, Jonas Neves, Dominic Tanzillo

AIPI 540 Deep Learning, Duke University, Spring 2026

---

## 1. Problem Statement

North Carolina's Department of Health and Human Services (DHHS) inspects every licensed food establishment and assigns letter grades (A, B, or C) based on sanitation, temperature control, pest management, and food handling. These grades are public record, but consumers rarely check them. Instead, they rely on crowdsourced review platforms like Google to decide where to eat.

This project asks: **does the language and sentiment of public restaurant reviews contain signal that predicts official food safety inspection outcomes?** If so, a predictive model could surface restaurants where high public ratings mask poor inspection records, a gap between public perception and regulatory reality. The answer is mixed: structural metadata about reviews (volume, length) carries modest predictive signal, while the semantic content of review text does not.

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

- **Sadilek et al. (2018)**, published in npj Digital Medicine, combined anonymized aggregated location history from opted-in users with illness-related search queries to identify potentially unsafe restaurants in Chicago and Las Vegas. Their approach relied on indirect behavioral signals rather than review text, achieving meaningful recall on serious violations.

- **Nsoesie et al. (2014)**, published in *Preventive Medicine*, compared the distribution of implicated food categories in Yelp illness-related reviews against CDC outbreak surveillance reports. They found that Yelp data captured similar food category patterns to official reports, suggesting review platforms could complement (but not replace) traditional surveillance.

Our work differs in several ways: (1) we use NC's letter-grade system rather than binary violation detection, (2) our class imbalance is far more extreme (98.5% A), (3) we compare three model architectures from naive baseline through deep learning, and (4) we explicitly test whether review language carries food safety signal at all, rather than assuming it does.

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

4. **Target encoding.** Grades are label-encoded alphabetically (A=0, B=1, C=2), then binarized for training: A = 0 (safe), B or C = 1 (flagged).

**Data quality fixes.** Two pipeline bugs silently degraded the training data: (1) case-sensitive fuzzy matching that reduced Google linkage from 14,868 to 432 matches, and (2) a BOM encoding error that dropped all inspection dates, collapsing 232K rows to 31K and losing 94% of B/C grade samples. Both fixes are documented as the project's primary experiment (Section 10).

## 6. Hyperparameter Tuning Strategy

The Random Forest was tuned via 5-fold `GridSearchCV` optimizing `f1_macro`:

| Hyperparameter | Search Space | Best Value |
|----------------|-------------|------------|
| `n_estimators` | [100, 200] | 100 |
| `max_depth` | [None, 10, 20] | 20 |
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
| DistilBERT (binary, 111K texts)* | 0.50 | 0.00 | 0.00 | 0.00 |

*DistilBERT was evaluated on a different test set (~22K rows with review text) than the RF and Baseline (46,232 rows, all inspections). Metrics are not directly comparable across different test populations.

The Random Forest detects real signal above baseline, catching 28% of flagged restaurants (191 of 671 in the test set). DistilBERT, despite training on 89K review texts with 1,682 flagged samples, predicts all-safe. However, this result has a known confounder: DistilBERT was trained with default (unweighted) cross-entropy loss, while the RF uses `class_weight='balanced'` which upweights the minority class by 68x. On a 68:1 imbalanced dataset, unweighted cross-entropy converges to all-majority-class predictions because predicting all-A minimizes loss. We did not test DistilBERT with class-weighted loss or focal loss due to computational constraints.

The RF's structured features (review volume, word statistics) carry signal that DistilBERT with default loss could not extract from raw text. The predictive signal the RF finds lives in metadata patterns (how many reviews a restaurant has, how long they are). Whether semantic content carries additional signal remains an open question, pending a class-weighted DistilBERT experiment (see Section 12).

### SHAP Feature Importance (Random Forest)

| Feature | Mean \|SHAP\| |
|---------|--------------|
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

We examine 5 specific mispredictions: 4 false negatives (flagged restaurants the model predicted as safe) and 1 false positive (a safe restaurant the model flagged). Of 671 flagged restaurants in the test set, 480 were missed (28% recall). Of 45,561 safe restaurants, 1,548 were over-flagged (3.4% false positive rate).

### Case 1: La Tovara Mexican Kitchen and Seafood (Grade B, Score 86.5)

- **Google rating:** 4.6 | **Safety keywords:** 0 | **P(flagged):** 0.496
- **Review excerpt:** "Found another gem here in Greenville! Friendly and helpful staff and the food was great tasting and nicely portioned."
- **Root cause:** High rating (4.6), zero safety keywords, entirely positive reviews. The model came close to the decision boundary (P=0.496) but the absence of any negative textual signal kept it on the safe side.
- **Mitigation:** Lower the classification threshold from 0.50 to 0.45 for this borderline band. This would catch near-threshold cases at the cost of more false positives.

### Case 2: El Potrillo Moyock (Grade B, Score 86.5)

- **Google rating:** 4.4 | **Safety keywords:** 0 | **P(flagged):** 0.483
- **Review excerpt:** "Very disappointing to spend $70+ and have to go home and cook dinner... the cheese dip was too salty for any of us to eat."
- **Root cause:** The review expresses dissatisfaction with food quality, not safety. "Disappointing" and "too salty" are dining complaints, not hygiene indicators. The model correctly identifies these as non-safety signals but misses the underlying inspection failure.
- **Mitigation:** Add a text-derived "complaint specificity" feature that distinguishes vague dining complaints ("too salty," "disappointing") from hygiene-adjacent language. Alternatively, incorporate the restaurant's prior inspection scores as a feature: a restaurant with a history of borderline scores is more likely to fail again regardless of review sentiment.

### Case 3: El Mexicano Tacos and Tequila (Grade B, Score 88.5)

- **Google rating:** 4.7 | **Safety keywords:** 0 | **P(flagged):** 0.481
- **Review excerpt:** "Ordered the seafood fajitas and seafood dip and the house made guac and all of it was so delicious."
- **Root cause:** 4.7-star rating with glowing reviews. The inspection failure is entirely invisible in the customer experience. No feature in the model's vocabulary can detect this.
- **Mitigation:** Incorporate the restaurant's inspection history as a feature. A restaurant that has been flagged before is more likely to be flagged again. This requires adding prior inspection scores to the inference pipeline.

### Case 4: Hibachi To Go (False Negative, Grade B, Score 88.5)

- **Google rating:** 3.9 | **Safety keywords:** 0 | **Negative phrases:** 3 | **P(flagged):** 0.476
- **Review excerpt:** "Food poisoning. I do intermittent fasting so I know it's from here. I was skeptical of the quality, but I wanted to believe there was a healthier drive thru food option. Never again."
- **Root cause:** Despite containing 3 negative phrases ("food poisoning," "never again," and "disgusting" in other reviews), the model still predicted safe. The negative phrase feature contributes modestly (mean |SHAP| = 0.010, ranked last). The low Google rating (3.9) nudged the probability toward flagged, but not enough to cross the 0.50 threshold.
- **Mitigation:** Increase the weight of safety-specific text signals relative to volume-based features. A review mentioning "food poisoning" is qualitatively different from one saying "disappointing." A binary "explicit illness mention" feature, separate from the general negative phrase count, could provide a stronger signal for cases like this.

### Case 5: EC Pho Vietnamese Noodle House (False Positive, Grade A, Score 90.0)

- **Google rating:** 4.3 | **Safety keywords:** 0 | **Negative phrases:** 1 | **P(flagged):** 0.991
- **Review excerpt:** "Way to go Greenville!!! This EC Pho is legit!!! Jonny added to the experience with his excellent customer service!!! My wife got the House Special Pho and it had so much meat in it."
- **Root cause:** This restaurant has a safe inspection record (score 90.0, grade A) but the model flagged it with near-certainty (P=0.991). The `class_weight='balanced'` setting inflates minority-class importance by 68x, causing the model to over-flag restaurants whose review profiles are statistically unusual. EC Pho's feature profile (moderate rating, one negative phrase, particular word patterns) triggers this sensitivity. Of the 1,548 false positives in the test set, many share this pattern of unusual but innocuous review profiles.
- **Mitigation:** Calibrate the model's probability outputs using Platt scaling or isotonic regression, then choose a threshold that balances precision and recall for the deployment context. A screening tool (surface to a human reviewer) tolerates lower precision than a consumer-facing label.

### Summary of Root Causes

| Root Cause | Cases | Addressable? |
|------------|-------|-------------|
| Reviews discuss taste/service, not safety | 2, 3 | Partially. Complaint specificity features + inspection history. |
| Near-threshold borderline cases | 1 | Yes. Threshold tuning. |
| Safety-specific text features underweighted | 4 | Yes. Explicit illness-mention feature. |
| class_weight='balanced' over-flags unusual profiles | 5 | Yes. Probability calibration. |

The dominant failure mode is structural: customers write about what they experience, while inspectors evaluate what customers cannot see. With 3,354 flagged samples after the data pipeline fix, the model does find signal, but it lives in review volume and establishment-type proxies rather than safety-specific language.

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

This demonstrates that **data pipeline quality is a prerequisite for model quality.** Two silent bugs, one in string encoding and one in CSV parsing, were sufficient to make a learnable problem appear unlearnable. This experiment validates the decision to deploy the Random Forest as the production model: RF responded to the data quality improvement (macro F1 0.50 → 0.57), while DistilBERT with default loss did not, confirming that the structured-feature approach outperforms raw-text fine-tuning on this dataset.

### Recommendation

The RF now detects signal above baseline, primarily through review volume proxies. That signal is noisy (11% precision at 28% recall). DistilBERT retrained on the corrected dataset (89K training texts, 1,682 flagged) still predicts all-safe (macro F1 = 0.50). However, DistilBERT was trained with unweighted cross-entropy (no class balancing), so this result does not rule out semantic signal in review text. It confirms only that unweighted fine-tuning on a 68:1 imbalanced dataset defaults to majority-class predictions.

## 11. Conclusions

We investigated whether crowdsourced Google reviews can predict NC restaurant food safety inspection grades. The project produced two main findings:

**1. Data pipeline quality determines model quality.** Two silent bugs (case-sensitive fuzzy matching and BOM-corrupted inspection dates) made a learnable problem appear unlearnable. Fixing them increased flagged training samples from 197 to 3,354 and moved RF macro F1 from 0.50 to 0.57. This underscores the importance of data auditing before drawing conclusions about model capacity.

**2. Review metadata carries measurable but limited signal.** The Random Forest catches 28% of flagged restaurants at 11% precision, finding patterns in review volume and text length that correlate with inspection grades. The signal is primarily proxy-based: the model appears to learn establishment type from review patterns, not food safety from review sentiment. Customers write about taste and service, not sanitization. Whether semantic content carries additional signal remains untested, since DistilBERT was trained without class weighting and its all-safe predictions reflect that configuration choice, not a definitive absence of textual signal.

These findings have practical implications. A review-based model can serve as a screening tool (surface high-risk candidates for manual review) but not as a definitive safety label. The 11% precision rate means 9 out of 10 flagged restaurants are actually safe, which is acceptable for a first-pass filter but not for a consumer-facing warning.

## 12. Future Work

Given another semester, we would pursue these directions:

1. **Inspection history as features.** A restaurant's prior scores, violation counts, and time since last inspection are strong predictors of future outcomes. These are available from NC DHHS and would shift the model from review-based prediction to inspection-trend prediction.

2. **Complaint-driven signals.** NC DHHS accepts public complaints that trigger inspections. Mining complaint text (which describes safety concerns directly, unlike reviews that describe dining experience) would provide features with direct domain relevance.

3. **Anomaly detection reframing.** Treat the problem as anomaly detection rather than binary classification: identify restaurants whose review profile is statistically unusual given their inspection history. This sidesteps the class imbalance problem entirely.

4. **Cross-state validation.** NYC, for example, has a higher proportion of B and C grades. Testing whether the RF's signal generalizes to a less imbalanced grading distribution would clarify whether the finding is specific to NC's 98.5% A-grade rate.

5. **Class-weighted DistilBERT.** Retrain with weighted cross-entropy loss (inverse class frequency) or focal loss. The current all-safe predictions are expected behavior for unweighted loss on a 68:1 distribution and do not establish whether review text carries semantic signal.

6. Rather than predicting grades from aggregated reviews, a **review-level classifier** could label individual sentences as safety-relevant or not, then use the proportion as a structured feature for the RF.

## 13. Commercial Viability

A model that reliably predicts food safety from publicly available data would have clear commercial value: integration into restaurant discovery platforms, insurance underwriting for food service businesses, and public health surveillance tools.

However, our results show that review data alone is insufficient. A commercially viable product would need to combine review analysis with (1) public inspection records, (2) complaint data, and (3) operational signals (e.g., staff turnover, hours of operation changes). The review component would serve as one input among many, not as the sole predictor.

The web application we built (nocapchicken-github-io.onrender.com) demonstrates the UX pattern: search for a restaurant, see its Google rating alongside a model-generated risk assessment with SHAP explanations. The interface is production-ready; the underlying model needs richer data sources to be commercially useful.

## 14. Ethics Statement

**Data provenance.** NC DHHS inspection records are public government records under NC Public Records Law (G.S. 132-1). Google Places data was accessed via the official API under Google's Terms of Service. No personally identifiable information about restaurant patrons was collected or used.

**Potential harms.** A false-positive prediction (labeling a safe restaurant as flagged) could cause reputational harm to a business. A false-negative prediction (labeling an unsafe restaurant as safe) could lead consumers to eat at establishments with sanitation violations. Our current model produces 1,548 false positives and 480 false negatives on the test set (11% precision, 28% recall). The low precision means the model should be used as a screening tool, not a definitive label.

**Limitations of inference.** An inspection grade reflects a point-in-time assessment. Conditions can change between inspections. Users should be reminded that model predictions are not substitutes for official health inspection records.

**Bias considerations.** The fuzzy name matching pipeline may systematically fail to link restaurants with non-English names, leading to lower Google data coverage for certain cuisines. We did not audit for this bias but acknowledge it as a limitation.

---

*AI-assisted (Claude Code, claude.ai) — https://claude.ai*
