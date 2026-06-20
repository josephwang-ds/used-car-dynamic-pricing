# Technical Deep Dive: Used-Car Dynamic Pricing

## 1. Problem Definition

The project models used-car pricing as a supervised regression problem:

- **Input:** vehicle attributes, listing metadata, categorical identifiers, anonymous numerical features, and engineered market features.
- **Target:** listing price.
- **Output:** estimated fair price.
- **Business layer:** compare listing price against estimated fair price to classify listings as overpriced, underpriced, or fair.

The technical objective is to minimize validation error while keeping the workflow reproducible and explainable enough for a portfolio demo.

## 2. Dataset Shape

- Raw dataset: **150,000 rows x 31 columns**.
- Engineered dataset: **49 features** after feature engineering.
- Target variable: `price`.
- Public repo policy: raw data, processed data, serialized models, and output CSVs are excluded from GitHub.

## 3. Notebook Workflow

| Step | Notebook | Purpose |
|------|----------|---------|
| 1 | `01_EDA.ipynb` | Explore missing values, distributions, outliers, correlations, categorical patterns |
| 2 | `02_Feature_Engineering.ipynb` | Clean data and create model-ready pricing features |
| 3 | `03_XGBoost_Baseline.ipynb` | Train baseline gradient boosting model |
| 4 | `04_CatBoost_Optimization.ipynb` | Train categorical-aware boosting model |
| 5 | `05_LightGBM_Benchmark.ipynb` | Train fast histogram-based boosting model |
| 6 | `06_Ensemble.ipynb` | Blend model predictions and compare ensemble performance |
| 7 | `07_Pricing_Recommendation.ipynb` | Convert fair-price predictions into recommendation logic |

## 4. Feature Engineering

### Vehicle Age Features

Vehicle age captures depreciation:

- `car_age_days`
- `car_age_years`
- registration year/month/day
- listing year/month/day

Why it matters:

- Used-car value usually decreases with age.
- Age can interact with mileage and brand.
- Date-derived fields can capture market/time effects if listing date is meaningful.

### Usage Features

Mileage is not enough by itself. Annualized mileage provides a better wear proxy:

- `kilometer`
- `km_per_year`

Why it matters:

- A 10-year-old car with 80,000 km is different from a 3-year-old car with 80,000 km.
- Mileage intensity helps separate normal depreciation from heavy usage.

### Power Features

Power can be noisy and may contain outliers:

- capped power values
- `power_bin`
- model-power aggregate statistics

Why it matters:

- Power can proxy trim level, performance, and vehicle class.
- Binning reduces sensitivity to extreme values.

### Brand and Model Aggregates

Aggregate features encode market-level pricing patterns:

- `brand_price_mean`
- `brand_price_median`
- `brand_price_std`
- model-level price/power statistics

Why it matters:

- Brand and model strongly influence residual value.
- Aggregates give tree models useful prior information.

Technical caution:

- These features can leak target information if computed on the full dataset before splitting.
- In production, aggregate encoders should be fit only on training data and applied to validation/test data.

### Categorical Features

Important categorical fields include:

- `brand`
- `model`
- `bodyType`
- `fuelType`
- `gearbox`
- `notRepairedDamage`

Handling strategy:

- CatBoost can handle categorical structure well.
- XGBoost and LightGBM typically require numeric encodings or engineered aggregate features.

## 5. Model Design

### XGBoost

Role:

- Strong baseline for tabular regression.
- Useful benchmark for gradient boosting performance.

Strengths:

- Robust regularization.
- Strong performance on structured tabular data.
- Mature early stopping and feature importance tooling.

### CatBoost

Role:

- Best individual model in this workflow.

Strengths:

- Handles categorical-style data effectively.
- Ordered boosting can reduce target leakage risk in categorical encodings.
- Often strong when categorical identifiers such as brand/model are important.

Result:

- Validation MAE: **501.67 CNY**.

### LightGBM

Role:

- Fast benchmark model.

Strengths:

- Efficient histogram-based training.
- Often strong on large tabular datasets.
- Useful for quick iteration and benchmark comparison.

Result:

- Validation MAE: **591.00 CNY**.

### Weighted Ensemble

Formula:

```text
0.2 x XGBoost + 0.5 x CatBoost + 0.3 x LightGBM
```

Why it works:

- Different boosting implementations learn different trees and residual patterns.
- Blending can reduce variance and smooth model-specific errors.
- The strongest model, CatBoost, receives the largest weight.

Result:

- Validation MAE: **496.83 CNY**.

## 6. Evaluation

### Metric

Primary metric: **Mean Absolute Error (MAE)**.

Why MAE:

- Easy to interpret in CNY.
- Directly explains average absolute pricing error.
- Less dominated by large outliers than RMSE.
- Suitable for business communication.

### Validation Split

All models use the same holdout split:

```text
test_size = 0.2
random_state = 42
```

Why this matters:

- Ensures model comparison is fair.
- Avoids comparing models on different validation samples.
- Keeps notebook results reproducible.

### Results

| Model | Validation MAE |
|-------|---------------:|
| XGBoost | 514.23 |
| CatBoost | 501.67 |
| LightGBM | 591.00 |
| Weighted Ensemble | 496.83 |

Interpretation:

- CatBoost is the strongest single model.
- Weighted ensemble is the best overall model.
- The ensemble improves MAE by 4.84 CNY compared with CatBoost and 17.40 CNY compared with XGBoost.

## 7. Recommendation Layer

Prediction alone is not the final product. The project converts fair-price predictions into pricing actions.

### Pricing Gap

```text
price_gap = current_listing_price - fair_price
price_gap_pct = price_gap / fair_price
```

### Status Rules

| Status | Rule |
|--------|------|
| Overpriced | Listing price more than 10% above fair price |
| Underpriced | Listing price more than 10% below fair price |
| Fair Price | Listing price within +/-10% of fair price |

### Recommendation Rules

```text
if Overpriced:
    recommended_price = fair_price x 1.03
elif Underpriced:
    recommended_price = fair_price x 0.97
else:
    recommended_price = current_listing_price
```

Business interpretation:

- Overpriced listings may need discounting to improve conversion.
- Underpriced listings may have margin upside.
- Fair-price listings can remain unchanged while market feedback is monitored.

## 8. Streamlit App Architecture

The app is intentionally self-contained:

- no raw data import
- no processed feature import
- no model artifact loading
- no output CSV dependency

Why:

- Public GitHub and Streamlit Cloud deployment should be lightweight.
- Data and model files should not be exposed.
- The demo remains runnable even when private artifacts are absent.

Tabs:

- Overview
- Pricing Simulator
- Model Results
- Business Insights
- Methodology

## 9. Explainability

Current explainability:

- XGBoost feature importance
- CatBoost feature importance
- LightGBM feature importance
- supporting EDA figures

Strength:

- Good for portfolio-level model interpretation.
- Helps explain which feature groups drive pricing.

Limitation:

- Global feature importance does not explain individual listing predictions.

Production upgrade:

- Add SHAP summary plot.
- Add SHAP waterfall/force plot for single-listing explanations.
- Show seller-facing explanations such as age, mileage, brand, power, and damage status effects.

## 10. Data Leakage Risks

Main leakage risk:

- Aggregate price features computed using validation/test rows.

Mitigation:

- Split data before computing target-based aggregates.
- Fit aggregate encoders only on training data.
- Apply learned mappings to validation/test data.
- Use cross-fold target encoding for stronger leakage protection.

Other risks:

- Using future listing information not available at prediction time.
- Accidentally using recommendation outputs as training inputs.
- Evaluating on duplicated or near-duplicated listings.

## 11. Production Architecture

A production version could use:

```text
raw listing feed
  -> validation and cleaning
  -> feature pipeline
  -> model inference
  -> pricing rule layer
  -> recommendation API / dashboard
  -> monitoring and feedback loop
```

Components:

- data ingestion
- feature store or feature pipeline
- model registry
- batch scoring or real-time API
- Streamlit/internal dashboard
- monitoring jobs
- retraining pipeline

## 12. Monitoring

Model monitoring:

- prediction distribution drift
- feature distribution drift
- MAE / MAPE on delayed actual prices
- residual bias by brand/model/price band

Business monitoring:

- seller acceptance rate
- listing conversion rate
- inquiry rate
- time-to-sale
- margin impact
- share of listings flagged overpriced/underpriced

## 13. Experimentation Plan

A/B test design:

- Control: current pricing workflow.
- Treatment: model-assisted pricing recommendations.

Primary metrics:

- conversion rate
- time-to-sale
- seller acceptance rate

Guardrail metrics:

- gross margin
- cancellation/refund rate
- user complaints
- segment-level fairness across vehicle types

## 14. Known Limitations

- Public app is rule-based rather than true model inference.
- Data and model artifacts are excluded from GitHub.
- The current ensemble weights are manually selected rather than optimized by cross-validation.
- Feature importance is global, not listing-level.
- Production impact is not measured through live experiments yet.

## 15. Next Technical Improvements

- Add SHAP explanations.
- Use cross-validation for more robust model comparison.
- Optimize ensemble weights with out-of-fold predictions.
- Add model artifact loading in a private deployment.
- Add automated data validation.
- Add CI checks for app startup and dependency safety.
- Add monitoring and retraining scripts.

## 16. Design Rationale

Key technical decisions:

- MAE was selected because it is interpretable in CNY.
- CatBoost likely performed well because categorical identifiers matter in vehicle pricing.
- Ensemble improved performance by combining models with different error patterns.
- Target-based aggregate features are powerful but require leakage controls.
- Streamlit demo is intentionally decoupled from private artifacts.
- A production pricing system needs business metric validation, not only offline MAE.

## 17. What This Project Demonstrates

- Ability to structure an ML project end to end.
- Ability to compare models fairly.
- Ability to think about leakage and validation.
- Ability to connect model output to business decisions.
- Ability to build a public-facing demo with deployment constraints.
- Ability to communicate both technical and business value.
