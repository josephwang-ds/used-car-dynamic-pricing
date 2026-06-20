# Dynamic Pricing Optimization for Used Cars

End-to-end data science project for used-car **dynamic pricing**: exploratory analysis, feature engineering, gradient boosting benchmarks, ensemble modeling, and a Streamlit portfolio demo for pricing recommendations.

The Streamlit app wraps this pricing model inside a **CPO Retail Command Center** — a single decision loop that connects the model to dealer-network analytics: a National Executive Overview, separate Sales and After-sales KPI systems, a 50-dealer Dealer 360 with a composite Dealer Score, and a CPO pricing & inventory action layer, with full English / 中文 switching.

> The dealer network is **100% synthetic and deterministic**. No real company data, dealer names, customers, VINs, targets, or incentives are used. The dealer layer demonstrates a metric system and decision flow designed from premium automotive dealer-analytics experience; Dealer Score weights and KPI formulas are transparent portfolio assumptions, not company definitions. See the app's About tab.

The public app is intentionally lightweight. Raw data, processed data, model artifacts, and prediction outputs are excluded from GitHub, so Streamlit uses a transparent rule-based simulator while the notebooks preserve the full machine-learning workflow.

## Live Demo

- Streamlit demo: https://josephwang-dynamic-pricing.streamlit.app/
- GitHub repository: https://github.com/josephwang-ds/used-car-dynamic-pricing

## Portfolio Highlights

- Built a complete pricing workflow from EDA and feature engineering to model benchmarking, ensemble validation, and business recommendation logic.
- Benchmarked **XGBoost**, **CatBoost**, and **LightGBM** on the same validation split, then improved performance with a weighted ensemble.
- Achieved best validation MAE of **496.83 CNY**, outperforming each individual benchmark model.
- Converted model output into business-facing actions: fair price, overpricing flag, underpricing flag, and recommended listing price.
- Deployed a public Streamlit demo that stays runnable without raw data or model artifacts by using transparent rule-based pricing logic.

## Tech Stack

| Area | Tools |
|------|-------|
| Data analysis | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Modeling | Scikit-learn, XGBoost, CatBoost, LightGBM |
| App | Streamlit |
| Workflow | Jupyter notebooks, Git, GitHub |

## Business Problem

Used-car marketplaces need to answer three practical questions for every listing:

1. **What is a fair market price?** Estimate vehicle value from attributes and historical listing patterns.
2. **Is the current listing overpriced or underpriced?** Compare asking price to a fair-price estimate.
3. **What price should be recommended?** Translate model output into seller and marketplace actions.

This project builds a reproducible workflow from raw listings to model predictions and pricing recommendations.

## Dataset

- **Source:** Used car training data (`used_car_train_20200313.csv`)
- **Size:** 150,000 rows x 31 columns raw; 49 columns after feature engineering
- **Target:** `price`
- **Format:** Space-separated CSV; some fields use `-` for missing values

Place raw files under `data/raw/`. Processed features are written to `data/processed/train_fe.csv` by the feature engineering notebook. These files are intentionally ignored by Git.

## Project Architecture

```text
used-car-dynamic-pricing/
├── app/
│   └── streamlit_app.py        # Rule-based portfolio demo
├── data/
│   ├── raw/                    # Raw CSVs, not committed
│   └── processed/              # Feature tables, not committed
├── models/                     # Saved model files, not committed
├── notebooks/                  # 01-07 reproducible DS workflow
├── outputs/                    # Predictions/recommendations, not committed
├── reports/
│   ├── figures/                # Committed analysis/model figures
│   └── screenshots/            # Streamlit screenshots after deployment
├── scripts/                    # Optional helpers
├── .streamlit/config.toml      # Streamlit Cloud configuration
├── README.md
└── requirements.txt
```

## Notebook Execution Order

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `01_EDA.ipynb` | Exploratory data analysis, missing values, target and feature distributions |
| 2 | `02_Feature_Engineering.ipynb` | Cleaning, date/business features, brand/model aggregations |
| 3 | `03_XGBoost_Baseline.ipynb` | Gradient boosting baseline with MAE and early stopping |
| 4 | `04_CatBoost_Optimization.ipynb` | CatBoost with categorical handling and optimization |
| 5 | `05_LightGBM_Benchmark.ipynb` | LightGBM benchmark on the same validation split |
| 6 | `06_Ensemble.ipynb` | Equal-weight and weighted ensemble of model predictions |
| 7 | `07_Pricing_Recommendation.ipynb` | Fair price, over/underpriced flags, and recommended prices |

## Feature Engineering

Key engineered features include:

| Feature | Description |
|---------|-------------|
| `car_age_days` | Days between registration and listing date |
| `car_age_years` | Vehicle age in years |
| `km_per_year` | Annualized mileage as a usage proxy |
| `power_bin` | Binned engine power |
| `brand_price_mean` | Mean price by brand, computed on training data |
| `brand_price_median` | Median price by brand |
| `brand_price_std` | Price dispersion by brand |

Additional fields include calendar parts, model-power statistics, anonymous `v_0`-`v_14` features, and categorical encodings used by the benchmark models.

## Models

- **XGBoost:** tree boosting baseline
- **CatBoost:** categorical-aware gradient boosting
- **LightGBM:** leaf-wise histogram boosting benchmark
- **Weighted Ensemble:** `0.2 x XGBoost + 0.5 x CatBoost + 0.3 x LightGBM`

All models use the same holdout split: `test_size=0.2`, `random_state=42`.

## Model Results

Validation MAE on the shared 20% holdout:

| Model | Validation MAE |
|-------|---------------:|
| XGBoost | 514.23 |
| CatBoost | 501.67 |
| LightGBM | 591.00 |
| **Weighted Ensemble** | **496.83** |

CatBoost is the best single model, and the weighted ensemble achieves the lowest MAE on this split.

## Business Interpretation

The model estimates a fair market price for each listing. The pricing layer then compares the seller's current asking price to that fair-price estimate:

- **Overpriced:** listing price is more than 10% above fair price
- **Underpriced:** listing price is more than 10% below fair price
- **Fair Price:** listing is within the +/-10% band

A validation MAE of **496.83 CNY** means the best model is typically within a few hundred yuan of the observed price on the holdout set. In a marketplace setting, this is most useful as a decision-support signal: flag materially mispriced listings, guide seller pricing, and prioritize operational review.

## Limitations

- The public Streamlit app uses rule-based logic because raw data and trained models are excluded from GitHub.
- Validation results come from the notebook workflow, not from live model inference inside Streamlit.
- A production version would require model artifact storage, data refresh jobs, drift monitoring, and seller/platform policy constraints.

## Streamlit Demo

The Streamlit app is a bilingual portfolio demo that combines the original used-car pricing workflow with a synthetic premium-automotive retail management layer. It is designed as one decision loop: detect a network gap, locate the dealer and business line, diagnose the driver, and turn the finding into a CPO pricing or inventory action.

Run locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app includes:

- **Executive Overview:** five Sales and five After-sales headline KPIs, trend context, and the management storyline.
- **Sales KPI:** NC, CPO, and Vans target achievement tracked separately, plus conversion, discount, model mix, and inventory diagnostics.
- **After-sales KPI:** revenue, gross profit, repair orders, retention, workshop utilization, technician efficiency, quality, and customer experience.
- **Dealer 360:** a synthetic 50-dealer network, transparent composite Dealer Score, ranking, and dealer-level root-cause drill-down.
- **CPO Pricing & Inventory:** single-vehicle price simulation plus a dealer action list for repricing, promotion, and stock transfer.
- **AI Copilot:** live, bilingual, KPI-grounded management recommendations through the OpenAI Responses API, with deterministic offline fallback answers.
- **About & Method:** metric definitions, model results, architecture, limitations, and the synthetic-data disclosure.

All dealer-network figures are deterministic synthetic data. The formulas and Dealer Score weights are proposed portfolio definitions rather than official former-employer definitions.

### Enable the LLM Copilot

In Streamlit Community Cloud, open **App settings → Secrets** and add:

```toml
OPENAI_API_KEY = "your-api-key"
OPENAI_MODEL = "gpt-4.1-mini"
```

The key is read only at runtime through `st.secrets`; never add it to Git. When the key is missing or the API is unavailable, the app keeps working with deterministic KPI-based fallback answers. Only aggregated synthetic demo metrics are sent to the LLM.

## Screenshots

Place Streamlit demo screenshots under `reports/screenshots/` after deployment.

Suggested screenshots:

- Executive Overview
- Sales KPI and After-sales KPI
- Dealer 360 drill-down for `DLR-E07`
- CPO Pricing & Inventory action list
- Chinese-language view

## Setup for Full Notebook Workflow

```bash
git clone https://github.com/josephwang-ds/used-car-dynamic-pricing.git
cd used-car-dynamic-pricing
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add training data to `data/raw/`, then run notebooks in order from `notebooks/`.

On macOS, if XGBoost fails to load OpenMP, install `libomp` with `brew install libomp` or use a conda environment with `xgboost` and `libomp` preinstalled.

## Deployment Notes

For Streamlit Cloud:

1. Push this repository to GitHub.
2. Create a Streamlit Cloud app pointing to `app/streamlit_app.py`.
3. Keep `requirements.txt` deployment-safe and do not upload data/model artifacts.
4. Add screenshots to `reports/screenshots/` after deployment.

## Data and Artifact Policy

Do not commit:

- Raw data files under `data/raw/`
- Processed feature files under `data/processed/`
- Serialized models under `models/`
- Prediction or recommendation CSVs under `outputs/`

The `.gitkeep` files preserve directory structure without exposing data artifacts.

## Future Work

- Add SHAP-based local explanations for individual listings
- Add optimized stacking or out-of-fold ensemble weights
- Add scheduled retraining and monitoring pipeline
- Add an API endpoint for batch pricing checks
- Replace synthetic dealer facts with an authorized warehouse model and governed KPI dictionary

## License

For portfolio and educational use. Verify the data license before commercial deployment.
