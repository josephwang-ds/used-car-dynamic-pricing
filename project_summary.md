# Used-Car Dynamic Pricing Project Summary

## Project Links

- Live demo: https://josephwang-dynamic-pricing.streamlit.app/
- GitHub: https://github.com/josephwang-ds/used-car-dynamic-pricing

## One-Line Summary

Built an end-to-end used-car dynamic pricing workflow on 150K listings, benchmarked XGBoost/CatBoost/LightGBM, achieved **496.83 CNY validation MAE** with a weighted ensemble, and deployed a Streamlit demo that translates fair-price estimates into pricing recommendations.

## 中文一句话总结

基于 15 万条二手车挂牌数据完成从 EDA、特征工程、XGBoost/CatBoost/LightGBM 建模到加权集成和 Streamlit 部署的完整动态定价项目，最佳验证集 MAE 为 **496.83 元**，并将模型结果转化为公允价、高估/低估判断和推荐挂牌价。

## Short Portfolio Summary

This project demonstrates a complete data science workflow for marketplace pricing. It starts with used-car listing data, builds pricing-relevant features, compares gradient boosting models, improves performance with an ensemble, and converts the prediction output into business-facing pricing actions. The public Streamlit demo uses rule-based logic so it can run without exposing raw data or model artifacts, while the notebooks preserve the full ML workflow and validation results.

## 中文作品集摘要

这个项目展示了一个完整的 marketplace 定价数据科学流程：从二手车挂牌数据出发，完成 EDA、特征工程、XGBoost/CatBoost/LightGBM 模型对比、加权集成，并把预测结果转化为业务可用的定价建议。公开 Streamlit demo 使用规则模拟逻辑，保证无需上传原始数据和模型文件也能运行；完整机器学习流程和结果保留在 01-07 notebooks 中。

## Technical Summary

- Dataset: 150,000 used-car listings, 31 raw columns, 49 engineered features.
- Target: `price`.
- Core features: vehicle age, annualized mileage, power bins, brand-level price aggregates, model-level aggregates, date features, categorical encodings, anonymous numerical features.
- Models: XGBoost, CatBoost, LightGBM, weighted ensemble.
- Best result: weighted ensemble with **496.83 CNY validation MAE**.
- Deployment: Streamlit Cloud app using rule-based demo logic for public reproducibility.

## Business Summary

The project solves a common marketplace problem: sellers and platform operators need a consistent way to identify whether a used-car listing is fairly priced. The model estimates fair market value, then the recommendation layer compares the current listing price against that estimate. Listings more than 10% above fair price are flagged as overpriced, listings more than 10% below fair price are flagged as underpriced, and listings within the band are treated as fair. This makes the model output directly usable for pricing review, seller guidance, and marketplace quality control.

## Resume Version

Used-Car Dynamic Pricing Optimization: built an end-to-end Python ML workflow on **150K used-car listings**, engineered pricing features, benchmarked **XGBoost/CatBoost/LightGBM**, and deployed a Streamlit portfolio demo; weighted ensemble achieved **496.83 CNY validation MAE** and converted predictions into fair-price, overpricing, and recommendation actions.

## Resume Bullets

- Built an end-to-end used-car dynamic pricing project on **150K listings**, covering EDA, feature engineering, model benchmarking, ensembling, and Streamlit deployment.
- Engineered vehicle age, mileage intensity, power bins, brand/model aggregate price features, and pricing-gap signals to support fair-market-value estimation.
- Benchmarked **XGBoost, CatBoost, and LightGBM** on a shared holdout split; weighted ensemble achieved the best validation result with **496.83 CNY MAE**.
- Translated model predictions into business actions: fair price estimate, over/underpriced classification, recommended listing price, and seller-facing interpretation.
- Published a public Streamlit demo using rule-based pricing logic so the app remains runnable without exposing raw data, processed data, or model artifacts.

## LinkedIn / Portfolio Post Version

I built and deployed a used-car dynamic pricing project as an end-to-end data science portfolio case. The workflow covers EDA, feature engineering, XGBoost/CatBoost/LightGBM benchmarks, weighted ensembling, and a business recommendation layer that turns fair-price predictions into overpricing/underpricing flags and suggested listing prices. The best validation result was **496.83 CNY MAE** from the weighted ensemble. I also deployed a Streamlit demo that stays public and reproducible without exposing raw data or trained model files.

## Interview 30-Second Version

This is an end-to-end used-car dynamic pricing project. I started with 150K vehicle listings, performed EDA and feature engineering, then benchmarked XGBoost, CatBoost, and LightGBM on a shared validation split. The weighted ensemble performed best with a validation MAE of 496.83 CNY. I then converted the model output into a business recommendation layer: fair price, over/underpriced status, and recommended listing price. The public Streamlit demo uses transparent rule-based pricing logic because raw data and model artifacts are excluded from GitHub.

## Interview 90-Second Version

The business problem is that a used-car marketplace needs a consistent way to decide whether each listing is fairly priced. I framed the task as supervised price prediction, then added a recommendation layer on top of the prediction.

The workflow has seven notebooks. The first notebook explores missingness, price distribution, outliers, and relationships between vehicle attributes and price. The second notebook creates features such as car age, annualized mileage, power bins, brand-level price statistics, and model-level aggregates. Then I trained XGBoost, CatBoost, and LightGBM using the same 80/20 validation split so the comparison was fair.

CatBoost was the best single model, with 501.67 CNY MAE. The weighted ensemble improved that to 496.83 CNY MAE. After that, I built a business layer: if listing price is more than 10% above fair price, it is flagged as overpriced; if more than 10% below, underpriced; otherwise fair. The Streamlit app demonstrates that workflow with a rule-based simulator, since I intentionally do not publish raw data or model artifacts in GitHub.

## STAR Version

**Situation:** Used-car listings can be mispriced because sellers and platforms need to compare vehicle attributes, mileage, age, brand, and market conditions at scale.

**Task:** Build a reproducible pricing workflow that estimates fair value and turns model output into practical pricing recommendations.

**Action:** I cleaned and explored 150K listings, engineered vehicle age and usage features, created brand/model aggregate price features, benchmarked XGBoost/CatBoost/LightGBM, and combined them into a weighted ensemble. I also built a Streamlit demo with tabs for overview, pricing simulator, model results, business insights, and methodology.

**Result:** The weighted ensemble achieved the best validation MAE at **496.83 CNY**. The project now has a public demo, clean README, deployment config, and a business-facing recommendation layer.

## Methodology Summary

1. EDA: analyzed target distribution, missing values, outliers, brand/model distributions, correlations, and feature-price relationships.
2. Feature engineering: created vehicle age, mileage intensity, date parts, power bins, brand aggregates, model aggregates, and cleaned categorical/numerical inputs.
3. Modeling: trained XGBoost, CatBoost, and LightGBM on a shared validation split.
4. Ensemble: combined model predictions with weighted blending.
5. Recommendation layer: converted fair-price estimates into overpricing/underpricing/fair-price status and suggested listing prices.
6. Deployment: built a public Streamlit demo with rule-based pricing logic and clear disclaimer.

## Model Results Summary

| Model | Validation MAE |
|-------|---------------:|
| XGBoost | 514.23 |
| CatBoost | 501.67 |
| LightGBM | 591.00 |
| Weighted Ensemble | 496.83 |

CatBoost was the best individual model, and the weighted ensemble produced the strongest validation result.

## App Summary

The Streamlit app is a portfolio demo with five tabs:

- Overview: project metrics, workflow, and links.
- Pricing Simulator: rule-based fair-price estimate and recommendation.
- Model Results: validation MAE table and feature-importance visuals.
- Business Insights: pricing bands, operational interpretation, and recommendation logic.
- Methodology: architecture, notebook order, and deployment boundary.

## Explainability Summary

The project includes feature-importance figures from XGBoost, CatBoost, and LightGBM. These provide a lightweight explainability layer for the portfolio demo. A production-grade version would add SHAP explanations for individual listings so sellers and operators can understand why a specific vehicle receives a particular fair-price estimate.

## Limitations Summary

- The public demo uses rule-based logic rather than loading trained models.
- Raw data, processed features, model files, and output CSVs are intentionally excluded from GitHub.
- Validation metrics come from the notebook workflow, not live inference in Streamlit.
- Production deployment would require artifact storage, scheduled retraining, drift monitoring, access control, and business policy constraints.

## Future Work Summary

- Add SHAP-based listing-level explanations.
- Store trained models in a private model registry or cloud storage.
- Add automated retraining and monitoring.
- Add regional, seasonal, seller, and supply-demand features if available.
- A/B test pricing recommendations against conversion, time-to-sale, and gross margin.

## Best-Fit Roles

- Product Data Scientist
- Senior Data Analyst
- Pricing Data Scientist
- Marketplace Analytics
- Business Intelligence Analyst with ML exposure

## Skills Demonstrated

- Python data science workflow
- Exploratory data analysis
- Feature engineering for tabular ML
- Gradient boosting models
- Model benchmarking and validation
- Ensemble modeling
- Business translation of ML output
- Streamlit deployment
- GitHub portfolio hygiene
