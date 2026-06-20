# CPO Retail Command Center — Project Summary

## Project Links

- Live demo: https://josephwang-dynamic-pricing.streamlit.app/
- GitHub: https://github.com/josephwang-ds/used-car-dynamic-pricing

## Executive Summary

This project combines an end-to-end used-car dynamic-pricing workflow with a bilingual automotive retail decision dashboard. A weighted XGBoost, CatBoost, and LightGBM ensemble estimates fair value on 150,000 public used-car listings. The Streamlit app extends that pricing layer into a synthetic CPO Retail Command Center covering national Sales and After-sales performance, 50-dealer diagnosis, and vehicle-level pricing and inventory actions.

The demo is designed as one operating loop:

1. Detect a network or business-line performance gap.
2. Locate the affected region and dealer.
3. Diagnose Sales, After-sales, customer-experience, and inventory drivers.
4. Translate the finding into CPO reprice, promote, transfer, or retention actions.

## 中文项目摘要

本项目把端到端二手车动态定价流程扩展成一个中英文双语汽车零售决策产品。定价部分基于 15 万条公开二手车挂牌数据，对比 XGBoost、CatBoost、LightGBM，并通过加权集成估算车辆公允价值。Streamlit 应用进一步加入全国 Sales 与 After-sales 经营监控、50 家合成经销商诊断，以及单车 CPO 定价与库存行动。

演示形成一个完整决策闭环：发现全国或业务线异常 → 定位区域和经销商 → 诊断销售、售后、客户体验与库存原因 → 落地调价、促销、跨店调拨或客户召回行动。

## Business Scope

The dashboard keeps the main automotive retail business lines and metric families explicit:

- **NC:** new passenger-car retail deliveries.
- **CPO:** certified pre-owned passenger-car retail deliveries.
- **Vans:** new light-commercial-vehicle retail deliveries in the demo scope.
- **Sales:** units, target achievement, conversion, discount, average selling price, days supply, and inventory aging.
- **After-sales:** revenue, gross profit, repair orders, service retention, absorption, workshop utilization, technician efficiency, first-time fix, and CSI.
- **Dealer performance:** a transparent proposed composite of 45% Sales, 35% After-sales, 15% customer experience, and 5% compliance.

NC, CPO, and Vans target achievement are calculated separately. Vans are excluded from CPO penetration.

## Streamlit App

The app contains seven connected views:

1. **Executive Overview** — ten headline Sales and After-sales KPIs, trends, and management alerts.
2. **Sales KPI** — NC, CPO, and Vans performance, region comparisons, conversion, pricing, and inventory diagnostics.
3. **After-sales KPI** — revenue, repair-order, capacity, retention, quality, and customer-experience measures.
4. **Dealer 360** — synthetic 50-dealer ranking, transparent Dealer Score, raw-KPI drill-down, and root-cause diagnosis.
5. **CPO Pricing & Inventory** — single-vehicle price simulation and an operational action list for aged stock.
6. **AI Copilot** — bilingual, KPI-grounded management recommendations through the OpenAI Responses API, with deterministic offline fallback answers.
7. **About & Method** — metric definitions, model evidence, architecture, limitations, and the synthetic-data boundary.

## Pricing Model

| Model | Validation MAE (CNY) |
|---|---:|
| XGBoost | 514.23 |
| CatBoost | 501.67 |
| LightGBM | 591.00 |
| **Weighted Ensemble** | **496.83** |

The shared holdout split uses `test_size=0.2` and `random_state=42`. The recommendation layer compares the current listing with estimated fair value:

- More than 10% above fair value: overpriced.
- More than 10% below fair value: underpriced.
- Within the ±10% band: fair price.

The public app uses a transparent rule-based simulator because raw data and trained model artifacts are intentionally excluded from the repository. Validation metrics come from the notebook workflow rather than live Streamlit inference.

## LLM Copilot

When `OPENAI_API_KEY` is configured in Streamlit Secrets, the Copilot sends only a compact set of aggregated synthetic KPIs to the OpenAI Responses API. The prompt requires the model to:

- use only the supplied KPI context;
- keep NC, CPO, and Vans distinct;
- separate observations from recommendations;
- avoid inventing company policies, customer facts, or official KPI definitions;
- return a concise diagnosis, evidence, prioritized actions, and a measurement guardrail.

If the key is missing or the API is unavailable, the app remains functional through deterministic KPI-based answers.

## Data and Evidence Boundary

The dealer network is **100% synthetic and deterministic**. No real company data, dealer names, customers, VINs, targets, incentives, or internal extracts are used.

Dealer Score weights and detailed KPI formulas are transparent portfolio assumptions. They must not be interpreted as official definitions from a former employer. The used-car modeling workflow is based on a public dataset, while the dealer-network layer demonstrates product design and automotive retail analytics logic.

## Technical Architecture

- Python, pandas, NumPy, scikit-learn
- XGBoost, CatBoost, LightGBM
- Matplotlib and Streamlit
- OpenAI Python SDK and Responses API
- Deterministic synthetic dealer and inventory data
- English / 中文 interface with CJK-compatible chart fonts
- Streamlit Community Cloud deployment

## Limitations

- Dealer-network data and targets are synthetic.
- The public app does not load the trained ensemble artifact.
- Suggested prices are decision-support estimates, not market truth.
- The Dealer Score is a proposed portfolio construct.
- The LLM explains supplied KPIs but does not calculate authoritative financial or operational metrics.
- Production use would require governed source tables, approved KPI definitions, access control, monitoring, and human review.

## Future Work

- Add SHAP explanations for individual vehicle recommendations.
- Replace synthetic dealer facts with an authorized warehouse model and governed KPI dictionary.
- Add scheduled data-quality checks, drift monitoring, and retraining.
- Evaluate pricing recommendations against conversion, time-to-sale, margin, and retention outcomes.
- Add structured LLM output validation and recommendation feedback tracking.
