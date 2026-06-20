# CPO Retail Command Center — Project Overview

> Reference sheet for updating a résumé and a personal site. Copy the blocks you need.
> Live demo: https://josephwang-dynamic-pricing.streamlit.app/ · Code: https://github.com/josephwang-ds/used-car-dynamic-pricing

---

## 1. One-liner

**EN —** An end-to-end automotive-retail decision system that turns a used-car price-prediction model into a full dealer-network command center: national KPIs → dealer diagnosis → per-vehicle CPO pricing & inventory actions.

**中文 —** 一个端到端的汽车零售决策系统:把二手车价格预测模型扩展成完整的经销商网络指挥中心 —— 从全国 KPI,到经销商诊断,再到单车 CPO 定价与库存行动。

---

## 2. Elevator pitch (30 seconds)

**EN —** I built a used-car dynamic-pricing model (XGBoost / CatBoost / LightGBM ensemble, **496.83 CNY validation MAE** on 150K listings) and then wrapped it in a **CPO Retail Command Center** — a bilingual Streamlit product with a national executive overview, separate Sales and After-sales KPI systems, a 50-dealer "Dealer 360" with a composite Dealer Score, and a CPO pricing-and-inventory action layer. It demonstrates the full loop from monitoring an anomaly nationally, to diagnosing a single dealer, to recommending reprice / promote / transfer actions on specific vehicles.

**中文 —** 我做了一个二手车动态定价模型(XGBoost / CatBoost / LightGBM 加权集成,15 万条数据,**验证集 MAE 496.83 元**),并把它扩展成一个**CPO 零售指挥中心** —— 一个中英文双语的 Streamlit 产品:全国经营总览、独立的 Sales 与 After-sales KPI 体系、50 家经销商的 "Dealer 360" 综合评分,以及 CPO 定价与库存行动层。它完整展示了"全国发现异常 → 单店诊断 → 单车调价/促销/调拨"的决策闭环。

---

## 3. The problem & the product

Used-car and certified-pre-owned (CPO) retail loses money on both ends: overpriced cars sit and age in inventory, underpriced cars give away margin. A pricing model alone is not a product — the value comes from connecting the prediction to the operating decisions a dealer network actually makes.

This project does that in one loop:

1. **Monitor** — a national executive overview surfaces the 10 headline Sales / After-sales KPIs and raises an alert when a dealer slips in the ranking.
2. **Diagnose** — Dealer 360 explains *why* a dealer dropped, using a transparent composite score that always drills back to raw KPIs.
3. **Act** — the CPO pricing & inventory layer scores each vehicle with the ML model and turns aged / overpriced stock into reprice, promote, or transfer recommendations.

---

## 4. Architecture / the decision loop

```
National Executive Overview      ← 10 headline KPIs, monthly trends, anomaly alert
        │  drill down
        ▼
Sales KPI  +  After-sales KPI    ← NC/CPO/Vans tracked separately; absorption, retention, FTF, CSI
        │
        ▼
Dealer 360                       ← 50-dealer ranking, Dealer Score = 0.45 Sales + 0.35 After-sales
        │                          + 0.15 CX + 0.05 Compliance, radar + raw-KPI drill, auto-diagnosis
        ▼
CPO Pricing & Inventory Copilot  ← used-car ML model scores each car → reprice / promote / transfer
                                   + single-vehicle pricing simulator (fair value, gap, recommended)
```

Two supporting tabs: an **AI Narrative / ChatBI** view that answers business questions over the KPI layer, and an **About & Method** tab with the formulas, methodology, and evidence boundary.

---

## 5. Key features (what's in the demo)

- **7 tabs, ~40 KPI tiles, 7 data tables**, full **English / 中文** switching (UI and charts).
- National executive overview with 10 headline KPIs and a 12-month trend.
- Sales KPI system tracking **NC, CPO, and Vans achievement separately** (different targets, mix, inventory logic).
- After-sales KPI system limited to metrics that change a dealer's action: absorption, workshop utilization, technician efficiency, retention, first-time-fix, CSI.
- **Dealer 360** — 50-dealer ranking, a transparent composite Dealer Score, a four-axis radar, raw-KPI drill-down, and automatic root-cause diagnosis.
- **CPO Pricing & Inventory Copilot** — the ML pricing model as an action layer: a single-vehicle simulator plus a dealer inventory list that recommends reprice / promote / transfer with 30-day sell-probability and margin-at-risk.
- Deterministic **synthetic dealer network** (50 dealers, 12 months) so the demo runs anywhere with no private data.

---

## 6. The ML model

| Item | Value |
|---|---|
| Data | 150K used-car listings, 49 engineered features |
| Features | Vehicle age, mileage intensity, power bins, brand/model price aggregates, date parts |
| Models | XGBoost (514.23) · CatBoost (501.67, best single) · LightGBM (591.00) |
| **Ensemble** | **Weighted ensemble — 496.83 CNY validation MAE** |
| Output | Fair value → over/under/fair flag → recommended listing price |
| Explainability | Tree-based feature importance (age + mileage ≈ 44% of output) |

The public app uses a transparent rule-based simulator that mirrors the final recommendation notebook, so the repo stays lightweight and reproducible without shipping raw data or model artifacts.

---

## 7. Demo storyline (use this when presenting)

National sales achievement looks healthy (~100%), but an alert flags dealer **MB-E07** at the bottom of the 50-dealer ranking. Drilling in: new-car sales are fine, but **CPO achievement is only 78%**, with high days-supply and 90+ day aging. After-sales tells the same story — **retention 61%, CSI low**. Dealer 360 confirms the root cause via the composite score and radar. Finally, the CPO tab shows **29 E-Class units stuck 90+ days, overpriced vs. the model's fair value**, and recommends repricing those first, promoting the rest, transferring slow movers, and bundling a service package to win back retention. One loop: national monitoring → dealer diagnosis → single-car pricing action.

---

## 8. Tech stack

Python · pandas · NumPy · scikit-learn · **XGBoost / CatBoost / LightGBM** · Matplotlib · **Streamlit** · Jupyter · Git/GitHub. Bilingual UI; CJK-aware charts with graceful English fallback; `packages.txt` for Streamlit Cloud font support.

---

## 9. Résumé bullets (copy-paste)

### Concise (EN)

- Built a used-car dynamic-pricing model (XGBoost/CatBoost/LightGBM weighted ensemble, **496.83 CNY validation MAE** on 150K listings) and turned it into pricing actions (over/under/fair flags, recommended price).
- Extended it into a **bilingual CPO Retail Command Center** (Streamlit): national executive overview, separate Sales/After-sales KPI systems, and a 50-dealer "Dealer 360" with a composite Dealer Score (45/35/15/5) and root-cause drill-down.
- Designed a **CPO pricing-and-inventory action layer** that flags aged, overpriced stock and recommends reprice / promote / transfer with 30-day sell-probability and margin-at-risk.

### Concise (中文)

- 构建二手车动态定价模型(XGBoost/CatBoost/LightGBM 加权集成,15 万条数据,**验证集 MAE 496.83 元**),并落地为定价行动(高估/低估/公允判定与建议价)。
- 扩展为**中英文双语 CPO 零售指挥中心**(Streamlit):全国经营总览、独立的 Sales/After-sales KPI 体系、50 家经销商 "Dealer 360" 综合评分(45/35/15/5)与根因下钻。
- 设计 **CPO 定价与库存行动层**:识别老化、定价偏高库存,给出调价/促销/跨店调拨建议,附 30 天售出概率与风险毛利。

### One-line (for a dense résumé)

> **Automotive Retail Pricing & Analytics (personal project)** — Built a 150K-listing used-car pricing ensemble (496.83 CNY MAE) and a bilingual Streamlit "CPO Retail Command Center" connecting national KPIs, a 50-dealer composite score, and per-vehicle reprice/promote/transfer actions.

---

## 10. Personal-site copy

### Card blurb (short)

**CPO Retail Command Center** — A used-car pricing model (496.83 CNY MAE) wrapped in a full automotive-retail decision system: national KPIs → 50-dealer diagnosis → per-vehicle CPO pricing & inventory actions. Bilingual Streamlit app. [Live demo →]

### Longer description

This project started as a used-car dynamic-pricing model and grew into a complete retail decision product. A weighted ensemble of XGBoost, CatBoost, and LightGBM estimates fair market value on 150K listings at **496.83 CNY validation MAE**, then a recommendation layer flags over- and under-priced cars.

I wrapped that model in a **CPO Retail Command Center**: a bilingual Streamlit app with a national executive overview, separate Sales and After-sales KPI systems, a 50-dealer "Dealer 360" with a transparent composite score and automatic root-cause diagnosis, and a CPO pricing-and-inventory action layer that turns aged, overpriced stock into reprice / promote / transfer recommendations. The design reflects real dealer-analytics thinking — NC, CPO, and Vans tracked separately; after-sales limited to action-driving metrics; every composite score drilling back to raw KPIs — while running entirely on synthetic, reproducible data.

---

## 11. Caveat / compliance note (keep this honest)

The dealer network is **100% synthetic and deterministic**. No real Mercedes-Benz data, dealer names, customers, VINs, targets, or incentives are used. The dealer layer demonstrates a metric system and decision flow designed from dealer-analytics experience; the Dealer Score weights, model mix, and KPI formulas are transparent portfolio assumptions, not company definitions. The single-vehicle pricing model is trained on a public used-car dataset.
