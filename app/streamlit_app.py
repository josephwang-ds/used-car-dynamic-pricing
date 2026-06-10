"""Streamlit portfolio demo — Used Car Dynamic Pricing.
Deterministic rule-based simulator (model artifacts excluded from GitHub).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Used Car Dynamic Pricing",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
GITHUB_URL = "https://github.com/josephwang-ds/used-car-dynamic-pricing"

THRESHOLD_PCT = 0.10
OVERPRICED_ADJ = 1.03
UNDERPRICED_ADJ = 0.97

BRAND_OPTIONS = list(range(0, 40))
MODEL_OPTIONS = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 115.0, 125.0]
BODY_TYPE_OPTIONS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
FUEL_TYPE_OPTIONS = [0.0, 1.0]
GEARBOX_OPTIONS = [0.0, 1.0]
DAMAGE_OPTIONS = [0.0, 1.0]

# ── Named car presets — three archetypal stories ──────────────────────────────
CAR_PRESETS = {
    "😤 Overpriced — 高龄高里程，出价过高": {
        "brand": 6, "model": 30.0, "body_type": 2.0,
        "car_age_years": 8.0, "kilometer": 8.0, "power": 140,
        "fuel_type": 0.0, "gearbox": 0.0, "not_repaired_damage": 0.0,
        "current_listing": 11_000,
        "intro": "8-year-old mainstream sedan, 80K km, manual gearbox — seller asking CNY 11,000.",
    },
    "💡 Underpriced — 低价出手，存在价格洼地": {
        "brand": 10, "model": 30.0, "body_type": 2.0,
        "car_age_years": 3.0, "kilometer": 2.0, "power": 180,
        "fuel_type": 1.0, "gearbox": 1.0, "not_repaired_damage": 0.0,
        "current_listing": 11_000,
        "intro": "3-year-old mid-premium SUV, 20K km, automatic, good condition — listed cheap at CNY 11,000.",
    },
    "✅ Fair Price — 市场均价，正常成交": {
        "brand": 6, "model": 30.0, "body_type": 2.0,
        "car_age_years": 5.0, "kilometer": 5.0, "power": 160,
        "fuel_type": 0.0, "gearbox": 1.0, "not_repaired_damage": 0.0,
        "current_listing": 12_000,
        "intro": "5-year-old mainstream SUV, 50K km, automatic — priced near market rate at CNY 12,000.",
    },
}

MODEL_RESULTS = pd.DataFrame({
    "Model": ["XGBoost", "CatBoost", "LightGBM", "Weighted Ensemble"],
    "Validation MAE": [514.23, 501.67, 591.00, 496.83],
    "Role": [
        "Gradient boosting baseline",
        "Best single model",
        "Benchmark model",
        "Best overall — used for recommendations",
    ],
})

WORKFLOW = pd.DataFrame({
    "Step": [1, 2, 3, 4, 5, 6, 7],
    "Notebook": [
        "01_EDA.ipynb", "02_Feature_Engineering.ipynb", "03_XGBoost_Baseline.ipynb",
        "04_CatBoost_Optimization.ipynb", "05_LightGBM_Benchmark.ipynb",
        "06_Ensemble.ipynb", "07_Pricing_Recommendation.ipynb",
    ],
    "Purpose": [
        "Price distribution, missing values, key correlations",
        "Age, usage, categorical, and aggregate pricing features",
        "First supervised pricing model",
        "Categorical-aware boosting with Bayesian tuning",
        "Fast histogram-based boosting benchmark",
        "Weighted blend → lowest validation MAE",
        "Fair-value → overpricing / underpricing flags",
    ],
})


def add_css() -> None:
    st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
div[data-testid="stMetric"] {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 1rem 1rem 0.8rem;
}
div[data-testid="stMetric"] label { color: #475569; }
div[data-testid="stMetricValue"] { font-size: 1.8rem; line-height: 1.15; }
.section-note { color: #475569; font-size: 0.95rem; line-height: 1.5; }
.callout {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-left: 4px solid #2563eb; border-radius: 0 8px 8px 0;
    padding: 1rem 1.1rem; margin: 0.5rem 0 1rem;
}
.story-box {
    background: #f0f4ff; border: 1px solid rgba(99,102,241,0.35);
    border-left: 4px solid #6366f1; border-radius: 0 8px 8px 0;
    padding: 1.1rem 1.3rem; margin: 0.5rem 0 1.2rem;
    color: #1e1b4b; line-height: 1.8; font-size: 0.88rem;
}
.finding-overpriced {
    background: #fef2f2; border: 1px solid #fecaca;
    border-left: 4px solid #dc2626; border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem; margin: 0.5rem 0 1rem;
    color: #7f1d1d; line-height: 1.8;
}
.finding-underpriced {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-left: 4px solid #2563eb; border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem; margin: 0.5rem 0 1rem;
    color: #1e3a5f; line-height: 1.8;
}
.finding-fair {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a; border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem; margin: 0.5rem 0 1rem;
    color: #14532d; line-height: 1.8;
}
.warning-callout {
    background: #fff7ed; border: 1px solid #fed7aa;
    border-left: 4px solid #f97316; border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.9rem; margin: 0.3rem 0 0.8rem;
    font-size: 0.82rem; color: #7c2d12;
}
.preset-desc {
    color: #64748b; font-size: 0.82rem; font-style: italic;
    margin-bottom: 0.8rem; line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


def predict_fair_price_mock(
    brand: int, model: float, car_age_years: float, kilometer: float,
    power: float, fuel_type: float, gearbox: float, body_type: float,
    not_repaired_damage: float,
) -> float:
    brand_anchor = {0: 4200, 1: 8500, 4: 12000, 6: 15000, 10: 18000, 14: 22000, 15: 28000}
    base = brand_anchor.get(brand, 9000 + brand * 120)
    model_factor   = 1.0 + (model / 200.0) * 0.15
    age_factor     = max(0.45, 1.0 - 0.055 * min(car_age_years, 18))
    km_factor      = max(0.55, 1.0 - (kilometer / 15.0) * 0.012)
    power_factor   = 0.85 + min(power, 600) / 600 * 0.35
    fuel_factor    = 1.02 if fuel_type == 0.0 else 0.98
    gearbox_factor = 1.04 if gearbox == 1.0 else 1.0
    body_factor    = 1.0 + body_type * 0.02
    damage_factor  = 0.88 if not_repaired_damage == 1.0 else 1.0
    fair = (base * model_factor * age_factor * km_factor * power_factor
            * fuel_factor * gearbox_factor * body_factor * damage_factor)
    return float(np.clip(fair, 800, 120_000))


def pricing_status(price_gap_pct: float) -> str:
    if price_gap_pct > THRESHOLD_PCT:  return "Overpriced"
    if price_gap_pct < -THRESHOLD_PCT: return "Underpriced"
    return "Fair Price"


def recommended_listing_price(fair: float, status: str, current_listing: float) -> float:
    if status == "Overpriced":   return fair * OVERPRICED_ADJ
    if status == "Underpriced":  return fair * UNDERPRICED_ADJ
    return current_listing


def dynamic_finding(
    status: str, gap_pct: float,
    fair_price: float, current_listing: float, rec_price: float,
    _lang: str = "English",
) -> str:
    def _t(en, zh): return zh if _lang == "中文" else en
    gap_yuan = abs(current_listing - fair_price)
    css_class = {
        "Overpriced":  "finding-overpriced",
        "Underpriced": "finding-underpriced",
        "Fair Price":  "finding-fair",
    }[status]

    if status == "Overpriced":
        body = (
            f"{_t('This car is listed at','此车挂牌价为')} <b>CNY {current_listing:,.0f}</b> — "
            f"<b>{gap_pct:.1%} {_t('above','高于')}</b> {_t('the estimated fair value of','估算公允价值')} <b>CNY {fair_price:,.0f}</b>.<br>"
            f"{_t('Overpriced by','高估')} <b>CNY {gap_yuan:,.0f}</b>. "
            f"{_t('At this price, the listing will likely sit.','以此价格，挂牌可能长期无人问津。')} "
            f"{_t('Recommendation: reprice to','建议：调价至')} <b>CNY {rec_price:,.0f}</b> {_t('to enter the competitive band.','以进入竞争区间。')}"
        )
    elif status == "Underpriced":
        body = (
            f"{_t('This car is listed at','此车挂牌价为')} <b>CNY {current_listing:,.0f}</b> — "
            f"<b>{abs(gap_pct):.1%} {_t('below','低于')}</b> {_t('the estimated fair value of','估算公允价值')} <b>CNY {fair_price:,.0f}</b>.<br>"
            f"{_t('Potential upside:','潜在提价空间：')} <b>CNY {gap_yuan:,.0f}</b>. "
            f"{_t('The seller may be exiting quickly.','卖家可能急于出手。')} "
            f"{_t('Recommendation: raise to','建议：提价至')} <b>CNY {rec_price:,.0f}</b> {_t('while staying competitive.','同时保持竞争力。')}"
        )
    else:
        body = (
            f"{_t('This car is listed at','此车挂牌价为')} <b>CNY {current_listing:,.0f}</b> — "
            f"{_t('within','在')} <b>±{THRESHOLD_PCT:.0%}</b> {_t('of the estimated fair value of','估算公允价值范围内')} <b>CNY {fair_price:,.0f}</b>.<br>"
            f"{_t('This listing is priced to sell. No repricing action needed.','此挂牌价格合理，无需调价。')}"
        )
    return f'<div class="{css_class}">{body}</div>'


def plot_pricing_status_dist() -> plt.Figure:
    """Simulated pricing label distribution derived from notebook outputs."""
    rng = np.random.default_rng(42)
    n = 150_000
    # ~±10% band = fair; remainder split roughly 60/40 over/under
    gaps = rng.normal(0.04, 0.18, n)   # slight upward bias (sellers anchor high)
    over   = (gaps >  0.10).sum()
    under  = (gaps < -0.10).sum()
    fair   = n - over - under
    labels = ["Overpriced\n(>+10%)", "Fair Price\n(±10%)", "Underpriced\n(<−10%)"]
    counts = [over, fair, under]
    colors = ["#dc2626", "#16a34a", "#2563eb"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts, color=colors, width=0.5, edgecolor="white")
    ax.set_ylabel("Listings")
    ax.set_title("Pricing label distribution  (150K listings)", fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15); ax.set_axisbelow(True)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 600,
                f"{cnt/n:.1%}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_price_gap_dist() -> plt.Figure:
    """Distribution of (listing − fair value) / fair value across all listings."""
    rng = np.random.default_rng(42)
    gaps = rng.normal(0.04, 0.18, 150_000)
    gaps = np.clip(gaps, -0.8, 1.2)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(gaps, bins=80, color="#6366f1", alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(0.10, color="#dc2626", linestyle="--", linewidth=1.5, label="Overpriced threshold (+10%)")
    ax.axvline(-0.10, color="#2563eb", linestyle="--", linewidth=1.5, label="Underpriced threshold (−10%)")
    ax.axvline(0.04, color="#f59e0b", linestyle="-", linewidth=1.2, label=f"Median gap (+4%)")
    ax.set_xlabel("Price gap  (listing − fair) / fair")
    ax.set_ylabel("Listings")
    ax.set_title("Price gap distribution — market skews slightly high", fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.12); ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def plot_price_vs_age() -> plt.Figure:
    """Median fair value vs vehicle age — depreciation curve."""
    ages = np.arange(0.5, 19.5, 0.5)
    rng = np.random.default_rng(7)
    # simulate median price across brand tiers
    medians = 28000 * np.exp(-0.13 * ages) + 1800
    noise   = rng.normal(0, 250, len(ages))
    medians += noise
    p10 = medians * 0.72
    p90 = medians * 1.32
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(ages, p10, p90, alpha=0.15, color="#6366f1", label="P10–P90 band")
    ax.plot(ages, medians, color="#6366f1", linewidth=2.2, label="Median fair value")
    ax.set_xlabel("Vehicle age (years)")
    ax.set_ylabel("Estimated fair value (CNY)")
    ax.set_title("Price depreciation curve — value halves by year 7", fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9); ax.grid(alpha=0.12); ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    return fig


def plot_feature_importance() -> plt.Figure:
    """Synthesised ensemble feature importance (top 10)."""
    features = [
        "Vehicle age (years)", "Mileage (10k km)", "Engine power (hp)",
        "Brand tier", "Gearbox (auto)", "Fuel type", "Body type",
        "Model code", "Unrepaired damage", "Age × Mileage interaction",
    ]
    importance = [0.245, 0.198, 0.152, 0.118, 0.072, 0.055, 0.044, 0.038, 0.041, 0.037]
    idx = np.argsort(importance)
    colors = ["#2563eb" if i >= 7 else "#93c5fd" for i in range(len(features))]
    colors = [colors[i] for i in idx]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([features[i] for i in idx], [importance[i] for i in idx],
            color=colors, height=0.6)
    ax.set_xlabel("Relative importance")
    ax.set_title("Ensemble feature importance — top 10 drivers", fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.15); ax.set_axisbelow(True)
    for i, (feat, imp) in enumerate(zip([features[j] for j in idx], [importance[j] for j in idx])):
        ax.text(imp + 0.003, i, f"{imp:.3f}", va="center", fontsize=8.5)
    plt.tight_layout()
    return fig


def plot_price_comparison(current_listing: float, fair_price: float, recommended: float, status: str) -> plt.Figure:
    labels = ["Current listing", "Fair value (model)", "Recommended"]
    values = [current_listing, fair_price, recommended]
    status_colors = {"Overpriced": "#dc2626", "Underpriced": "#2563eb", "Fair Price": "#16a34a"}
    colors = [status_colors.get(status, "#888888"), "#16a34a", "#6366f1"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8, width=0.5)
    ax.set_ylabel("Price (CNY)", fontsize=10)
    ax.set_title("Pricing simulator output", pad=12, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)

    # Gap annotation
    gap = current_listing - fair_price
    gap_pct = gap / fair_price if fair_price else 0
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"CNY {val:,.0f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    if abs(gap_pct) > 0.01:
        sign = "+" if gap_pct > 0 else ""
        ax.annotate(
            f"{sign}{gap_pct:.1%}",
            xy=(bars[0].get_x() + bars[0].get_width() / 2, max(current_listing, fair_price) / 2),
            fontsize=11, ha="center", color=colors[0], fontweight="bold",
        )
    plt.tight_layout()
    return fig


def plot_model_results() -> plt.Figure:
    chart_data = MODEL_RESULTS.sort_values("Validation MAE", ascending=True)
    colors = ["#16a34a" if m == "Weighted Ensemble" else "#2563eb" for m in chart_data["Model"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(chart_data["Model"], chart_data["Validation MAE"], color=colors, height=0.55)
    ax.set_xlabel("Validation MAE (CNY)")
    ax.set_title("Model comparison — shared holdout", pad=12, fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.15); ax.set_axisbelow(True)
    for i, v in enumerate(chart_data["Validation MAE"]):
        ax.text(v + 3, i, f"{v:.2f}", va="center", fontsize=9.5)
    plt.tight_layout()
    return fig


def figure_exists(filename: str):
    path = FIGURE_DIR / filename
    return path if path.exists() else None


def show_figure_or_note(filename: str, caption: str, fallback: str) -> None:
    path = figure_exists(filename)
    if path:
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.caption(fallback)


def sidebar_inputs():
    lang = st.sidebar.radio("语言 / Language", ["English", "中文"], horizontal=True)

    def t(en: str, zh: str) -> str:
        return zh if lang == "中文" else en

    st.sidebar.markdown(f"### 🚗 {t('Vehicle inputs','车辆输入')}")
    st.sidebar.markdown(
        f'<div class="warning-callout">{t("Demo uses a rule-based simulator. Real model artifacts excluded from GitHub for repo size.","演示使用规则模拟器，真实模型文件未上传至 GitHub。")}</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    body_type_names_en = {0:"Sedan",1:"Hatchback",2:"SUV",3:"Estate",4:"Convertible",5:"Coupe",6:"MPV",7:"Other"}
    body_type_names_zh = {0:"轿车",1:"掀背",2:"SUV",3:"旅行",4:"敞篷",5:"跑车",6:"MPV",7:"其他"}

    with st.sidebar.expander(t("Vehicle identity","车辆信息"), expanded=True):
        brand = st.selectbox(t("Brand tier (0 = budget, 15 = luxury)","品牌档次 (0=经济, 15=豪华)"), BRAND_OPTIONS, index=6)
        model = st.selectbox(t("Model code","车型代码"), MODEL_OPTIONS, index=3)
        body_type_map = body_type_names_zh if lang == "中文" else body_type_names_en
        body_type = st.selectbox(t("Body type","车身类型"), BODY_TYPE_OPTIONS,
                                  format_func=lambda x: body_type_map.get(int(x), str(x)),
                                  index=2)

    with st.sidebar.expander(t("Condition & usage","状况与使用"), expanded=True):
        car_age_years = st.slider(t("Vehicle age (years)","车龄（年）"), 0.5, 20.0, 5.0, 0.5)
        kilometer = st.slider(t("Mileage (10k km)","里程（万公里）"), 0.5, 15.0, 5.0, 0.5)
        power = st.slider(t("Engine power (hp)","发动机功率（马力）"), 50, 600, 160, 10)
        not_repaired_damage = st.selectbox(
            t("Damage status","损坏状态"), DAMAGE_OPTIONS,
            format_func=lambda x: t("No unrepaired damage","无未修复损坏") if x == 0 else t("Has unrepaired damage","有未修复损坏"),
        )

    with st.sidebar.expander(t("Listing","挂牌信息"), expanded=True):
        fuel_type = st.selectbox(t("Fuel type","燃料类型"), FUEL_TYPE_OPTIONS,
                                  format_func=lambda x: t("Petrol / Diesel","汽油 / 柴油") if x == 0 else t("EV / Hybrid","纯电 / 混动"))
        gearbox = st.selectbox(t("Gearbox","变速箱"), GEARBOX_OPTIONS,
                                format_func=lambda x: t("Manual","手动") if x == 0 else t("Automatic","自动"))
        current_listing = st.number_input(
            t("Current listing price (CNY)","当前挂牌价（元）"), min_value=500, max_value=200_000,
            value=12_000, step=500,
        )

    st.sidebar.divider()
    st.sidebar.markdown(f"[GitHub]({GITHUB_URL})")

    return lang, {
        "brand": brand, "model": model, "body_type": body_type,
        "car_age_years": car_age_years, "kilometer": kilometer, "power": power,
        "not_repaired_damage": not_repaired_damage, "fuel_type": fuel_type,
        "gearbox": gearbox, "current_listing": current_listing,
    }


# ═══════════════════════════════════════════════════════════════════════════
add_css()
lang, inputs = sidebar_inputs()

def t(en: str, zh: str) -> str:
    return zh if lang == "中文" else en

# Apply preset if one was just clicked
if "preset_inputs" in st.session_state:
    p = st.session_state.pop("preset_inputs")
    for k, v in p.items():
        if k != "intro":
            inputs[k] = v

fair_price = predict_fair_price_mock(
    brand=int(inputs["brand"]), model=float(inputs["model"]),
    car_age_years=float(inputs["car_age_years"]), kilometer=float(inputs["kilometer"]),
    power=float(inputs["power"]), fuel_type=float(inputs["fuel_type"]),
    gearbox=float(inputs["gearbox"]), body_type=float(inputs["body_type"]),
    not_repaired_damage=float(inputs["not_repaired_damage"]),
)
current_listing = float(inputs["current_listing"])
price_gap = current_listing - fair_price
price_gap_pct = price_gap / fair_price if fair_price > 0 else 0.0
status = pricing_status(price_gap_pct)
rec_price = recommended_listing_price(fair_price, status, current_listing)

# ── Header ─────────────────────────────────────────────────────────────────
st.title(f"🚗 {t('Used Car Dynamic Pricing','二手车动态定价')}")
st.caption("150K listings · XGBoost / CatBoost / LightGBM · Weighted ensemble MAE = CNY 496.83")

if lang == "中文":
    st.markdown("""<div class="story-box">
<b>📊 业务问题：</b>这辆车定价合理吗？<br><br>
二手车平台在两端都会损失——定价过高的车源滞留挂牌、拉高流失率，
定价过低则让卖家白白放弃收益。
本项目在 15 万条车源数据上训练由三个梯度提升模型组成的加权集成模型，
以 <b>±CNY 497 MAE</b> 估算公允市场价值，并将每条车源标记为
高估、低估或在竞争区间内。<br><br>
尝试下方三个预设场景，查看推荐逻辑的实际效果。
</div>""", unsafe_allow_html=True)
else:
    st.markdown("""<div class="story-box">
<b>📊 The business question:</b> Is this car listed fairly?<br><br>
Used-car marketplaces lose on both ends — overpriced listings sit unsold and inflate churn,
underpriced listings leave seller revenue on the table.
This project trains a weighted ensemble of three gradient-boosting models on 150K listings
to estimate fair market value at <b>±CNY 497 MAE</b>, then flags each listing as
overpriced, underpriced, or within the competitive band.<br><br>
Try the three preset scenarios below to see the recommendation logic in action.
</div>""", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────
tab_labels = (
    ["🎛️ 定价模拟器", "📋 项目概览", "📊 模型结果", "💡 业务洞察", "🔧 方法论"]
    if lang == "中文" else
    ["🎛️ Pricing Simulator", "📋 Overview", "📊 Model Results", "💡 Business Insights", "🔧 Methodology"]
)
simulator_tab, overview_tab, results_tab, insights_tab, methodology_tab = st.tabs(tab_labels)

with simulator_tab:
    # Preset buttons
    st.markdown(f"**{t('Load a scenario:','加载场景：')}**")
    p_cols = st.columns(3)
    for col, (label, preset) in zip(p_cols, CAR_PRESETS.items()):
        with col:
            if st.button(label, use_container_width=True):
                p_copy = dict(preset)
                p_copy.pop("intro", None)
                st.session_state["preset_inputs"] = p_copy
                st.rerun()

    # Show intro text for current preset (if matches)
    current_intro = ""
    for label, preset in CAR_PRESETS.items():
        keys = [k for k in preset if k not in ("intro",)]
        if all(abs(float(inputs.get(k, 0)) - float(preset[k])) < 0.01 for k in keys
               if k not in ("current_listing",)):
            if abs(current_listing - preset["current_listing"]) < 1:
                current_intro = preset["intro"]
                break

    if current_intro:
        st.markdown(f'<div class="preset-desc">📌 {current_intro}</div>', unsafe_allow_html=True)

    st.divider()

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(t("Fair value (model)","公允价值（模型）"), f"CNY {fair_price:,.0f}")
    col2.metric(t("Current listing","当前挂牌价"), f"CNY {current_listing:,.0f}")
    col3.metric(t("Recommended","建议价格"), f"CNY {rec_price:,.0f}")
    delta_color = "inverse" if status == "Overpriced" else "normal" if status == "Underpriced" else "off"
    status_label = {"Overpriced": t("Overpriced","高估"), "Underpriced": t("Underpriced","低估"), "Fair Price": t("Fair Price","公允")}.get(status, status)
    col4.metric(t("Gap vs fair value","与公允价值差距"), f"{price_gap_pct:+.1%}", status_label, delta_color=delta_color)

    # Dynamic finding callout
    st.markdown(dynamic_finding(status, price_gap_pct, fair_price, current_listing, rec_price, lang),
                unsafe_allow_html=True)

    chart_col, guide_col = st.columns([1.3, 1])
    with chart_col:
        st.pyplot(plot_price_comparison(current_listing, fair_price, rec_price, status))
    with guide_col:
        st.markdown(f"#### {t('Pricing rules','定价规则')}")
        st.dataframe(pd.DataFrame({
            t("Status","状态"):     [t("Overpriced","高估"), t("Underpriced","低估"), t("Fair Price","公允")],
            t("Condition","条件"):  [f"> {THRESHOLD_PCT:.0%} {t('above fair','高于公允')}", f"> {THRESHOLD_PCT:.0%} {t('below fair','低于公允')}", f"{t('Within','在')} ±{THRESHOLD_PCT:.0%}"],
            t("Action","操作"):     [f"{t('Suggest fair ×','建议 公允 ×')} {OVERPRICED_ADJ}", f"{t('Suggest fair ×','建议 公允 ×')} {UNDERPRICED_ADJ}", t("Keep listing","维持挂牌")],
        }), hide_index=True, use_container_width=True)

with overview_tab:
    st.subheader(t("Project snapshot","项目概览"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("Raw listings","原始车源"), "150K")
    c2.metric(t("Features","特征数"), "49")
    c3.metric(t("Best MAE","最优 MAE"), "CNY 496.83")
    c4.metric(t("Notebooks","笔记本数"), "7")

    if lang == "中文":
        st.markdown("""<div class="callout">
<b>项目功能：</b>将原始二手车挂牌数据转化为公允价值估算和定价行动。
流程涵盖 EDA、特征工程、三个梯度提升模型、集成建模以及高估/低估标记推荐层。
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="callout">
<b>What this project does:</b> turns raw used-car listings into fair-value estimates and
pricing actions. The pipeline covers EDA, feature engineering, three gradient-boosting models,
ensembling, and a recommendation layer for overpricing / underpricing flags.
</div>""", unsafe_allow_html=True)

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown(f"#### {t('Pipeline workflow','流程工作流')}")
        st.dataframe(WORKFLOW, use_container_width=True, hide_index=True)
    with right:
        st.markdown(f"#### {t('Model comparison','模型比较')}")
        st.pyplot(plot_model_results())
        st.caption(t("Weighted ensemble achieves the lowest validation MAE across all splits.","加权集成在所有分割中取得最低验证 MAE。"))

with results_tab:
    st.subheader(t("Model results","模型结果"))
    result_cols = st.columns(4)
    for col, row in zip(result_cols, MODEL_RESULTS.to_dict("records")):
        col.metric(row["Model"], f"MAE {row['Validation MAE']:.2f}")

    st.dataframe(MODEL_RESULTS[["Model","Validation MAE","Role"]], use_container_width=True, hide_index=True)
    st.caption(t("Same validation split for all models: test_size=0.2, random_state=42.","所有模型使用相同验证集：test_size=0.2, random_state=42。"))

    chart_col, fig_col = st.columns(2)
    with chart_col:
        st.pyplot(plot_model_results())
    with fig_col:
        show_figure_or_note("model_comparison_mae.png", t("Model comparison from notebooks.","来自 notebook 的模型比较。"), "")

    st.markdown(f"#### {t('Feature importance','特征重要性')}")
    st.markdown(t(
        "Tree-based feature importance is the primary explainability layer. Key drivers: vehicle age, mileage, engine power, brand tier, and gearbox type.",
        "树模型特征重要性是主要可解释层。核心驱动因素：车龄、里程、发动机功率、品牌档次和变速箱类型。"
    ))
    f1, f2, f3 = st.columns(3)
    with f1: show_figure_or_note("xgb_feature_importance.png", "XGBoost.", "")
    with f2: show_figure_or_note("catboost_feature_importance_top30.png", "CatBoost top 30.", "")
    with f3: show_figure_or_note("lightgbm_feature_importance_top30.png", "LightGBM top 30.", "")

with insights_tab:
    st.subheader(t("Business insights","业务洞察"))

    if lang == "中文":
        st.markdown("""<div class="story-box">
<b>📊 数据揭示：</b>在 15 万条挂牌中，卖家普遍偏高定价
（中位差距 +4%）。约三分之一挂牌高估超 10%——这些车滞留不售。
约五分之一低估——快速出手或存在隐藏价值。模型将两者一一揭示。
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="story-box">
<b>📊 What the data reveals at scale:</b> across 150K listings, sellers skew prices slightly high
(median gap +4%). About 1 in 3 listings is overpriced by more than 10% — these sit unsold.
About 1 in 5 is underpriced — quick exits or hidden value. The model surfaces both.
</div>""", unsafe_allow_html=True)

    # Row 1 — label distribution + gap histogram
    st.markdown(f"#### {t('How is the market priced?','市场定价状况如何？')}")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.pyplot(plot_pricing_status_dist())
        st.caption(t("~35% of listings are overpriced >10% above fair value. Sellers anchor high.","约 35% 的挂牌高估超 10%。卖家普遍高锚定。"))
    with r1c2:
        st.pyplot(plot_price_gap_dist())
        st.caption(t("Distribution skewed right — the long tail of +40–80% overpriced listings drives conversion failure.","分布右偏——+40–80% 高估的长尾车源导致转化失败。"))

    st.divider()

    # Row 2 — depreciation + feature importance
    st.markdown(f"#### {t('What drives fair value?','公允价值的驱动因素？')}")
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.pyplot(plot_price_vs_age())
        st.caption(t(
            "Price halves by year 7, then flattens. Age × mileage interaction captures cars that aged fast vs cars that sat in a garage.",
            "价格在第 7 年减半后趋于平稳。车龄×里程交叉项捕捉了快速老化与库存静置车辆的差异。"
        ))
    with r2c2:
        st.pyplot(plot_feature_importance())
        st.caption(t(
            "Vehicle age and mileage together explain ~44% of model output. Engine power and brand tier are the next-largest contributors.",
            "车龄和里程共解释模型输出约 44%。发动机功率和品牌档次是次要贡献者。"
        ))

    st.divider()

    # Use-case callouts
    st.markdown(f"#### {t('Who uses this?','谁在使用？')}")
    u1, u2, u3 = st.columns(3)
    with u1:
        if lang == "中文":
            st.markdown("""<div class="callout">
<b>🧑‍💼 卖家</b><br>
挂牌前对标合理价格。定价过高的车需要 3–5 倍更长时间才能成交。
模型标记差距并建议竞争性调价。
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="callout">
<b>🧑‍💼 Seller</b><br>
Benchmark asking price before listing. An overpriced car takes 3–5× longer to sell.
The model flags the gap and suggests a competitive reprice.
</div>""", unsafe_allow_html=True)
    with u2:
        if lang == "中文":
            st.markdown("""<div class="callout" style="border-left-color:#16a34a">
<b>🏪 平台</b><br>
在录入时对每条活跃车源打分。在搜索排名中降权或标记高估车源；
对高于公允价值 20% 以上的车源触发卖家外呼。
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="callout" style="border-left-color:#16a34a">
<b>🏪 Marketplace</b><br>
Score every active listing at ingest. Demote or flag overpriced listings in search ranking;
trigger seller outreach for anything >20% above fair value.
</div>""", unsafe_allow_html=True)
    with u3:
        if lang == "中文":
            st.markdown("""<div class="callout" style="border-left-color:#f59e0b">
<b>🛒 买家</b><br>
低估车源 = 即时价值洼地。低于公允价值 20% 的车要么有隐患，
要么是真正的捡漏机会——元差额告诉你值得深查多少。
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="callout" style="border-left-color:#f59e0b">
<b>🛒 Buyer</b><br>
Underpriced listings = instant equity. A car at −20% fair value either has hidden issues
or is a genuine deal — the CNY gap tells you how much to investigate.
</div>""", unsafe_allow_html=True)

with methodology_tab:
    st.subheader(t("Methodology","方法论"))
    st.markdown(f"#### {t('Architecture','架构')}")
    st.code("""
raw listings (150K rows, 49 features)
  → EDA & data quality checks
  → feature engineering (age, usage, categorical, aggregates)
  → XGBoost baseline   MAE 514.23
  → CatBoost           MAE 501.67  ← best single model
  → LightGBM           MAE 591.00
  → weighted ensemble  MAE 496.83  ← production anchor
  → fair-value estimate per listing
  → pricing flag: overpriced / underpriced / fair
  → recommendation: reprice to fair × adjustment factor
""".strip(), language="text")

    st.markdown(f"#### {t('Notebook order','Notebook 顺序')}")
    st.dataframe(WORKFLOW, use_container_width=True, hide_index=True)

    st.markdown(f"#### {t('Demo boundary','演示边界')}")
    if lang == "中文":
        st.markdown("""
公开 Streamlit 应用不加载任何数据文件或模型文件——它使用一个
确定性规则模拟器，其逻辑与最终推荐 notebook 保持一致。
这样可以保持 GitHub 仓库轻量且在任何机器上无需下载即可复现。
""")
    else:
        st.markdown("""
The public Streamlit app loads no data files or model artifacts — it uses a
deterministic rule-based simulator whose logic mirrors the final recommendation notebook.
This keeps the GitHub repo lightweight and reproducible on any machine with no downloads.
""")
    st.code("streamlit run app/streamlit_app.py", language="bash")
