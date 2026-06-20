"""Streamlit portfolio demo — Mercedes-style CPO Retail Command Center.

A single decision-loop product: National Executive Overview → Sales / After-sales
KPI systems → Dealer 360 diagnosis → CPO Pricing & Inventory action layer (wrapping
the used-car dynamic-pricing ML model). All dealer-network figures are synthetic and
deterministic. No real Mercedes-Benz data, dealer names, customers, VINs, targets, or
incentives are used. See the About tab for the evidence boundary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import streamlit as st


def setup_cjk_font() -> bool:
    """Enable Chinese glyphs in charts if any CJK font is available on the host.

    Works on the user's Mac (PingFang / Heiti), Streamlit Cloud (Noto CJK via
    packages.txt), and most Linux boxes. If no CJK font is found, charts fall back
    to English text so they never render tofu boxes.
    """
    explicit = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",     # Debian / Streamlit Cloud
        "/System/Library/Fonts/PingFang.ttc",                          # macOS
        "/System/Library/Fonts/STHeiti Light.ttc",                     # macOS
        "/Library/Fonts/Arial Unicode.ttf",                            # macOS
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",                # Linux
        "C:/Windows/Fonts/msyh.ttc",                                   # Windows
    ]
    for p in explicit:
        try:
            if Path(p).exists():
                fm.fontManager.addfont(p)
        except Exception:
            pass
    candidates = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans SC", "PingFang SC",
        "Heiti SC", "STHeiti", "Hiragino Sans GB", "Microsoft YaHei", "SimHei",
        "Source Han Sans SC", "Source Han Sans CN", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for fam in candidates:
        if fam in available:
            plt.rcParams["font.sans-serif"] = [fam] + list(plt.rcParams.get("font.sans-serif", []))
            plt.rcParams["axes.unicode_minus"] = False
            return True
    plt.rcParams["axes.unicode_minus"] = False
    return False


CJK_OK = setup_cjk_font()


st.set_page_config(
    page_title="CPO Retail Command Center",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
GITHUB_URL = "https://github.com/josephwang-ds/used-car-dynamic-pricing"

# ── Pricing-model constants (public used-car dataset scale) ───────────────────
THRESHOLD_PCT = 0.10
OVERPRICED_ADJ = 1.03
UNDERPRICED_ADJ = 0.97

BRAND_OPTIONS = list(range(0, 40))
MODEL_OPTIONS = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 115.0, 125.0]
BODY_TYPE_OPTIONS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
FUEL_TYPE_OPTIONS = [0.0, 1.0]
GEARBOX_OPTIONS = [0.0, 1.0]
DAMAGE_OPTIONS = [0.0, 1.0]

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

# ── Synthetic dealer-network universe ─────────────────────────────────────────
REGIONS = ["North", "East", "South", "West", "Central"]
REGIONS_ZH = {"North": "华北", "East": "华东", "South": "华南", "West": "西部", "Central": "华中"}
DEALERS_PER_REGION = 10
FOCUS_DEALER_ID = "MB-E07"        # the dealer that drops in ranking (demo story)
CPO_MODELS = ["A-Class", "C-Class", "E-Class", "S-Class", "GLA", "GLC", "GLE", "EQE"]
CPO_MODELS_ZH = {
    "A-Class": "A 级", "C-Class": "C 级", "E-Class": "E 级", "S-Class": "S 级",
    "GLA": "GLA", "GLC": "GLC", "GLE": "GLE", "EQE": "EQE",
}


@st.cache_data(show_spinner=False)
def generate_dealer_network() -> Dict[str, pd.DataFrame]:
    """Deterministic synthetic Mercedes-style dealer network (illustrative only)."""
    rng = np.random.default_rng(2026)
    rows = []
    for r_i, region in enumerate(REGIONS):
        for d in range(DEALERS_PER_REGION):
            did = f"MB-{region[0]}{d + 1:02d}"
            focus = did == FOCUS_DEALER_ID

            # ── Sales targets & achievement (NC / CPO / Vans tracked separately) ──
            nc_target = int(rng.integers(900, 1700))
            cpo_target = int(rng.integers(380, 760))
            vans_target = int(rng.integers(120, 320))
            nc_achv = float(rng.normal(1.01, 0.10))
            cpo_achv = float(rng.normal(0.97, 0.14))
            vans_achv = float(rng.normal(1.00, 0.16))
            if focus:                       # engineer the story: CPO miss, slow stock
                cpo_achv = 0.78
                nc_achv = 1.02
            nc_achv = float(np.clip(nc_achv, 0.7, 1.35))
            cpo_achv = float(np.clip(cpo_achv, 0.6, 1.35))
            vans_achv = float(np.clip(vans_achv, 0.6, 1.4))
            nc_units = int(nc_target * nc_achv)
            cpo_units = int(cpo_target * cpo_achv)
            vans_units = int(vans_target * vans_achv)
            retail_units = nc_units + cpo_units + vans_units

            cpo_penetration = cpo_units / max(nc_units + cpo_units, 1)
            asp = float(rng.normal(415_000, 38_000))
            gp_per_unit = float(rng.normal(18_500, 3_200))
            discount_rate = float(np.clip(rng.normal(0.083, 0.018), 0.03, 0.16))
            conversion = float(np.clip(rng.normal(0.235, 0.04), 0.12, 0.36))
            days_to_sale = float(np.clip(rng.normal(38, 9), 18, 80))
            days_supply = float(np.clip(rng.normal(62, 14), 30, 130))
            aging_90 = float(np.clip(rng.normal(0.11, 0.045), 0.02, 0.34))
            yoy = float(np.clip(rng.normal(0.046, 0.06), -0.12, 0.20))
            if focus:
                days_supply, aging_90, days_to_sale = 104.0, 0.27, 58.0

            # ── After-sales ──
            as_revenue = float(rng.normal(86_000_000, 12_000_000))
            as_target = as_revenue / float(np.clip(rng.normal(1.0, 0.07), 0.85, 1.18))
            as_gp_margin = float(np.clip(rng.normal(0.42, 0.04), 0.32, 0.52))
            as_gp = as_revenue * as_gp_margin
            ro_volume = int(rng.normal(19_500, 2_800))
            absorption = float(np.clip(rng.normal(0.92, 0.10), 0.6, 1.25))
            workshop_util = float(np.clip(rng.normal(0.81, 0.07), 0.55, 0.97))
            tech_eff = float(np.clip(rng.normal(0.96, 0.06), 0.78, 1.12))
            aro = as_revenue / max(ro_volume, 1)
            parts_per_ro = float(np.clip(rng.normal(3.4, 0.5), 1.8, 5.0))
            parts_fill = float(np.clip(rng.normal(0.93, 0.03), 0.82, 0.99))
            service_retention = float(np.clip(rng.normal(0.71, 0.07), 0.5, 0.9))
            ftf = float(np.clip(rng.normal(0.89, 0.04), 0.74, 0.97))
            repeat_repair = float(np.clip(rng.normal(0.058, 0.02), 0.01, 0.13))
            csi = float(np.clip(rng.normal(88.5, 3.0), 78, 97))
            warranty_mix = float(np.clip(rng.normal(0.34, 0.06), 0.2, 0.5))
            if focus:
                service_retention, csi, absorption = 0.61, 83.0, 0.79

            compliance = float(np.clip(rng.normal(0.95, 0.04), 0.78, 1.0))

            rows.append(dict(
                dealer_id=did, region=region,
                nc_target=nc_target, nc_units=nc_units, nc_achv=nc_achv,
                cpo_target=cpo_target, cpo_units=cpo_units, cpo_achv=cpo_achv,
                vans_target=vans_target, vans_units=vans_units, vans_achv=vans_achv,
                retail_units=retail_units, cpo_penetration=cpo_penetration,
                asp=asp, gp_per_unit=gp_per_unit, discount_rate=discount_rate,
                conversion=conversion, days_to_sale=days_to_sale,
                days_supply=days_supply, aging_90=aging_90, yoy=yoy,
                as_revenue=as_revenue, as_target=as_target, as_gp=as_gp,
                ro_volume=ro_volume, absorption=absorption,
                workshop_util=workshop_util, tech_eff=tech_eff, aro=aro,
                parts_per_ro=parts_per_ro, parts_fill=parts_fill,
                service_retention=service_retention, ftf=ftf,
                repeat_repair=repeat_repair, csi=csi, warranty_mix=warranty_mix,
                compliance=compliance,
            ))
    dealers = pd.DataFrame(rows)

    # ── Composite indices (min-max scaled 0–100 across the network) ──
    def mm(s, invert=False):
        lo, hi = s.min(), s.max()
        z = (s - lo) / (hi - lo) if hi > lo else s * 0 + 0.5
        return (1 - z) * 100 if invert else z * 100

    total_achv = (
        (dealers.nc_units + dealers.cpo_units + dealers.vans_units)
        / (dealers.nc_target + dealers.cpo_target + dealers.vans_target)
    )
    sales_index = (
        0.45 * mm(total_achv) + 0.20 * mm(dealers.gp_per_unit)
        + 0.15 * mm(dealers.conversion) + 0.20 * mm(dealers.days_supply, invert=True)
    )
    aftersales_index = (
        0.35 * mm(dealers.absorption) + 0.20 * mm(dealers.service_retention)
        + 0.20 * mm(dealers.workshop_util) + 0.15 * mm(dealers.ftf)
        + 0.10 * mm(dealers.as_gp)
    )
    cx_index = 0.6 * mm(dealers.csi) + 0.4 * mm(dealers.service_retention)
    compliance_index = mm(dealers.compliance)
    dealers["sales_index"] = sales_index
    dealers["aftersales_index"] = aftersales_index
    dealers["cx_index"] = cx_index
    dealers["compliance_index"] = compliance_index
    dealers["dealer_score"] = (
        0.45 * sales_index + 0.35 * aftersales_index
        + 0.15 * cx_index + 0.05 * compliance_index
    )
    dealers = dealers.sort_values("dealer_score", ascending=False).reset_index(drop=True)
    dealers["rank"] = dealers.index + 1

    # ── Monthly trend for the network (seasonality + noise) ──
    months = pd.date_range("2025-07-01", periods=12, freq="MS")
    season = 1 + 0.12 * np.sin(np.linspace(0, 2 * np.pi, 12))
    m_rows = []
    nat_units = dealers.retail_units.sum() / 12
    nat_as = dealers.as_revenue.sum() / 12
    for i, m in enumerate(months):
        m_rows.append(dict(
            month=m.strftime("%Y-%m"),
            retail_units=int(nat_units * season[i] * rng.normal(1.0, 0.03)),
            as_revenue=float(nat_as * season[i] * rng.normal(1.0, 0.03)),
        ))
    monthly = pd.DataFrame(m_rows)

    # ── CPO inventory for the focus dealer (E-Class overstocked & overpriced) ──
    inv_rng = np.random.default_rng(77)
    base_value = {"A-Class": 195_000, "C-Class": 285_000, "E-Class": 405_000,
                  "S-Class": 720_000, "GLA": 235_000, "GLC": 345_000,
                  "GLE": 525_000, "EQE": 480_000}
    weights = np.array([0.08, 0.16, 0.34, 0.05, 0.10, 0.14, 0.08, 0.05])  # E-Class heavy
    inv = []
    for i in range(140):
        model = inv_rng.choice(CPO_MODELS, p=weights)
        age = float(np.clip(inv_rng.normal(3.2, 1.1), 0.8, 7))
        km = float(np.clip(inv_rng.normal(5.5, 2.2), 0.8, 14))      # 10k km
        # E-Class skews older stock; everything else fresher
        days_in_stock = int(np.clip(inv_rng.normal(95 if model == "E-Class" else 46, 28), 5, 190))
        fair = base_value[model] * max(0.5, 1 - 0.09 * age) * max(0.7, 1 - 0.012 * km)
        fair *= inv_rng.normal(1.0, 0.03)
        # overpricing concentrated in aged E-Class
        if model == "E-Class" and days_in_stock >= 75:
            gap = float(inv_rng.normal(0.14, 0.04))
        else:
            gap = float(inv_rng.normal(0.02, 0.07))
        current = fair * (1 + gap)
        sell_prob = float(np.clip(0.62 - 1.3 * max(gap, 0) - 0.0016 * days_in_stock
                                  + inv_rng.normal(0, 0.05), 0.04, 0.92))
        est_gp = current * 0.07
        if gap > 0.08 and days_in_stock >= 60:
            action = "Reprice"
        elif days_in_stock >= 90 and gap <= 0.08:
            action = "Transfer" if inv_rng.random() < 0.5 else "Promote"
        elif gap < -0.06:
            action = "Raise"
        else:
            action = "Hold"
        inv.append(dict(
            vehicle_id=f"CPO-{i + 1:03d}", model=model, age_years=round(age, 1),
            kilometer=round(km, 1), days_in_stock=days_in_stock,
            current_price=current, fair_price=fair, gap_pct=gap,
            sell_prob_30d=sell_prob, est_gp=est_gp, action=action,
        ))
    cpo_inventory = pd.DataFrame(inv)

    return {"dealers": dealers, "monthly": monthly, "cpo_inventory": cpo_inventory}


def national_kpis(dealers: pd.DataFrame) -> dict:
    units = dealers.retail_units.sum()
    target = (dealers.nc_target + dealers.cpo_target + dealers.vans_target).sum()
    gp = (dealers.gp_per_unit * dealers.retail_units).sum()
    return dict(
        retail_units=int(units),
        sales_achv=units / target,
        yoy=float((dealers.yoy * dealers.retail_units).sum() / units),
        gp_per_unit=gp / units,
        days_supply=float((dealers.days_supply * dealers.retail_units).sum() / units),
        as_revenue=float(dealers.as_revenue.sum()),
        as_achv=float(dealers.as_revenue.sum() / dealers.as_target.sum()),
        ro_volume=int(dealers.ro_volume.sum()),
        retention=float((dealers.service_retention * dealers.ro_volume).sum() / dealers.ro_volume.sum()),
        absorption=float((dealers.absorption * dealers.as_revenue).sum() / dealers.as_revenue.sum()),
    )


# ── CSS ───────────────────────────────────────────────────────────────────────
def add_css() -> None:
    st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
div[data-testid="stMetric"] {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 0.9rem 0.9rem 0.7rem;
}
div[data-testid="stMetric"] label { color: #475569; }
div[data-testid="stMetricValue"] { font-size: 1.5rem; line-height: 1.15; }
.section-note { color: #475569; font-size: 0.95rem; line-height: 1.5; }
.callout { background:#f8fafc; border:1px solid #e2e8f0; border-left:4px solid #2563eb;
    border-radius:0 8px 8px 0; padding:1rem 1.1rem; margin:0.5rem 0 1rem; }
.story-box { background:#f0f4ff; border:1px solid rgba(99,102,241,0.35); border-left:4px solid #6366f1;
    border-radius:0 8px 8px 0; padding:1.1rem 1.3rem; margin:0.5rem 0 1.2rem;
    color:#1e1b4b; line-height:1.8; font-size:0.88rem; }
.finding-overpriced { background:#fef2f2; border:1px solid #fecaca; border-left:4px solid #dc2626;
    border-radius:0 8px 8px 0; padding:1rem 1.2rem; margin:0.5rem 0 1rem; color:#7f1d1d; line-height:1.8; }
.finding-underpriced { background:#eff6ff; border:1px solid #bfdbfe; border-left:4px solid #2563eb;
    border-radius:0 8px 8px 0; padding:1rem 1.2rem; margin:0.5rem 0 1rem; color:#1e3a5f; line-height:1.8; }
.finding-fair { background:#f0fdf4; border:1px solid #bbf7d0; border-left:4px solid #16a34a;
    border-radius:0 8px 8px 0; padding:1rem 1.2rem; margin:0.5rem 0 1rem; color:#14532d; line-height:1.8; }
.warning-callout { background:#fff7ed; border:1px solid #fed7aa; border-left:4px solid #f97316;
    border-radius:0 8px 8px 0; padding:0.6rem 0.9rem; margin:0.3rem 0 0.8rem; font-size:0.82rem; color:#7c2d12; }
.preset-desc { color:#64748b; font-size:0.82rem; font-style:italic; margin-bottom:0.8rem; line-height:1.5; }
.alert-box { background:#fff7ed; border:1px solid #fdba74; border-left:4px solid #ea580c;
    border-radius:0 8px 8px 0; padding:1rem 1.2rem; margin:0.5rem 0 1rem; color:#7c2d12; line-height:1.7; }
</style>
""", unsafe_allow_html=True)


# ── Pricing-model functions (public used-car dataset) ─────────────────────────
def predict_fair_price_mock(brand, model, car_age_years, kilometer, power,
                            fuel_type, gearbox, body_type, not_repaired_damage) -> float:
    brand_anchor = {0: 4200, 1: 8500, 4: 12000, 6: 15000, 10: 18000, 14: 22000, 15: 28000}
    base = brand_anchor.get(brand, 9000 + brand * 120)
    model_factor = 1.0 + (model / 200.0) * 0.15
    age_factor = max(0.45, 1.0 - 0.055 * min(car_age_years, 18))
    km_factor = max(0.55, 1.0 - (kilometer / 15.0) * 0.012)
    power_factor = 0.85 + min(power, 600) / 600 * 0.35
    fuel_factor = 1.02 if fuel_type == 0.0 else 0.98
    gearbox_factor = 1.04 if gearbox == 1.0 else 1.0
    body_factor = 1.0 + body_type * 0.02
    damage_factor = 0.88 if not_repaired_damage == 1.0 else 1.0
    fair = (base * model_factor * age_factor * km_factor * power_factor
            * fuel_factor * gearbox_factor * body_factor * damage_factor)
    return float(np.clip(fair, 800, 120_000))


def pricing_status(price_gap_pct: float) -> str:
    if price_gap_pct > THRESHOLD_PCT:  return "Overpriced"
    if price_gap_pct < -THRESHOLD_PCT: return "Underpriced"
    return "Fair Price"


def recommended_listing_price(fair, status, current_listing) -> float:
    if status == "Overpriced":  return fair * OVERPRICED_ADJ
    if status == "Underpriced": return fair * UNDERPRICED_ADJ
    return current_listing


def dynamic_finding(status, gap_pct, fair_price, current_listing, rec_price, _lang="English") -> str:
    def _t(en, zh): return zh if _lang == "中文" else en
    gap_yuan = abs(current_listing - fair_price)
    css_class = {"Overpriced": "finding-overpriced", "Underpriced": "finding-underpriced",
                 "Fair Price": "finding-fair"}[status]
    if status == "Overpriced":
        body = (f"{_t('This car is listed at','此车挂牌价为')} <b>CNY {current_listing:,.0f}</b> — "
                f"<b>{gap_pct:.1%} {_t('above','高于')}</b> {_t('the estimated fair value of','估算公允价值')} <b>CNY {fair_price:,.0f}</b>.<br>"
                f"{_t('Overpriced by','高估')} <b>CNY {gap_yuan:,.0f}</b>. "
                f"{_t('At this price, the listing will likely sit.','以此价格，挂牌可能长期无人问津。')} "
                f"{_t('Recommendation: reprice to','建议：调价至')} <b>CNY {rec_price:,.0f}</b>.")
    elif status == "Underpriced":
        body = (f"{_t('This car is listed at','此车挂牌价为')} <b>CNY {current_listing:,.0f}</b> — "
                f"<b>{abs(gap_pct):.1%} {_t('below','低于')}</b> {_t('the estimated fair value of','估算公允价值')} <b>CNY {fair_price:,.0f}</b>.<br>"
                f"{_t('Potential upside:','潜在提价空间：')} <b>CNY {gap_yuan:,.0f}</b>. "
                f"{_t('Recommendation: raise to','建议：提价至')} <b>CNY {rec_price:,.0f}</b>.")
    else:
        body = (f"{_t('This car is listed at','此车挂牌价为')} <b>CNY {current_listing:,.0f}</b> — "
                f"{_t('within','在')} <b>±{THRESHOLD_PCT:.0%}</b> {_t('of fair value','公允价值范围内')} <b>CNY {fair_price:,.0f}</b>.<br>"
                f"{_t('Priced to sell. No repricing action needed.','价格合理，无需调价。')}")
    return f'<div class="{css_class}">{body}</div>'


# ── Plot helpers ──────────────────────────────────────────────────────────────
def plot_price_comparison(current_listing, fair_price, recommended, status, txt=None) -> plt.Figure:
    txt = txt or {}
    labels = [txt.get("cur", "Current listing"), txt.get("fair", "Fair value (model)"), txt.get("rec", "Recommended")]
    values = [current_listing, fair_price, recommended]
    status_colors = {"Overpriced": "#dc2626", "Underpriced": "#2563eb", "Fair Price": "#16a34a"}
    colors = [status_colors.get(status, "#888888"), "#16a34a", "#6366f1"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8, width=0.5)
    ax.set_ylabel(txt.get("ylab", "Price (CNY)"), fontsize=10)
    ax.set_title(txt.get("title", "Pricing simulator output"), pad=12, fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15); ax.set_axisbelow(True)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"CNY {val:,.0f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_model_results(title="Pricing model comparison — shared holdout", xlab="Validation MAE (CNY)") -> plt.Figure:
    cd = MODEL_RESULTS.sort_values("Validation MAE", ascending=True)
    colors = ["#16a34a" if m == "Weighted Ensemble" else "#2563eb" for m in cd["Model"]]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(cd["Model"], cd["Validation MAE"], color=colors, height=0.55)
    ax.set_xlabel(xlab)
    ax.set_title(title, pad=10, fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.15); ax.set_axisbelow(True)
    for i, v in enumerate(cd["Validation MAE"]):
        ax.text(v + 3, i, f"{v:.2f}", va="center", fontsize=9.5)
    plt.tight_layout()
    return fig


def plot_monthly(monthly: pd.DataFrame, metric: str, title: str, ylab: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(monthly["month"], monthly[metric], color="#6366f1", linewidth=2.2, marker="o", markersize=4)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_ylabel(ylab)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.12); ax.set_axisbelow(True)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    return fig


def plot_dealer_ranking(dealers: pd.DataFrame, focus_id: str,
                        title="Top dealers by composite score", xlab="Dealer Score (0–100)") -> plt.Figure:
    top = dealers.head(15)
    colors = ["#ea580c" if d == focus_id else "#93c5fd" for d in top["dealer_id"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["dealer_id"][::-1], top["dealer_score"][::-1], color=colors[::-1], height=0.62)
    ax.set_xlabel(xlab)
    ax.set_title(title, fontsize=11, pad=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.15); ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def plot_dealer_radar(row, lang) -> plt.Figure:
    zh = lang == "中文" and CJK_OK
    cats = (["销售", "售后", "客户体验", "合规"] if zh else ["Sales", "After-sales", "CX", "Compliance"])
    title = "经销商指数构成" if zh else "Dealer index breakdown"
    vals = [row.sales_index, row.aftersales_index, row.cx_index, row.compliance_index]
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    vals_c = vals + vals[:1]; angles_c = angles + angles[:1]
    fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw=dict(polar=True))
    ax.plot(angles_c, vals_c, color="#6366f1", linewidth=2)
    ax.fill(angles_c, vals_c, color="#6366f1", alpha=0.18)
    ax.set_xticks(angles); ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylim(0, 100); ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=7, color="#94a3b8")
    ax.set_title(title, fontsize=11, pad=18)
    plt.tight_layout()
    return fig


def figure_exists(filename: str):
    path = FIGURE_DIR / filename
    return path if path.exists() else None


def show_figure_or_note(filename, caption, fallback) -> None:
    path = figure_exists(filename)
    if path:
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.caption(fallback)


# ═══════════════════════════════════════════════════════════════════════════
add_css()
lang = st.sidebar.radio("语言 / Language", ["English", "中文"], horizontal=True)


def t(en: str, zh: str) -> str:
    return zh if lang == "中文" else en


def ct(en: str, zh: str) -> str:
    """Chart text: Chinese only when a CJK font is loaded, else English (no tofu)."""
    return zh if (lang == "中文" and CJK_OK) else en


data = generate_dealer_network()
dealers = data["dealers"]
monthly = data["monthly"]
cpo_inventory = data["cpo_inventory"]
nat = national_kpis(dealers)


# ── Sidebar: CPO single-vehicle pricing inputs ────────────────────────────────
st.sidebar.markdown(f"### 🚗 {t('CPO pricing inputs','CPO 定价输入')}")
st.sidebar.markdown(
    f'<div class="warning-callout">{t("Network figures are synthetic. The single-vehicle pricing model runs on a public used-car dataset (rule-based simulator in this demo).","经销商网络为合成数据。单车定价模型基于公开二手车数据集（本演示用规则模拟器）。")}</div>',
    unsafe_allow_html=True,
)
body_type_names_en = {0: "Sedan", 1: "Hatchback", 2: "SUV", 3: "Estate", 4: "Convertible", 5: "Coupe", 6: "MPV", 7: "Other"}
body_type_names_zh = {0: "轿车", 1: "掀背", 2: "SUV", 3: "旅行", 4: "敞篷", 5: "跑车", 6: "MPV", 7: "其他"}
with st.sidebar.expander(t("Vehicle inputs", "车辆输入"), expanded=False):
    brand = st.selectbox(t("Brand tier (0=budget, 15=luxury)", "品牌档次 (0=经济,15=豪华)"), BRAND_OPTIONS, index=6)
    model = st.selectbox(t("Model code", "车型代码"), MODEL_OPTIONS, index=3)
    body_type_map = body_type_names_zh if lang == "中文" else body_type_names_en
    body_type = st.selectbox(t("Body type", "车身类型"), BODY_TYPE_OPTIONS,
                             format_func=lambda x: body_type_map.get(int(x), str(x)), index=2)
    car_age_years = st.slider(t("Vehicle age (years)", "车龄（年）"), 0.5, 20.0, 5.0, 0.5)
    kilometer = st.slider(t("Mileage (10k km)", "里程（万公里）"), 0.5, 15.0, 5.0, 0.5)
    power = st.slider(t("Engine power (hp)", "发动机功率（马力）"), 50, 600, 160, 10)
    not_repaired_damage = st.selectbox(t("Damage status", "损坏状态"), DAMAGE_OPTIONS,
                                       format_func=lambda x: t("No unrepaired damage", "无未修复损坏") if x == 0 else t("Has unrepaired damage", "有未修复损坏"))
    fuel_type = st.selectbox(t("Fuel type", "燃料类型"), FUEL_TYPE_OPTIONS,
                             format_func=lambda x: t("Petrol / Diesel", "汽油 / 柴油") if x == 0 else t("EV / Hybrid", "纯电 / 混动"))
    gearbox = st.selectbox(t("Gearbox", "变速箱"), GEARBOX_OPTIONS,
                           format_func=lambda x: t("Manual", "手动") if x == 0 else t("Automatic", "自动"))
    current_listing = st.number_input(t("Current listing price (CNY)", "当前挂牌价（元）"),
                                      min_value=500, max_value=200_000, value=12_000, step=500)
inputs = dict(brand=brand, model=model, body_type=body_type, car_age_years=car_age_years,
              kilometer=kilometer, power=power, not_repaired_damage=not_repaired_damage,
              fuel_type=fuel_type, gearbox=gearbox, current_listing=current_listing)
if "preset_inputs" in st.session_state:
    for k, v in st.session_state.pop("preset_inputs").items():
        if k != "intro":
            inputs[k] = v
st.sidebar.divider()
st.sidebar.markdown(f"[GitHub]({GITHUB_URL})")

fair_price = predict_fair_price_mock(int(inputs["brand"]), float(inputs["model"]),
                                     float(inputs["car_age_years"]), float(inputs["kilometer"]),
                                     float(inputs["power"]), float(inputs["fuel_type"]),
                                     float(inputs["gearbox"]), float(inputs["body_type"]),
                                     float(inputs["not_repaired_damage"]))
current_listing = float(inputs["current_listing"])
price_gap = current_listing - fair_price
price_gap_pct = price_gap / fair_price if fair_price > 0 else 0.0
status = pricing_status(price_gap_pct)
rec_price = recommended_listing_price(fair_price, status, current_listing)


# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"🏁 {t('CPO Retail Command Center','CPO 零售指挥中心')}")
st.caption(t("National executive overview · Sales & After-sales KPI · 50-dealer 360 · CPO pricing & inventory action — one decision loop.",
             "全国经营总览 · Sales 与 After-sales KPI · 50 家经销商 360 · CPO 定价与库存行动 —— 一个完整决策闭环。"))

if lang == "中文":
    st.markdown("""<div class="story-box">
<b>📊 产品故事：</b>从全国经营监控，到经销商诊断，再到单车定价行动，构成一个闭环。<br><br>
本作品集把<b>奔驰式经销商分析经验</b>与公开二手车定价模型整合为一个产品：
全国总览发现异常 → Dealer 360 解释 Sales / After-sales 经营差距 →
CPO 定价库存层把洞察落到单车调价、促销和跨店调拨。<br><br>
<b>演示故事线：</b>全国销量达成正常，但 <b>MB-E07</b> 综合排名下滑——CPO 未达标、客户回厂率偏低，
其 E 级 90 天以上库存过多且定价偏高。点开各 Tab 跟随这条线索。
</div>""", unsafe_allow_html=True)
else:
    st.markdown("""<div class="story-box">
<b>📊 The product story:</b> one loop from national monitoring → dealer diagnosis → single-vehicle pricing action.<br><br>
This portfolio piece integrates <b>Mercedes-style dealer-analytics experience</b> with a public used-car
pricing model into a single product: the national overview finds the anomaly → Dealer 360 explains the
Sales / After-sales gap → the CPO pricing & inventory layer turns insight into per-car reprice, promote, or transfer.<br><br>
<b>Demo storyline:</b> national sales achievement is on track, yet <b>MB-E07</b> slips in the ranking — CPO target missed,
service retention low, and its E-Class 90+ day stock is overpriced. Follow that thread across the tabs.
</div>""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_labels = (
    ["🏠 全国总览", "📈 Sales KPI", "🔧 After-sales KPI", "🏆 Dealer 360",
     "🚗 CPO 定价与库存", "🤖 AI 问数", "ℹ️ 关于与方法"]
    if lang == "中文" else
    ["🏠 Executive Overview", "📈 Sales KPI", "🔧 After-sales KPI", "🏆 Dealer 360",
     "🚗 CPO Pricing & Inventory", "🤖 AI Narrative", "ℹ️ About & Method"]
)
(exec_tab, sales_tab, aftersales_tab, dealer_tab,
 cpo_tab, ai_tab, about_tab) = st.tabs(tab_labels)


# ════════════════════════ TAB 1 — EXECUTIVE OVERVIEW ═════════════════════════
with exec_tab:
    st.subheader(t("National headline KPIs", "全国首屏指标"))
    st.caption(t("10 metrics management watches first — Sales (left) and After-sales (right).",
                 "管理层最先看的 10 个指标 —— 左侧 Sales，右侧 After-sales。"))
    st.markdown(f"**{t('Sales','销售')}**")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric(t("Retail Sales Units", "零售销量"), f"{nat['retail_units']:,}")
    s2.metric(t("Sales Target Achievement", "销售目标达成"), f"{nat['sales_achv']:.1%}")
    s3.metric(t("YoY Growth", "同比增长"), f"{nat['yoy']:+.1%}")
    s4.metric(t("Gross Profit / Unit", "单车毛利"), f"CNY {nat['gp_per_unit']:,.0f}")
    s5.metric(t("Inventory Days Supply", "库存天数"), f"{nat['days_supply']:.0f}")
    st.markdown(f"**{t('After-sales','售后')}**")
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric(t("After-sales Revenue", "售后收入"), f"CNY {nat['as_revenue']/1e9:,.2f}B")
    a2.metric(t("AS Revenue Achievement", "售后目标达成"), f"{nat['as_achv']:.1%}")
    a3.metric(t("Repair Order Volume", "工单量"), f"{nat['ro_volume']:,}")
    a4.metric(t("Service Retention", "客户回厂率"), f"{nat['retention']:.1%}")
    a5.metric(t("Service Absorption", "吸收率"), f"{nat['absorption']:.0%}")

    st.markdown(f'<div class="alert-box">⚠️ {t("Alert: national sales on track, but dealer <b>MB-E07</b> dropped to rank "+str(int(dealers.loc[dealers.dealer_id==FOCUS_DEALER_ID,"rank"].iloc[0]))+" of 50. Drill into Sales → After-sales → Dealer 360 → CPO to see why.","预警：全国销量达成正常，但经销商 <b>MB-E07</b> 综合排名跌至第 "+str(int(dealers.loc[dealers.dealer_id==FOCUS_DEALER_ID,"rank"].iloc[0]))+"/50。沿 Sales → After-sales → Dealer 360 → CPO 下钻查看原因。")}</div>',
                unsafe_allow_html=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_monthly(monthly, "retail_units", ct("Retail units — last 12 months", "零售销量 — 近12个月"), ct("Units", "台")))
    with c2:
        st.pyplot(plot_monthly(monthly, "as_revenue", ct("After-sales revenue — last 12 months", "售后收入 — 近12个月"), "CNY"))

    st.markdown(f"#### {t('Drill-down entries','下钻入口')}")
    d1, d2, d3 = st.columns(3)
    d1.markdown(f'<div class="callout"><b>📈 Sales</b><br>{t("NC · CPO · Vans target achievement, pricing, conversion and inventory aging.","NC · CPO · Vans 目标达成、定价、转化与库存老化。")}</div>', unsafe_allow_html=True)
    d2.markdown(f'<div class="callout" style="border-left-color:#16a34a"><b>🔧 After-sales</b><br>{t("Revenue, absorption, workshop efficiency, retention and quality.","收入、吸收率、工位效率、客户回厂率与质量。")}</div>', unsafe_allow_html=True)
    d3.markdown(f'<div class="callout" style="border-left-color:#f59e0b"><b>🏆 Dealer Network</b><br>{t("Region · dealer · model — composite score and root-cause.","区域 · 经销商 · 车型 —— 综合评分与根因。")}</div>', unsafe_allow_html=True)


# ════════════════════════════ TAB 2 — SALES KPI ══════════════════════════════
with sales_tab:
    st.subheader(t("Sales KPI system", "Sales KPI 体系"))
    st.caption(t("NC, CPO and Vans tracked separately — different targets, mix and inventory logic.",
                 "NC、CPO、Vans 分别计算 —— 业务目标、车型结构与库存逻辑不同。"))

    nc_achv = dealers.nc_units.sum() / dealers.nc_target.sum()
    cpo_achv = dealers.cpo_units.sum() / dealers.cpo_target.sum()
    vans_achv = dealers.vans_units.sum() / dealers.vans_target.sum()
    cpo_pen = dealers.cpo_units.sum() / (dealers.nc_units.sum() + dealers.cpo_units.sum())
    asp = float((dealers.asp * dealers.retail_units).sum() / dealers.retail_units.sum())
    disc = float((dealers.discount_rate * dealers.retail_units).sum() / dealers.retail_units.sum())
    conv = float(dealers.conversion.mean())
    dts = float((dealers.days_to_sale * dealers.retail_units).sum() / dealers.retail_units.sum())
    dsupply = nat["days_supply"]
    aging = float((dealers.aging_90 * dealers.retail_units).sum() / dealers.retail_units.sum())

    r1 = st.columns(5)
    r1[0].metric(t("NC Target Achievement", "NC 目标达成"), f"{nc_achv:.1%}")
    r1[1].metric(t("CPO Target Achievement", "CPO 目标达成"), f"{cpo_achv:.1%}")
    r1[2].metric(t("Vans Target Achievement", "Vans 目标达成"), f"{vans_achv:.1%}")
    r1[3].metric(t("CPO Penetration", "CPO 渗透率"), f"{cpo_pen:.1%}")
    r1[4].metric(t("Average Selling Price", "平均成交价"), f"CNY {asp:,.0f}")
    r2 = st.columns(5)
    r2[0].metric(t("Discount Rate", "折扣率"), f"{disc:.1%}")
    r2[1].metric(t("Lead-to-Sale Conversion", "线索成交转化"), f"{conv:.1%}")
    r2[2].metric(t("Days to Sale", "成交天数"), f"{dts:.0f}")
    r2[3].metric(t("Days Supply", "库存天数"), f"{dsupply:.0f}")
    r2[4].metric(t("90+ Days Aging Rate", "90天以上库存率"), f"{aging:.1%}")

    st.divider()
    st.markdown(f"#### {t('NC / CPO / Vans line summary','NC / CPO / Vans 业务线汇总')}")
    line_tbl = pd.DataFrame({
        t("Line", "业务线"): ["NC", "CPO", "Vans"],
        t("Units", "销量"): [dealers.nc_units.sum(), dealers.cpo_units.sum(), dealers.vans_units.sum()],
        t("Target", "目标"): [dealers.nc_target.sum(), dealers.cpo_target.sum(), dealers.vans_target.sum()],
        t("Achievement", "达成率"): [f"{nc_achv:.1%}", f"{cpo_achv:.1%}", f"{vans_achv:.1%}"],
    })
    st.dataframe(line_tbl, hide_index=True, use_container_width=True)

    st.markdown(f"#### {t('Sales by region','分区域销售')}")
    reg = dealers.groupby("region").agg(
        retail_units=("retail_units", "sum"),
        cpo_units=("cpo_units", "sum"),
        cpo_target=("cpo_target", "sum"),
        asp=("asp", "mean"),
        days_supply=("days_supply", "mean"),
        aging_90=("aging_90", "mean"),
    ).reindex(REGIONS).reset_index()
    reg_disp = pd.DataFrame({
        t("Region", "区域"): [REGIONS_ZH[r] if lang == "中文" else r for r in reg["region"]],
        t("Retail Units", "零售销量"): reg["retail_units"].map(lambda x: f"{x:,}"),
        t("CPO Achievement", "CPO 达成"): (reg["cpo_units"] / reg["cpo_target"]).map(lambda x: f"{x:.1%}"),
        t("Avg ASP", "平均成交价"): reg["asp"].map(lambda x: f"CNY {x:,.0f}"),
        t("Days Supply", "库存天数"): reg["days_supply"].map(lambda x: f"{x:.0f}"),
        t("90+ Aging", "90天+库存"): reg["aging_90"].map(lambda x: f"{x:.1%}"),
    })
    st.dataframe(reg_disp, hide_index=True, use_container_width=True)
    st.markdown(f'<div class="callout">{t("Design principle: NC / CPO / Vans achievement are computed separately, never summed before comparison. East region carries the CPO shortfall driven by MB-E07.","设计原则：NC / CPO / Vans 达成率分别计算，不简单相加后比较。华东区的 CPO 缺口主要来自 MB-E07。")}</div>', unsafe_allow_html=True)


# ═══════════════════════════ TAB 3 — AFTER-SALES KPI ═════════════════════════
with aftersales_tab:
    st.subheader(t("After-sales KPI system", "After-sales KPI 体系"))
    st.caption(t("Only metrics that can change a dealer's action — result, efficiency, customer & quality.",
                 "只选择能改变门店动作的指标 —— 结果、效率、客户与质量。"))

    as_rev = dealers.as_revenue.sum()
    as_gp = dealers.as_gp.sum()
    absn = nat["absorption"]
    wu = float((dealers.workshop_util * dealers.ro_volume).sum() / dealers.ro_volume.sum())
    te = float((dealers.tech_eff * dealers.ro_volume).sum() / dealers.ro_volume.sum())
    aro = float(as_rev / dealers.ro_volume.sum())
    pf = float((dealers.parts_fill * dealers.ro_volume).sum() / dealers.ro_volume.sum())
    ret = nat["retention"]
    ftf = float((dealers.ftf * dealers.ro_volume).sum() / dealers.ro_volume.sum())
    csi = float((dealers.csi * dealers.ro_volume).sum() / dealers.ro_volume.sum())

    r1 = st.columns(5)
    r1[0].metric(t("After-sales Revenue", "售后收入"), f"CNY {as_rev/1e9:,.2f}B")
    r1[1].metric(t("After-sales Gross Profit", "售后毛利"), f"CNY {as_gp/1e9:,.2f}B")
    r1[2].metric(t("Service Absorption", "吸收率"), f"{absn:.0%}")
    r1[3].metric(t("Workshop Utilization", "工位利用率"), f"{wu:.0%}")
    r1[4].metric(t("Technician Efficiency", "技师效率"), f"{te:.0%}")
    r2 = st.columns(5)
    r2[0].metric(t("Avg Repair Order Value", "平均客单价"), f"CNY {aro:,.0f}")
    r2[1].metric(t("Parts Fill Rate", "零件满足率"), f"{pf:.0%}")
    r2[2].metric(t("Service Retention", "客户回厂率"), f"{ret:.1%}")
    r2[3].metric(t("First-Time Fix Rate", "一次修复率"), f"{ftf:.0%}")
    r2[4].metric(t("CSI / Service NPS", "满意度 CSI"), f"{csi:.1f}")

    st.markdown(f'<div class="warning-callout">{t("Service Absorption = after-sales gross profit ÷ dealer fixed operating cost. Fixed-cost is synthetic here; in a real deployment it is shown only when a credible expense field exists.","Service Absorption = 售后毛利 ÷ 经销商固定运营费用。此处费用为合成；真实部署中只有存在可靠费用字段时才展示。")}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown(f"#### {t('After-sales KPI by region','分区域售后指标')}")
    reg = dealers.groupby("region").agg(
        as_revenue=("as_revenue", "sum"),
        absorption=("absorption", "mean"),
        workshop_util=("workshop_util", "mean"),
        service_retention=("service_retention", "mean"),
        ftf=("ftf", "mean"),
        csi=("csi", "mean"),
    ).reindex(REGIONS).reset_index()
    reg_disp = pd.DataFrame({
        t("Region", "区域"): [REGIONS_ZH[r] if lang == "中文" else r for r in reg["region"]],
        t("AS Revenue", "售后收入"): reg["as_revenue"].map(lambda x: f"CNY {x/1e6:,.0f}M"),
        t("Absorption", "吸收率"): reg["absorption"].map(lambda x: f"{x:.0%}"),
        t("Workshop Util", "工位利用率"): reg["workshop_util"].map(lambda x: f"{x:.0%}"),
        t("Retention", "回厂率"): reg["service_retention"].map(lambda x: f"{x:.1%}"),
        t("First-Time Fix", "一次修复"): reg["ftf"].map(lambda x: f"{x:.0%}"),
        t("CSI", "满意度"): reg["csi"].map(lambda x: f"{x:.1f}"),
    })
    st.dataframe(reg_disp, hide_index=True, use_container_width=True)
    st.markdown(f'<div class="callout">{t("East region trails on retention and CSI — the same MB-E07 weakness that shows up in Dealer 360 and feeds the CPO action list.","华东区在回厂率与满意度上落后 —— 与 Dealer 360 中 MB-E07 的短板一致，并最终汇入 CPO 行动清单。")}</div>', unsafe_allow_html=True)


# ════════════════════════════ TAB 4 — DEALER 360 ═════════════════════════════
with dealer_tab:
    st.subheader(t("Dealer 360 — composite score & root cause", "Dealer 360 —— 综合评分与根因"))
    st.markdown(f'<div class="callout">{t("Dealer Score = 45% Sales + 35% After-sales + 15% Customer Experience + 5% Compliance. The composite ranks dealers; the raw indices explain why.","Dealer Score = 45% Sales + 35% After-sales + 15% 客户体验 + 5% 合规。综合分用于排名，原始指数解释原因。")}</div>', unsafe_allow_html=True)

    lc, rc = st.columns([1.1, 1])
    with lc:
        st.pyplot(plot_dealer_ranking(dealers, FOCUS_DEALER_ID,
                                      title=ct("Top dealers by composite score", "经销商综合评分排名"),
                                      xlab=ct("Dealer Score (0–100)", "综合分 (0–100)")))
    with rc:
        st.markdown(f"#### {t('Full ranking (50 dealers)','完整排名（50 家）')}")
        rank_tbl = pd.DataFrame({
            t("Rank", "排名"): dealers["rank"],
            t("Dealer", "经销商"): dealers["dealer_id"],
            t("Region", "区域"): [REGIONS_ZH[r] if lang == "中文" else r for r in dealers["region"]],
            t("Score", "综合分"): dealers["dealer_score"].map(lambda x: f"{x:.1f}"),
            t("Sales", "销售"): dealers["sales_index"].map(lambda x: f"{x:.0f}"),
            t("After-sales", "售后"): dealers["aftersales_index"].map(lambda x: f"{x:.0f}"),
            t("CX", "体验"): dealers["cx_index"].map(lambda x: f"{x:.0f}"),
        })
        st.dataframe(rank_tbl, hide_index=True, use_container_width=True, height=360)

    st.divider()
    default_idx = int(dealers.index[dealers.dealer_id == FOCUS_DEALER_ID][0])
    sel = st.selectbox(t("Select a dealer to diagnose", "选择经销商进行诊断"),
                       dealers["dealer_id"].tolist(), index=default_idx)
    row = dealers[dealers.dealer_id == sel].iloc[0]

    st.markdown(f"#### {t('Scorecard','记分卡')} — {sel} ({REGIONS_ZH[row.region] if lang=='中文' else row.region}, {t('rank','排名')} {int(row['rank'])}/50)")
    m = st.columns(4)
    m[0].metric(t("Sales Index", "销售指数"), f"{row.sales_index:.0f}")
    m[1].metric(t("After-sales Index", "售后指数"), f"{row.aftersales_index:.0f}")
    m[2].metric(t("CX Index", "体验指数"), f"{row.cx_index:.0f}")
    m[3].metric(t("Dealer Score", "综合分"), f"{row.dealer_score:.1f}")

    g1, g2 = st.columns([1, 1.2])
    with g1:
        st.pyplot(plot_dealer_radar(row, lang))
    with g2:
        st.markdown(f"#### {t('Raw KPI drill','原始 KPI 下钻')}")
        drill = pd.DataFrame({
            t("KPI", "指标"): [t("NC Achievement", "NC 达成"), t("CPO Achievement", "CPO 达成"),
                               t("Days Supply", "库存天数"), t("90+ Aging", "90天+库存"),
                               t("Service Retention", "客户回厂率"), t("Absorption", "吸收率"),
                               t("Workshop Util", "工位利用率"), t("CSI", "满意度")],
            t("Value", "数值"): [f"{row.nc_achv:.1%}", f"{row.cpo_achv:.1%}", f"{row.days_supply:.0f}",
                                f"{row.aging_90:.1%}", f"{row.service_retention:.1%}", f"{row.absorption:.0%}",
                                f"{row.workshop_util:.0%}", f"{row.csi:.1f}"],
        })
        st.dataframe(drill, hide_index=True, use_container_width=True)

    # automatic diagnosis
    issues_en, issues_zh = [], []
    if row.cpo_achv < 0.9:
        issues_en.append("CPO sales below target"); issues_zh.append("CPO 销量未达标")
    if row.aging_90 > 0.18 or row.days_supply > 85:
        issues_en.append("CPO inventory aging / overstock"); issues_zh.append("CPO 库存老化、积压")
    if row.service_retention < 0.66:
        issues_en.append("after-sales customer retention low"); issues_zh.append("售后客户回厂率偏低")
    if row.workshop_util < 0.7:
        issues_en.append("workshop utilization weak"); issues_zh.append("工位利用率不足")
    if row.csi < 85:
        issues_en.append("service satisfaction declining"); issues_zh.append("服务满意度下降")
    if not issues_en:
        issues_en.append("no material weakness — balanced performer"); issues_zh.append("无明显短板，表现均衡")
    diag = "；".join(issues_zh) if lang == "中文" else "; ".join(issues_en)
    st.markdown(f'<div class="alert-box"><b>{t("Diagnosis","诊断")}:</b> {diag}.</div>', unsafe_allow_html=True)


# ═══════════════════════ TAB 5 — CPO PRICING & INVENTORY ═════════════════════
with cpo_tab:
    st.subheader(t("CPO Pricing & Inventory Copilot", "CPO 定价与库存副驾"))
    st.caption("150K listings · XGBoost / CatBoost / LightGBM · Weighted ensemble MAE = CNY 496.83")
    st.markdown(f'<div class="callout">{t("This is the action layer: the used-car pricing model scores each vehicle, and aged / overpriced stock turns into reprice, promote or transfer actions.","这是行动层：二手车定价模型为每辆车打分，把老化、定价偏高的库存转化为调价、促销或跨店调拨。")}</div>', unsafe_allow_html=True)

    # ── A) Single-vehicle pricing simulator (public dataset model) ──
    st.markdown(f"#### {t('A. Single-vehicle pricing simulator','A. 单车定价模拟器')}")
    st.markdown(f"**{t('Load a scenario:','加载场景：')}**")
    p_cols = st.columns(3)
    for col, (label, preset) in zip(p_cols, CAR_PRESETS.items()):
        with col:
            if st.button(label, use_container_width=True):
                p_copy = dict(preset); p_copy.pop("intro", None)
                st.session_state["preset_inputs"] = p_copy
                st.rerun()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(t("Fair value (model)", "公允价值（模型）"), f"CNY {fair_price:,.0f}")
    col2.metric(t("Current listing", "当前挂牌价"), f"CNY {current_listing:,.0f}")
    col3.metric(t("Recommended", "建议价格"), f"CNY {rec_price:,.0f}")
    delta_color = "inverse" if status == "Overpriced" else "normal" if status == "Underpriced" else "off"
    status_label = {"Overpriced": t("Overpriced", "高估"), "Underpriced": t("Underpriced", "低估"), "Fair Price": t("Fair Price", "公允")}.get(status, status)
    col4.metric(t("Gap vs fair value", "与公允价值差距"), f"{price_gap_pct:+.1%}", status_label, delta_color=delta_color)
    st.markdown(dynamic_finding(status, price_gap_pct, fair_price, current_listing, rec_price, lang), unsafe_allow_html=True)
    cc, gc = st.columns([1.3, 1])
    with cc:
        st.pyplot(plot_price_comparison(current_listing, fair_price, rec_price, status, txt={
            "cur": ct("Current listing", "当前挂牌价"), "fair": ct("Fair value (model)", "公允价值"),
            "rec": ct("Recommended", "建议价格"), "ylab": ct("Price (CNY)", "价格（元）"),
            "title": ct("Pricing simulator output", "定价模拟器输出"),
        }))
    with gc:
        st.markdown(f"#### {t('Pricing rules','定价规则')}")
        st.dataframe(pd.DataFrame({
            t("Status", "状态"): [t("Overpriced", "高估"), t("Underpriced", "低估"), t("Fair Price", "公允")],
            t("Condition", "条件"): [f"> {THRESHOLD_PCT:.0%} {t('above fair','高于公允')}", f"> {THRESHOLD_PCT:.0%} {t('below fair','低于公允')}", f"{t('Within','在')} ±{THRESHOLD_PCT:.0%}"],
            t("Action", "操作"): [f"{t('fair ×','公允 ×')} {OVERPRICED_ADJ}", f"{t('fair ×','公允 ×')} {UNDERPRICED_ADJ}", t("Keep listing", "维持挂牌")],
        }), hide_index=True, use_container_width=True)
        st.pyplot(plot_model_results(title=ct("Pricing model comparison — shared holdout", "定价模型比较 — 同一验证集"),
                                     xlab=ct("Validation MAE (CNY)", "验证集 MAE（元）")))

    st.divider()
    # ── B) Dealer CPO inventory action list (focus dealer) ──
    st.markdown(f"#### {t('B. CPO inventory & repricing actions','B. CPO 库存与调价行动')} — {FOCUS_DEALER_ID}")
    aged = cpo_inventory[cpo_inventory.days_in_stock >= 90]
    eclass_aged = aged[aged.model == "E-Class"]
    margin_at_risk = float((aged[aged.gap_pct > 0.08].current_price * 0.05).sum())
    k = st.columns(4)
    k[0].metric(t("Total CPO units", "CPO 在库总数"), f"{len(cpo_inventory)}")
    k[1].metric(t("Units 90+ days", "90天+库存"), f"{len(aged)}", f"{len(aged)/len(cpo_inventory):.0%}")
    k[2].metric(t("E-Class 90+ days", "E级 90天+"), f"{len(eclass_aged)}")
    k[3].metric(t("Est. margin at risk", "预计风险毛利"), f"CNY {margin_at_risk/1e6:,.2f}M")

    model_filter = st.multiselect(t("Filter by model", "按车型筛选"),
                                  [CPO_MODELS_ZH[m] if lang == "中文" else m for m in CPO_MODELS],
                                  default=[])
    inv = cpo_inventory.copy()
    if model_filter:
        wanted = {m for m in CPO_MODELS if (CPO_MODELS_ZH[m] if lang == "中文" else m) in model_filter}
        inv = inv[inv.model.isin(wanted)]
    inv = inv.sort_values(["action", "days_in_stock"], ascending=[True, False])
    action_map = {"Reprice": t("Reprice", "调价"), "Promote": t("Promote", "促销"),
                  "Transfer": t("Transfer", "跨店调拨"), "Raise": t("Raise", "提价"), "Hold": t("Hold", "维持")}
    inv_disp = pd.DataFrame({
        t("Vehicle", "车辆"): inv.vehicle_id,
        t("Model", "车型"): inv.model.map(lambda m: CPO_MODELS_ZH[m] if lang == "中文" else m),
        t("Age (yr)", "车龄"): inv.age_years,
        t("Days in stock", "在库天数"): inv.days_in_stock,
        t("Current", "现价"): inv.current_price.map(lambda x: f"{x/1e4:,.1f}万"),
        t("Fair (model)", "公允价"): inv.fair_price.map(lambda x: f"{x/1e4:,.1f}万"),
        t("Gap", "价差"): inv.gap_pct.map(lambda x: f"{x:+.1%}"),
        t("Sell prob 30d", "30天售出概率"): inv.sell_prob_30d.map(lambda x: f"{x:.0%}"),
        t("Action", "行动"): inv.action.map(lambda a: action_map[a]),
    })
    st.dataframe(inv_disp, hide_index=True, use_container_width=True, height=380)

    rep = (inv.action == "Reprice").sum(); pro = (inv.action == "Promote").sum()
    tr = (inv.action == "Transfer").sum()
    st.markdown(f'<div class="story-box">{t("Combined action plan for MB-E07","MB-E07 组合行动建议")}: '
                f'<b>{rep}</b> {t("vehicles to reprice","辆调价")}, <b>{pro}</b> {t("to promote","辆促销")}, '
                f'<b>{tr}</b> {t("to transfer cross-store","辆跨店调拨")}. '
                f'{t("E-Class 90+ day stock is the largest overpriced cluster — reprice first, then bundle a service package to lift retention.","E 级 90 天以上库存是最大的高估集群 —— 先调价，再配合保养套餐提升回厂率。")}</div>',
                unsafe_allow_html=True)


# ═══════════════════════════ TAB 6 — AI NARRATIVE ════════════════════════════
with ai_tab:
    st.subheader(t("AI Narrative / ChatBI", "AI 问数 / ChatBI"))
    st.caption(t("Ask a business question in natural language; the assistant answers from the synthetic KPI layer.",
                 "用自然语言提问；助手基于合成 KPI 层回答。"))
    questions_en = [
        "Why did MB-E07 drop in the ranking?",
        "Which region has the weakest CPO achievement?",
        "What is the biggest after-sales risk right now?",
        "What action should we take on aged CPO inventory?",
    ]
    questions_zh = [
        "MB-E07 为什么排名下滑？",
        "哪个区域 CPO 达成最弱？",
        "当前最大的售后风险是什么？",
        "老化的 CPO 库存应该采取什么行动？",
    ]
    qs = questions_zh if lang == "中文" else questions_en
    q = st.radio(t("Pick a question", "选择一个问题"), qs, index=0)
    idx = qs.index(q)

    focus = dealers[dealers.dealer_id == FOCUS_DEALER_ID].iloc[0]
    weak_region = dealers.groupby("region").apply(
        lambda g: g.cpo_units.sum() / g.cpo_target.sum()).idxmin()
    aged = cpo_inventory[cpo_inventory.days_in_stock >= 90]
    answers_en = [
        f"MB-E07 sits at rank {int(focus['rank'])}/50. Sales index is fine on NC ({focus.nc_achv:.0%}) but CPO achievement is only {focus.cpo_achv:.0%}, days-supply is {focus.days_supply:.0f} and 90+ aging is {focus.aging_90:.0%}. After-sales drags it further: retention {focus.service_retention:.0%} and CSI {focus.csi:.0f}. The composite weighting (45/35/15/5) turns those two gaps into a visible rank drop.",
        f"The {weak_region} region has the lowest CPO achievement, pulled down by MB-E07's {focus.cpo_achv:.0%}. NC and Vans are on track there, which is why the national number still looks healthy.",
        f"Service retention and CSI at MB-E07 ({focus.service_retention:.0%} / {focus.csi:.0f}) are the clearest after-sales risk. Customers who bought there are not coming back for service, which compounds the CPO inventory problem.",
        f"{len(aged)} CPO units are 90+ days in stock, concentrated in E-Class. Reprice the overpriced ones toward model fair value, promote the rest, and transfer slow movers to higher-demand stores. Estimated margin at risk is meaningful but recoverable with early action.",
    ]
    answers_zh = [
        f"MB-E07 排名第 {int(focus['rank'])}/50。NC 达成正常（{focus.nc_achv:.0%}），但 CPO 仅 {focus.cpo_achv:.0%}，库存天数 {focus.days_supply:.0f}，90 天以上库存率 {focus.aging_90:.0%}。售后进一步拖累：回厂率 {focus.service_retention:.0%}、满意度 {focus.csi:.0f}。在 45/35/15/5 的综合权重下，这两个缺口直接拉低了综合排名。",
        f"{REGIONS_ZH.get(weak_region, weak_region)}区 CPO 达成最弱，主要被 MB-E07 的 {focus.cpo_achv:.0%} 拉低。该区 NC 与 Vans 正常，所以全国数字看起来仍健康。",
        f"MB-E07 的回厂率与满意度（{focus.service_retention:.0%} / {focus.csi:.0f}）是最明确的售后风险。在该店购车的客户没有回厂保养，这又加剧了 CPO 库存问题。",
        f"共有 {len(aged)} 辆 CPO 车在库超过 90 天，集中在 E 级。把定价偏高的调向模型公允价、其余促销、滞销车跨店调拨到高需求门店。预计风险毛利可观，但及早行动可挽回。",
    ]
    ans = (answers_zh if lang == "中文" else answers_en)[idx]
    st.markdown(f'<div class="story-box">🤖 {ans}</div>', unsafe_allow_html=True)
    st.caption(t("Demo note: answers are templated over the synthetic KPI layer, not a live LLM call.",
                 "演示说明：回答基于合成 KPI 层的模板生成，非实时大模型调用。"))


# ═══════════════════════════ TAB 7 — ABOUT & METHOD ══════════════════════════
with about_tab:
    st.subheader(t("About this demo", "关于本演示"))
    if lang == "中文":
        st.markdown("""<div class="callout">
<b>这是什么：</b>一个作品集产品，把我在<b>奔驰（梅赛德斯-奔驰）经销商数据分析</b>中的经验，
与一个公开数据集上的二手车动态定价模型，整合成一个完整的零售经营决策系统。<br><br>
<b>我在奔驰做过什么 → 所以这样设计：</b><br>
• 做过全国 50+ 经销商 KPI 看板 → 这里用<b>全国总览 + 10 个首屏指标</b>再现管理层视角；<br>
• 做过 Sales 与 After-sales 经营分析 → 这里把两者拆成<b>各自的 KPI 体系</b>，NC/CPO/Vans 分别核算；<br>
• 做过定价与库存分析 → 这里用 <b>CPO 定价库存副驾</b>把模型落到单车调价、促销、调拨行动；<br>
• 关注经销商综合健康度 → 这里用 <b>Dealer Score（45/35/15/5）</b>排名并下钻根因。
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="callout">
<b>What this is:</b> a portfolio product that combines my experience in
<b>Mercedes-Benz dealer data analytics</b> with a used-car dynamic-pricing model trained on a public
dataset, integrated into one retail decision system.<br><br>
<b>What I did at Mercedes → so I designed it this way:</b><br>
• Built KPI dashboards for 50+ dealers nationally → reproduced here as the <b>Executive Overview + 10 headline KPIs</b>;<br>
• Ran Sales and After-sales operating analysis → split here into <b>separate KPI systems</b>, with NC/CPO/Vans tracked apart;<br>
• Worked on pricing & inventory analysis → landed here as the <b>CPO Pricing & Inventory Copilot</b> turning the model into per-car actions;<br>
• Monitored overall dealer health → captured here as the <b>Dealer Score (45/35/15/5)</b> ranking with root-cause drill-down.
</div>""", unsafe_allow_html=True)

    st.markdown(f"#### {t('Dealer Score formula','Dealer Score 公式')}")
    st.code("Dealer Score = 0.45·Sales Index + 0.35·After-sales Index + 0.15·CX Index + 0.05·Compliance Index\n"
            "(each index min-max scaled 0–100 across the 50-dealer network)", language="text")

    st.markdown(f"#### {t('Pricing model methodology','定价模型方法论')}")
    st.code("""
raw listings (150K rows, 49 features)
  → EDA & data quality
  → feature engineering (age, usage, categorical, aggregates)
  → XGBoost  514.23  ·  CatBoost  501.67 (best single)  ·  LightGBM  591.00
  → weighted ensemble  496.83  ← production anchor
  → fair-value estimate → pricing flag → reprice recommendation
""".strip(), language="text")

    st.markdown(f"#### {t('Evidence boundary & caveats','证据边界与声明')}")
    if lang == "中文":
        st.markdown("""
本作品集**不使用任何真实奔驰数据**。所有经销商网络数字、目标、客户、车辆、VIN 均为合成且确定性生成，
仅用于演示指标体系与决策闭环。Dealer Score 权重、车型组合与 KPI 公式是透明的作品集设计假设，
**不是奔驰官方定义**。E/C/A/S 级等车型名称仅为示意。单车定价模型基于公开二手车数据集，
公开应用使用规则模拟器以保持仓库轻量。
""")
    else:
        st.markdown("""
This portfolio uses **no real Mercedes-Benz data**. All dealer-network figures, targets, customers,
vehicles and VINs are synthetic and deterministically generated, purely to demonstrate the KPI system
and decision loop. Dealer Score weights, model mix and KPI formulas are transparent portfolio design
assumptions, **not official Mercedes-Benz definitions**. Model names (E/C/A/S-Class, etc.) are illustrative.
The single-vehicle pricing model is trained on a public used-car dataset; the public app uses a rule-based
simulator to keep the repo lightweight.
""")
    st.code("streamlit run app/streamlit_app.py", language="bash")
