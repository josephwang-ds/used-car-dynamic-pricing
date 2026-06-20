"""Generate the synthetic Mercedes-style dealer network used by the CPO Retail
Command Center demo, and export it to data/synthetic/ as CSV + a data dictionary.

The logic mirrors `generate_dealer_network()` in app/streamlit_app.py exactly
(same seed = 2026), so the exported tables match what the app renders.

NO real Mercedes-Benz data is used. Everything here is synthetic, deterministic,
and for portfolio demonstration only. Run:

    python scripts/generate_dealer_network.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic"

REGIONS = ["North", "East", "South", "West", "Central"]
DEALERS_PER_REGION = 10
FOCUS_DEALER_ID = "MB-E07"
CPO_MODELS = ["A-Class", "C-Class", "E-Class", "S-Class", "GLA", "GLC", "GLE", "EQE"]


def build() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(2026)
    rows = []
    for region in REGIONS:
        for d in range(DEALERS_PER_REGION):
            did = f"MB-{region[0]}{d + 1:02d}"
            focus = did == FOCUS_DEALER_ID

            nc_target = int(rng.integers(900, 1700))
            cpo_target = int(rng.integers(380, 760))
            vans_target = int(rng.integers(120, 320))
            nc_achv = float(rng.normal(1.01, 0.10))
            cpo_achv = float(rng.normal(0.97, 0.14))
            vans_achv = float(rng.normal(1.00, 0.16))
            if focus:
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

    def mm(s, invert=False):
        lo, hi = s.min(), s.max()
        z = (s - lo) / (hi - lo) if hi > lo else s * 0 + 0.5
        return (1 - z) * 100 if invert else z * 100

    total_achv = ((dealers.nc_units + dealers.cpo_units + dealers.vans_units)
                  / (dealers.nc_target + dealers.cpo_target + dealers.vans_target))
    sales_index = (0.45 * mm(total_achv) + 0.20 * mm(dealers.gp_per_unit)
                   + 0.15 * mm(dealers.conversion) + 0.20 * mm(dealers.days_supply, invert=True))
    aftersales_index = (0.35 * mm(dealers.absorption) + 0.20 * mm(dealers.service_retention)
                        + 0.20 * mm(dealers.workshop_util) + 0.15 * mm(dealers.ftf)
                        + 0.10 * mm(dealers.as_gp))
    cx_index = 0.6 * mm(dealers.csi) + 0.4 * mm(dealers.service_retention)
    compliance_index = mm(dealers.compliance)
    dealers["sales_index"] = sales_index
    dealers["aftersales_index"] = aftersales_index
    dealers["cx_index"] = cx_index
    dealers["compliance_index"] = compliance_index
    dealers["dealer_score"] = (0.45 * sales_index + 0.35 * aftersales_index
                               + 0.15 * cx_index + 0.05 * compliance_index)
    dealers = dealers.sort_values("dealer_score", ascending=False).reset_index(drop=True)
    dealers["rank"] = dealers.index + 1

    months = pd.date_range("2025-07-01", periods=12, freq="MS")
    season = 1 + 0.12 * np.sin(np.linspace(0, 2 * np.pi, 12))
    nat_units = dealers.retail_units.sum() / 12
    nat_as = dealers.as_revenue.sum() / 12
    monthly = pd.DataFrame([dict(
        month=m.strftime("%Y-%m"),
        retail_units=int(nat_units * season[i] * rng.normal(1.0, 0.03)),
        as_revenue=float(nat_as * season[i] * rng.normal(1.0, 0.03)),
    ) for i, m in enumerate(months)])

    inv_rng = np.random.default_rng(77)
    base_value = {"A-Class": 195_000, "C-Class": 285_000, "E-Class": 405_000,
                  "S-Class": 720_000, "GLA": 235_000, "GLC": 345_000,
                  "GLE": 525_000, "EQE": 480_000}
    weights = np.array([0.08, 0.16, 0.34, 0.05, 0.10, 0.14, 0.08, 0.05])
    inv = []
    for i in range(140):
        model = inv_rng.choice(CPO_MODELS, p=weights)
        age = float(np.clip(inv_rng.normal(3.2, 1.1), 0.8, 7))
        km = float(np.clip(inv_rng.normal(5.5, 2.2), 0.8, 14))
        days_in_stock = int(np.clip(inv_rng.normal(95 if model == "E-Class" else 46, 28), 5, 190))
        fair = base_value[model] * max(0.5, 1 - 0.09 * age) * max(0.7, 1 - 0.012 * km)
        fair *= inv_rng.normal(1.0, 0.03)
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
            dealer_id=FOCUS_DEALER_ID, vehicle_id=f"CPO-{i + 1:03d}", model=model,
            age_years=round(age, 1), kilometer=round(km, 1), days_in_stock=days_in_stock,
            current_price=current, fair_price=fair, gap_pct=gap,
            sell_prob_30d=sell_prob, est_gp=est_gp, action=action,
        ))
    cpo_inventory = pd.DataFrame(inv)
    return {"dealers": dealers, "monthly": monthly, "cpo_inventory": cpo_inventory}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = build()
    for name, df in data.items():
        path = OUT / f"{name}.csv"
        df.round(4).to_csv(path, index=False)
        print(f"wrote {path}  ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
