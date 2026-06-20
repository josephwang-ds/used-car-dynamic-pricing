# Synthetic Dealer Network — Data Dictionary

These tables power the **CPO Retail Command Center** demo. They are produced by
`scripts/generate_dealer_network.py` (seed `2026`) and match exactly what the
Streamlit app renders.

> ⚠️ **100% synthetic.** No real Mercedes-Benz data, dealer names, customers, VINs,
> targets, or incentives are used. Values, model names (A/C/E/S-Class, etc.), KPI
> formulas, and the Dealer Score weights are transparent portfolio assumptions, not
> company definitions. For demonstration only.

Regenerate with:

```bash
python scripts/generate_dealer_network.py
```

---

## `dealers.csv` — 50 dealers × 43 columns

One row per dealer (5 regions × 10 dealers). `dealer_id` format `MB-<RegionInitial><nn>`
(e.g. `MB-E07`). `MB-E07` is the engineered "problem dealer" in the demo storyline.

| Column | Meaning |
|---|---|
| `dealer_id`, `region` | Anonymized dealer code and region (North / East / South / West / Central) |
| `nc_target`, `nc_units`, `nc_achv` | New-car units target, actual, achievement ratio |
| `cpo_target`, `cpo_units`, `cpo_achv` | Certified-pre-owned units target, actual, achievement ratio |
| `vans_target`, `vans_units`, `vans_achv` | Vans units target, actual, achievement ratio |
| `retail_units` | NC + CPO + Vans units sold |
| `cpo_penetration` | CPO ÷ (NC + CPO) |
| `asp` | Average selling price (CNY) |
| `gp_per_unit` | Gross profit per unit (CNY) |
| `discount_rate` | Average discount off list |
| `conversion` | Lead-to-sale conversion rate |
| `days_to_sale` | Average days from listing to sale |
| `days_supply` | Inventory days supply |
| `aging_90` | Share of stock aged 90+ days |
| `yoy` | Year-over-year sales growth |
| `as_revenue`, `as_target`, `as_gp` | After-sales revenue, target, gross profit (CNY) |
| `ro_volume` | Repair-order volume |
| `absorption` | Service absorption = after-sales GP ÷ fixed operating cost (cost synthetic) |
| `workshop_util` | Workshop utilization (sold hrs ÷ available tech hrs) |
| `tech_eff` | Technician efficiency (standard hrs ÷ actual hrs) |
| `aro` | Average repair order value = after-sales revenue ÷ RO volume |
| `parts_per_ro` | Parts lines per repair order |
| `parts_fill` | Parts fill rate |
| `service_retention` | Service customer retention rate |
| `ftf` | First-time fix rate |
| `repeat_repair` | Repeat-repair rate |
| `csi` | Customer satisfaction index / service NPS proxy (0–100) |
| `warranty_mix` | Warranty vs customer-pay revenue mix |
| `compliance` | Data-quality / compliance score (0–1) |
| `sales_index`, `aftersales_index`, `cx_index`, `compliance_index` | Sub-indices, min-max scaled 0–100 across the network |
| `dealer_score` | `0.45·Sales + 0.35·After-sales + 0.15·CX + 0.05·Compliance` |
| `rank` | Rank by `dealer_score` (1 = best) |

## `monthly.csv` — 12 months × 3 columns

National monthly trend (seasonality + noise) for the Executive Overview charts.

| Column | Meaning |
|---|---|
| `month` | `YYYY-MM` (2025-07 … 2026-06) |
| `retail_units` | National retail units that month |
| `as_revenue` | National after-sales revenue that month (CNY) |

## `cpo_inventory.csv` — 140 vehicles × 12 columns

CPO stock for the focus dealer `MB-E07`. E-Class is deliberately overstocked and
overpriced in the 90+ day band — the action layer of the demo.

| Column | Meaning |
|---|---|
| `dealer_id`, `vehicle_id` | Owning dealer and masked vehicle code |
| `model` | CPO model (A/C/E/S-Class, GLA, GLC, GLE, EQE) |
| `age_years`, `kilometer` | Vehicle age (years) and mileage (10k km) |
| `days_in_stock` | Days on lot |
| `current_price`, `fair_price` | Listed price vs model fair value (CNY) |
| `gap_pct` | `(current − fair) / fair` |
| `sell_prob_30d` | Modeled 30-day sell probability |
| `est_gp` | Estimated gross profit (CNY) |
| `action` | Recommended action: Reprice / Promote / Transfer / Raise / Hold |
