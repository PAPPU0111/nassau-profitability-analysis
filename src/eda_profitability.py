"""
Nassau Candy Distributor
EDA + Product Line Profitability Analysis
=========================================
Run:
    pip install pandas numpy matplotlib seaborn scipy
    python eda_profitability.py

Outputs:
    reports/  →  all PNG charts + eda_summary.txt
"""

import os
import warnings
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "data/nassau_candy_clean.csv"
REPORT_DIR  = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

PALETTE     = {
    "Chocolate": "#6B3A2A",
    "Sugar":     "#F4A261",
    "Other":     "#457B9D",
}
BRAND_COLOR = "#2D6A4F"
sns.set_theme(style="whitegrid", font_scale=1.1)

# ── Helpers ───────────────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(REPORT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")

def hbar(ax, series, title, xlabel, color=BRAND_COLOR, annotate=True):
    series = series.sort_values()
    bars = ax.barh(series.index, series.values, color=color, edgecolor="white")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel)
    ax.axvline(series.mean(), color="red", linestyle="--", linewidth=1,
               label=f"Mean {series.mean():.1f}")
    ax.legend(fontsize=9)
    if annotate:
        for bar, val in zip(bars, series.values):
            ax.text(val + series.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:,.1f}", va="center", fontsize=9)

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n[1/9] Loading clean data …")
df = pd.read_csv(INPUT_FILE, parse_dates=["order_date", "ship_date"])
print(f"      {len(df):,} rows  |  {df['product_name'].nunique()} products  "
      f"|  {df['division'].nunique()} divisions")

# ── Section 1 – OVERVIEW KPIs ────────────────────────────────────────────────
print("[2/9] Computing overview KPIs …")

total_sales   = df["sales"].sum()
total_profit  = df["gross_profit"].sum()
total_cost    = df["cost"].sum()
total_units   = df["units"].sum()
overall_margin = total_profit / total_sales * 100

summary_lines = [
    "=" * 60,
    "  NASSAU CANDY DISTRIBUTOR – EDA SUMMARY REPORT",
    "=" * 60,
    f"  Total Revenue      : ${total_sales:>12,.2f}",
    f"  Total Gross Profit : ${total_profit:>12,.2f}",
    f"  Total Cost         : ${total_cost:>12,.2f}",
    f"  Total Units Sold   : {total_units:>13,.0f}",
    f"  Overall Margin     : {overall_margin:>11.2f}%",
    f"  Unique Products    : {df['product_name'].nunique()}",
    f"  Unique Customers   : {df['customer_id'].nunique()}",
    f"  Order Date Range   : {df['order_date'].min().date()} → {df['order_date'].max().date()}",
    "",
]

# ── Section 2 – PRODUCT-LEVEL METRICS ────────────────────────────────────────
print("[3/9] Building product-level profitability table …")

prod = (
    df.groupby("product_name")
    .agg(
        total_sales   = ("sales",        "sum"),
        total_profit  = ("gross_profit", "sum"),
        total_cost    = ("cost",         "sum"),
        total_units   = ("units",        "sum"),
        order_count   = ("order_id",     "nunique"),
    )
    .assign(
        gross_margin_pct = lambda x: x["total_profit"] / x["total_sales"] * 100,
        profit_per_unit  = lambda x: x["total_profit"] / x["total_units"],
        revenue_share    = lambda x: x["total_sales"]  / x["total_sales"].sum() * 100,
        profit_share     = lambda x: x["total_profit"] / x["total_profit"].sum() * 100,
    )
    .sort_values("total_profit", ascending=False)
    .round(2)
)

summary_lines += [
    "-" * 60,
    "  PRODUCT PROFITABILITY LEADERBOARD (by Gross Profit)",
    "-" * 60,
]
for rank, (name, row) in enumerate(prod.iterrows(), 1):
    summary_lines.append(
        f"  {rank:>2}. {name:<40}  "
        f"Margin: {row['gross_margin_pct']:>5.1f}%  "
        f"Profit: ${row['total_profit']:>9,.0f}"
    )
summary_lines.append("")

# ── Plot 2a – Gross Profit by Product ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
colors = [PALETTE.get(
    df.loc[df["product_name"] == p, "division"].iloc[0], BRAND_COLOR)
    for p in prod.index]
hbar(ax, prod["total_profit"] / 1000,
     "Gross Profit by Product  (USD '000)", "Gross Profit ($K)", color=colors)
# legend patches
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()]
ax.legend(handles=patches, title="Division", fontsize=9)
save(fig, "01_gross_profit_by_product.png")

# ── Plot 2b – Gross Margin % by Product ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
margin_colors = ["#d62728" if m < 50 else "#2ca02c" for m in prod["gross_margin_pct"]]
hbar(ax, prod["gross_margin_pct"],
     "Gross Margin % by Product", "Gross Margin (%)", color=margin_colors)
ax.axvline(50, color="orange", linestyle=":", linewidth=1.5, label="50 % threshold")
ax.legend(fontsize=9)
save(fig, "02_gross_margin_by_product.png")

# ── Plot 2c – Revenue vs Profit scatter ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for div, grp in prod.join(
        df.groupby("product_name")["division"].first()).groupby("division"):
    ax.scatter(grp["total_sales"] / 1000, grp["total_profit"] / 1000,
               color=PALETTE.get(div, "grey"), s=120, label=div, zorder=3)
    for name, row in grp.iterrows():
        ax.annotate(textwrap.shorten(name, 22),
                    (row["total_sales"] / 1000, row["total_profit"] / 1000),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

ax.set_xlabel("Total Revenue ($K)")
ax.set_ylabel("Gross Profit ($K)")
ax.set_title("Revenue vs Gross Profit by Product", fontweight="bold")
ax.legend(title="Division")
# Diagonal reference line (constant margin)
lim = max(prod["total_sales"].max(), prod["total_profit"].max()) / 1000 * 1.05
for m, style in [(0.5, ":"), (overall_margin / 100, "--")]:
    ax.plot([0, lim], [0, lim * m], linestyle=style, color="grey", linewidth=1,
            label=f"{m*100:.0f}% margin" if m != overall_margin / 100
                  else f"Overall avg {overall_margin:.1f}%")
ax.legend(fontsize=9)
save(fig, "03_revenue_vs_profit_scatter.png")

# ── Section 3 – DIVISION-LEVEL ────────────────────────────────────────────────
print("[4/9] Division-level analysis …")

div_df = (
    df.groupby("division")
    .agg(
        total_sales   = ("sales",        "sum"),
        total_profit  = ("gross_profit", "sum"),
        total_cost    = ("cost",         "sum"),
        total_units   = ("units",        "sum"),
    )
    .assign(
        gross_margin_pct = lambda x: x["total_profit"] / x["total_sales"] * 100,
        revenue_share    = lambda x: x["total_sales"]  / x["total_sales"].sum() * 100,
        profit_share     = lambda x: x["total_profit"] / x["total_profit"].sum() * 100,
    )
    .sort_values("total_profit", ascending=False)
    .round(2)
)

summary_lines += [
    "-" * 60,
    "  DIVISION PERFORMANCE",
    "-" * 60,
]
for div, row in div_df.iterrows():
    summary_lines.append(
        f"  {div:<12}  Revenue: ${row['total_sales']:>10,.0f}  "
        f"Profit: ${row['total_profit']:>9,.0f}  "
        f"Margin: {row['gross_margin_pct']:>5.1f}%  "
        f"Rev-share: {row['revenue_share']:>5.1f}%"
    )
summary_lines.append("")

# ── Plot 3a – Division Revenue vs Profit side-by-side ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
metrics = [("total_sales", "Revenue ($)"), ("total_profit", "Gross Profit ($)")]
for ax, (col, label) in zip(axes, metrics):
    bars = ax.bar(div_df.index,
                  div_df[col] / 1000,
                  color=[PALETTE.get(d, "grey") for d in div_df.index],
                  edgecolor="white", width=0.5)
    ax.set_title(f"{label} by Division", fontweight="bold")
    ax.set_ylabel("$K")
    for bar, val in zip(bars, div_df[col] / 1000):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 5,
                f"${val:,.0f}K", ha="center", fontsize=10, fontweight="bold")
fig.tight_layout()
save(fig, "04_division_revenue_profit.png")

# ── Plot 3b – Division margin comparison ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(div_df.index, div_df["gross_margin_pct"],
              color=[PALETTE.get(d, "grey") for d in div_df.index],
              edgecolor="white", width=0.4)
ax.axhline(overall_margin, color="red", linestyle="--", linewidth=1.5,
           label=f"Overall avg {overall_margin:.1f}%")
ax.set_title("Gross Margin % by Division", fontweight="bold")
ax.set_ylabel("Gross Margin (%)")
ax.set_ylim(0, 100)
for bar, val in zip(bars, div_df["gross_margin_pct"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
            f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.legend()
save(fig, "05_division_margin.png")

# ── Section 4 – PARETO ANALYSIS ───────────────────────────────────────────────
print("[5/9] Pareto (80/20) analysis …")

pareto_rev = prod.sort_values("total_sales", ascending=False).copy()
pareto_rev["cum_rev_pct"] = pareto_rev["total_sales"].cumsum() / pareto_rev["total_sales"].sum() * 100

pareto_pft = prod.sort_values("total_profit", ascending=False).copy()
pareto_pft["cum_pft_pct"] = pareto_pft["total_profit"].cumsum() / pareto_pft["total_profit"].sum() * 100

n_for_80_rev = (pareto_rev["cum_rev_pct"] <= 80).sum() + 1
n_for_80_pft = (pareto_pft["cum_pft_pct"] <= 80).sum() + 1

summary_lines += [
    "-" * 60,
    "  PARETO (80/20) CONCENTRATION",
    "-" * 60,
    f"  Products driving 80% of Revenue : {n_for_80_rev} / {len(prod)}",
    f"  Products driving 80% of Profit  : {n_for_80_pft} / {len(prod)}",
    "",
]

# ── Plot 4 – Pareto chart (profit) ───────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()
short_names = [textwrap.shorten(n, 22) for n in pareto_pft.index]
ax1.bar(short_names, pareto_pft["total_profit"] / 1000,
        color=BRAND_COLOR, alpha=0.85, label="Gross Profit")
ax2.plot(short_names, pareto_pft["cum_pft_pct"],
         color="red", marker="o", markersize=5, label="Cumulative %")
ax2.axhline(80, color="orange", linestyle="--", linewidth=1.5, label="80% line")
ax1.set_title("Pareto Analysis — Gross Profit Concentration", fontweight="bold")
ax1.set_ylabel("Gross Profit ($K)")
ax2.set_ylabel("Cumulative %")
ax2.set_ylim(0, 105)
plt.xticks(rotation=35, ha="right", fontsize=9)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right")
fig.tight_layout()
save(fig, "06_pareto_profit.png")

# ── Section 5 – COST STRUCTURE DIAGNOSTICS ───────────────────────────────────
print("[6/9] Cost structure diagnostics …")

# Cost ratio = Cost / Sales
prod_cost = (
    df.groupby("product_name")
    .agg(total_sales=("sales", "sum"), total_cost=("cost", "sum"),
         total_profit=("gross_profit", "sum"))
    .assign(cost_ratio=lambda x: x["total_cost"] / x["total_sales"] * 100)
    .join(df.groupby("product_name")["division"].first())
    .sort_values("cost_ratio", ascending=False)
)

summary_lines += [
    "-" * 60,
    "  COST STRUCTURE DIAGNOSTICS (Cost as % of Sales)",
    "-" * 60,
]
for name, row in prod_cost.iterrows():
    flag = " ← HIGH COST" if row["cost_ratio"] > 40 else ""
    summary_lines.append(
        f"  {name:<40}  Cost%: {row['cost_ratio']:>5.1f}%{flag}"
    )
summary_lines.append("")

# ── Plot 5a – Cost ratio bar ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = ["#d62728" if c > 40 else BRAND_COLOR for c in prod_cost["cost_ratio"]]
hbar(ax, prod_cost["cost_ratio"],
     "Cost as % of Sales by Product\n(red = cost ratio > 40%)", "Cost Ratio (%)",
     color=bar_colors)
ax.axvline(40, color="orange", linestyle=":", linewidth=1.5, label="40% threshold")
ax.legend(fontsize=9)
save(fig, "07_cost_ratio_by_product.png")

# ── Plot 5b – Cost vs Sales scatter with margin colour ───────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(prod_cost["total_sales"] / 1000,
                prod_cost["total_cost"] / 1000,
                c=prod_cost["cost_ratio"], cmap="RdYlGn_r",
                s=150, edgecolors="white", zorder=3)
plt.colorbar(sc, ax=ax, label="Cost Ratio (%)")
for name, row in prod_cost.iterrows():
    ax.annotate(textwrap.shorten(name, 20),
                (row["total_sales"] / 1000, row["total_cost"] / 1000),
                textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel("Total Sales ($K)")
ax.set_ylabel("Total Cost ($K)")
ax.set_title("Cost vs Sales — Colour = Cost Ratio  (greener = lower cost%)",
             fontweight="bold")
save(fig, "08_cost_vs_sales_scatter.png")

# ── Section 6 – REGION-LEVEL ─────────────────────────────────────────────────
print("[7/9] Regional analysis …")

reg = (
    df.groupby("region")
    .agg(total_sales=("sales","sum"), total_profit=("gross_profit","sum"),
         total_units=("units","sum"))
    .assign(gross_margin_pct=lambda x: x["total_profit"]/x["total_sales"]*100,
            revenue_share=lambda x: x["total_sales"]/x["total_sales"].sum()*100)
    .sort_values("total_profit", ascending=False)
    .round(2)
)

summary_lines += [
    "-" * 60,
    "  REGIONAL PERFORMANCE",
    "-" * 60,
]
for region, row in reg.iterrows():
    summary_lines.append(
        f"  {region:<12}  Rev: ${row['total_sales']:>10,.0f}  "
        f"Profit: ${row['total_profit']:>9,.0f}  "
        f"Margin: {row['gross_margin_pct']:>5.1f}%"
    )
summary_lines.append("")

# ── Plot 6 – Region margin heatmap (product × region) ────────────────────────
pivot_margin = (
    df.groupby(["region", "product_name"])
    .apply(lambda g: g["gross_profit"].sum() / g["sales"].sum() * 100)
    .unstack("product_name")
    .round(1)
)
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(pivot_margin, annot=True, fmt=".1f", cmap="RdYlGn",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Gross Margin %"},
            vmin=0, vmax=100)
ax.set_title("Gross Margin % — Region × Product", fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=40, ha="right", fontsize=8)
fig.tight_layout()
save(fig, "09_region_product_margin_heatmap.png")

# ── Section 7 – TIME TREND ────────────────────────────────────────────────────
print("[8/9] Monthly revenue & margin trend …")

df["year_month"] = df["order_date"].dt.to_period("M")
monthly = (
    df.groupby("year_month")
    .agg(revenue=("sales","sum"), profit=("gross_profit","sum"))
    .assign(margin=lambda x: x["profit"]/x["revenue"]*100)
)

fig, ax1 = plt.subplots(figsize=(14, 5))
ax2 = ax1.twinx()
ax1.fill_between(monthly.index.astype(str), monthly["revenue"] / 1000,
                 color=BRAND_COLOR, alpha=0.4, label="Revenue ($K)")
ax1.plot(monthly.index.astype(str), monthly["revenue"] / 1000,
         color=BRAND_COLOR, linewidth=1.5)
ax2.plot(monthly.index.astype(str), monthly["margin"],
         color="#d62728", linewidth=2, linestyle="--", label="Gross Margin %")
ax1.set_title("Monthly Revenue & Gross Margin Trend", fontweight="bold")
ax1.set_ylabel("Revenue ($K)", color=BRAND_COLOR)
ax2.set_ylabel("Gross Margin (%)", color="#d62728")
tick_step = max(1, len(monthly) // 12)
ticks = list(range(0, len(monthly), tick_step))
ax1.set_xticks(ticks)
ax1.set_xticklabels([str(monthly.index[i]) for i in ticks], rotation=45, ha="right")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
fig.tight_layout()
save(fig, "10_monthly_revenue_margin_trend.png")

# ── Section 8 – MARGIN RISK FLAGS ────────────────────────────────────────────
print("[9/9] Generating margin risk flags …")

# Classify each product
conditions = [
    (prod["total_sales"] >= prod["total_sales"].median()) & (prod["gross_margin_pct"] >= prod["gross_margin_pct"].median()),
    (prod["total_sales"] >= prod["total_sales"].median()) & (prod["gross_margin_pct"] <  prod["gross_margin_pct"].median()),
    (prod["total_sales"] <  prod["total_sales"].median()) & (prod["gross_margin_pct"] >= prod["gross_margin_pct"].median()),
    (prod["total_sales"] <  prod["total_sales"].median()) & (prod["gross_margin_pct"] <  prod["gross_margin_pct"].median()),
]
labels_quad = ["⭐ Star (High Rev, High Margin)",
               "⚠️  Volume Trap (High Rev, Low Margin)",
               "💎 Hidden Gem (Low Rev, High Margin)",
               "🚨 Danger Zone (Low Rev, Low Margin)"]
prod["quadrant"] = np.select(conditions, labels_quad, default="Unknown")

summary_lines += [
    "-" * 60,
    "  PRODUCT QUADRANT CLASSIFICATION",
    "-" * 60,
]
for quad in labels_quad:
    products_in = prod[prod["quadrant"] == quad].index.tolist()
    summary_lines.append(f"\n  {quad}")
    for p in products_in:
        summary_lines.append(f"    • {p}")
summary_lines.append("")

# ── Plot 8 – Quadrant scatter ─────────────────────────────────────────────────
quad_palette = {
    labels_quad[0]: "#2ca02c",
    labels_quad[1]: "#ff7f0e",
    labels_quad[2]: "#1f77b4",
    labels_quad[3]: "#d62728",
}
prod_plot = prod.join(df.groupby("product_name")["division"].first())
fig, ax = plt.subplots(figsize=(10, 7))
for quad, grp in prod_plot.groupby("quadrant"):
    ax.scatter(grp["total_sales"] / 1000, grp["gross_margin_pct"],
               color=quad_palette.get(quad, "grey"),
               s=grp["total_units"] / grp["total_units"].max() * 800 + 80,
               label=quad, alpha=0.85, edgecolors="white", zorder=3)
    for name, row in grp.iterrows():
        ax.annotate(textwrap.shorten(name, 20),
                    (row["total_sales"] / 1000, row["gross_margin_pct"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
# Quadrant lines
ax.axvline(prod["total_sales"].median() / 1000, color="grey",
           linestyle=":", linewidth=1)
ax.axhline(prod["gross_margin_pct"].median(), color="grey",
           linestyle=":", linewidth=1)
ax.set_xlabel("Total Revenue ($K)  →  Bubble size = units sold")
ax.set_ylabel("Gross Margin %")
ax.set_title("Product Quadrant Matrix\n(Revenue × Margin, size = units)",
             fontweight="bold")
ax.legend(title="Quadrant", fontsize=8, title_fontsize=9)
save(fig, "11_product_quadrant_matrix.png")

# ── Write summary text ────────────────────────────────────────────────────────
summary_lines += [
    "=" * 60,
    "  KEY RECOMMENDATIONS",
    "=" * 60,
]

# Auto-generate recs based on data
stars     = prod[prod["quadrant"] == labels_quad[0]].index.tolist()
traps     = prod[prod["quadrant"] == labels_quad[1]].index.tolist()
gems      = prod[prod["quadrant"] == labels_quad[2]].index.tolist()
danger    = prod[prod["quadrant"] == labels_quad[3]].index.tolist()
high_cost = prod_cost[prod_cost["cost_ratio"] > 40].index.tolist()

if stars:
    summary_lines.append(f"\n  1. PROTECT star products: {', '.join(stars)}")
    summary_lines.append("     → Prioritise inventory & promotional spend here.")
if traps:
    summary_lines.append(f"\n  2. INVESTIGATE volume traps: {', '.join(traps)}")
    summary_lines.append("     → High sales but margin leakage — review pricing or sourcing costs.")
if gems:
    summary_lines.append(f"\n  3. SCALE hidden gems: {', '.join(gems)}")
    summary_lines.append("     → Strong margins but under-selling — increase exposure.")
if danger:
    summary_lines.append(f"\n  4. REVIEW danger-zone products: {', '.join(danger)}")
    summary_lines.append("     → Low volume AND low margin — candidates for discontinuation.")
if high_cost:
    summary_lines.append(f"\n  5. RENEGOTIATE costs for: {', '.join(high_cost)}")
    summary_lines.append("     → Cost ratio exceeds 40% — pricing review or supplier negotiation needed.")

bottom_region = reg["gross_margin_pct"].idxmin()
summary_lines.append(
    f"\n  6. Focus on Region '{bottom_region}' — lowest margin "
    f"({reg.loc[bottom_region,'gross_margin_pct']:.1f}%) despite significant revenue."
)

summary_lines.append("\n" + "=" * 60 + "\n")

report_text = "\n".join(summary_lines)
report_path = os.path.join(REPORT_DIR, "eda_summary.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
print(report_text)
print(f"\n[DONE] All outputs saved to '{REPORT_DIR}/'")
print("  Charts  : 01 through 11  (.png)")
print("  Summary : eda_summary.txt")