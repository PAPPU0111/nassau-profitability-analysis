import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Nassau Profitability Dashboard", layout="wide")
st.title("🍫 Nassau Candy – AI Powered Profitability Dashboard")

# ------------------------------
# LOAD DATA
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/nassau_candy_clean.csv", parse_dates=["order_date"])
    return df

df = load_data()

# ------------------------------
# AI INSIGHTS FUNCTION
# ------------------------------
def generate_ai_insights(df):
    insights = []

    revenue = df["sales"].sum()
    profit = df["gross_profit"].sum()
    margin = (profit / revenue) * 100 if revenue != 0 else 0

    insights.append(f"Overall profit margin is {margin:.2f}%, indicating {'strong' if margin > 60 else 'moderate'} profitability.")

    # Division dependency
    division_share = df.groupby("division")["sales"].sum() / revenue * 100
    if len(division_share) > 0:
        top_div = division_share.idxmax()
        top_val = division_share.max()
        if top_val > 70:
            insights.append(f"{top_div} division contributes {top_val:.1f}% of revenue — high dependency risk.")

    # Pareto effect
    product_profit = df.groupby("product_name")["gross_profit"].sum().sort_values(ascending=False)
    if len(product_profit) > 0:
        top5 = product_profit.head(5).sum()
        total_profit = product_profit.sum()
        if total_profit > 0 and (top5 / total_profit) > 0.7:
            insights.append("Top 5 products contribute majority of profit — strong Pareto effect.")

    # High cost products
    df["cost_ratio"] = df["cost"] / df["sales"]
    high_cost = df[df["cost_ratio"] > 0.5]["product_name"].unique()
    if len(high_cost) > 0:
        insights.append(f"High cost products detected: {', '.join(high_cost[:3])} — margins may be leaking.")

    # Low margin products
    product_margin = df.groupby("product_name").apply(
        lambda x: (x["gross_profit"].sum() / x["sales"].sum()) * 100
    )
    low_margin = product_margin[product_margin < 50]
    if len(low_margin) > 0:
        insights.append(f"{len(low_margin)} products have margin below 50% — review pricing strategy.")

    # Region issue
    region_margin = df.groupby("region").apply(
        lambda x: (x["gross_profit"].sum() / x["sales"].sum()) * 100
    )
    if len(region_margin) > 0:
        worst_region = region_margin.idxmin()
        insights.append(f"{worst_region} region has lowest margin — investigate operational inefficiencies.")

    return insights

# ------------------------------
# SIDEBAR FILTERS
# ------------------------------
st.sidebar.header("🔍 Filters")

division = st.sidebar.multiselect(
    "Select Division",
    options=df["division"].unique(),
    default=df["division"].unique()
)

region = st.sidebar.multiselect(
    "Select Region",
    options=df["region"].unique(),
    default=df["region"].unique()
)

product = st.sidebar.multiselect(
    "Select Product",
    options=df["product_name"].unique(),
    default=df["product_name"].unique()
)

filtered_df = df[
    (df["division"].isin(division)) &
    (df["region"].isin(region)) &
    (df["product_name"].isin(product))
]

# ------------------------------
# KPIs
# ------------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

total_revenue = filtered_df["sales"].sum()
total_profit = filtered_df["gross_profit"].sum()
total_cost = filtered_df["cost"].sum()
margin = (total_profit / total_revenue) * 100 if total_revenue != 0 else 0

col1.metric("Revenue", f"${total_revenue:,.0f}")
col2.metric("Profit", f"${total_profit:,.0f}")
col3.metric("Cost", f"${total_cost:,.0f}")
col4.metric("Margin %", f"{margin:.2f}%")

# ------------------------------
# 🤖 AI INSIGHTS
# ------------------------------
st.subheader("🤖 AI-Generated Business Insights")

insights = generate_ai_insights(filtered_df)

for insight in insights:
    st.info(insight)

# ------------------------------
# 🏆 PRODUCT PROFITABILITY
# ------------------------------
st.subheader("🏆 Top Products by Profit")

product_profit = (
    filtered_df.groupby("product_name")["gross_profit"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots()
product_profit.plot(kind="bar", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# ------------------------------
# 📊 PIE CHART
# ------------------------------
st.subheader("📊 Division Revenue Share")

division_share = filtered_df.groupby("division")["sales"].sum()

fig, ax = plt.subplots()
ax.pie(division_share, labels=division_share.index, autopct='%1.1f%%')
st.pyplot(fig)

# ------------------------------
# 🏭 DIVISION ANALYSIS
# ------------------------------
st.subheader("🏭 Division Performance")

division_data = filtered_df.groupby("division")[["sales", "gross_profit"]].sum()

fig, ax = plt.subplots()
division_data.plot(kind="bar", ax=ax)
st.pyplot(fig)

# ------------------------------
# 🔥 HEATMAP
# ------------------------------
st.subheader("🔥 Region vs Product Heatmap")

heatmap_data = filtered_df.pivot_table(
    values="gross_profit",
    index="region",
    columns="product_name",
    aggfunc="sum"
)

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(heatmap_data, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ------------------------------
# 📈 MONTHLY TREND
# ------------------------------
st.subheader("📈 Monthly Revenue Trend")

monthly = filtered_df.groupby(filtered_df["order_date"].dt.to_period("M"))["sales"].sum()
monthly.index = monthly.index.astype(str)

fig, ax = plt.subplots()
monthly.plot(ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# ------------------------------
# 🌍 REGION ANALYSIS
# ------------------------------
st.subheader("🌍 Region-wise Profit")

region_profit = filtered_df.groupby("region")["gross_profit"].sum()

fig, ax = plt.subplots()
region_profit.plot(kind="bar", ax=ax)
st.pyplot(fig)

# ------------------------------
# 🎯 QUADRANT ANALYSIS
# ------------------------------
st.subheader("🎯 Product Quadrant (Profit vs Margin)")

product_summary = filtered_df.groupby("product_name").agg({
    "sales": "sum",
    "gross_profit": "sum"
})

product_summary["margin"] = (product_summary["gross_profit"] / product_summary["sales"]) * 100

fig, ax = plt.subplots()

sns.scatterplot(data=product_summary, x="sales", y="margin", ax=ax)

for i in product_summary.index:
    ax.text(product_summary.loc[i, "sales"], product_summary.loc[i, "margin"], i, fontsize=8)

ax.axhline(product_summary["margin"].mean(), linestyle="--")
ax.axvline(product_summary["sales"].mean(), linestyle="--")

st.pyplot(fig)

# ------------------------------
# 💸 COST VS SALES
# ------------------------------
st.subheader("💸 Cost vs Sales")

fig, ax = plt.subplots()
sns.scatterplot(data=filtered_df, x="sales", y="cost", ax=ax)
st.pyplot(fig)

# ------------------------------
# 📄 RAW DATA
# ------------------------------
st.subheader("📄 Raw Data Preview")
st.dataframe(filtered_df.head(50))