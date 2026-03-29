import pandas as pd
import numpy as np

# ── 1. LOAD ────────────────────────────────────────────────────────────────
def load_from_csv(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        print("[INFO] File loaded successfully")
        print("[INFO] Columns:", list(df.columns))
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load file: {e}")
        exit()


# ── 2. CLEAN ───────────────────────────────────────────────────────────────
def clean_nassau(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[INFO] Raw shape: {df.shape}")

    # ── Column names cleaning ───────────────────────────────────────────────
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\/\s]+", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )

    # Rename common columns
    rename_map = {
        "country_region": "country",
        "state_province": "state",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # ── Remove duplicates ───────────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[INFO] Dropped {before - len(df)} duplicate rows")

    # ── Date parsing (FIXED) ────────────────────────────────────────────────
    for col in ["order_date", "ship_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%d-%m-%Y", errors="coerce")
            print(df[["order_date", "ship_date"]].head(10))

    if "order_date" in df.columns and "ship_date" in df.columns:
        df["date_parse_error"] = df["order_date"].isna() | df["ship_date"].isna()

        df["lead_time_days"] = (df["ship_date"] - df["order_date"]).dt.days

        df["lead_time_days"] = (df["ship_date"] - df["order_date"]).dt.days

        df["lead_time_flag"] = df["lead_time_days"] < 0
    else:
        print("[WARNING] Date columns not found")

    # ── Numeric columns ─────────────────────────────────────────────────────
    numeric_cols = ["sales", "units", "gross_profit", "cost"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    existing_numeric = [c for c in numeric_cols if c in df.columns]

    if existing_numeric:
        df["numeric_null_flag"] = df[existing_numeric].isnull().any(axis=1)

    if "sales" in df.columns:
        df["zero_neg_sales_flag"] = df["sales"] <= 0

    if "units" in df.columns:
        df["zero_neg_units_flag"] = df["units"] <= 0

    # ── Profit validation ───────────────────────────────────────────────────
    if all(c in df.columns for c in ["sales", "cost", "gross_profit"]):
        df["calc_gross_profit"] = (df["sales"] - df["cost"]).round(2)

        df["profit_check_flag"] = (
            (df["gross_profit"] - df["calc_gross_profit"]).abs() > 0.02
        )

    # ── KPI calculations ────────────────────────────────────────────────────
    if all(c in df.columns for c in ["sales", "gross_profit", "units", "cost"]):

        df["gross_margin_pct"] = np.where(
            df["sales"] > 0,
            (df["gross_profit"] / df["sales"] * 100),
            0
        ).round(2)

        df["profit_per_unit"] = np.where(
            df["units"] > 0,
            (df["gross_profit"] / df["units"]),
            0
        ).round(4)

        df["cost_per_unit"] = np.where(
            df["units"] > 0,
            (df["cost"] / df["units"]),
            0
        ).round(4)

        total_sales = df["sales"].sum()
        total_profit = df["gross_profit"].sum()

        if total_sales != 0:
            df["revenue_contribution"] = df["sales"] / total_sales

        if total_profit != 0:
            df["profit_contribution"] = df["gross_profit"] / total_profit

    # ── Text cleaning ───────────────────────────────────────────────────────
    str_cols = ["ship_mode", "division", "region", "product_name",
                "country", "city", "state"]

    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # ── Product name fix ────────────────────────────────────────────────────
    if "product_name" in df.columns:
        df["product_name"] = df["product_name"].str.replace(r"\s+", " ", regex=True)

    # ── Division mismatch flag ──────────────────────────────────────────────
    if "product_id" in df.columns and "division" in df.columns:
        df["division_mismatch_flag"] = (
            (df["product_id"].str.startswith("OTH-") & (df["division"] != "Other")) |
            (df["product_id"].str.startswith("CHO-") & (df["division"] != "Chocolate")) |
            (df["product_id"].str.startswith("SUG-") & (df["division"] != "Sugar"))
        )

    # ── Postal code fix ─────────────────────────────────────────────────────
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].astype(str).str.strip()

    # ── Remove unusable rows ────────────────────────────────────────────────
    if "zero_neg_sales_flag" in df.columns and "numeric_null_flag" in df.columns:
        hard_drop = df["zero_neg_sales_flag"] & df["numeric_null_flag"]
        df = df[~hard_drop]

    # ── Reset index ─────────────────────────────────────────────────────────
    df.reset_index(drop=True, inplace=True)

    print(f"[INFO] Clean shape: {df.shape}")
    return df


# ── 3. REPORT ───────────────────────────────────────────────────────────────
def quality_report(df: pd.DataFrame):
    print("\n=== DATA QUALITY SUMMARY ===")

    flag_cols = [c for c in df.columns if c.endswith("_flag")]

    for col in flag_cols:
        n = df[col].sum()
        pct = (n / len(df)) * 100
        print(f"{col:<30} {n} rows ({pct:.1f}%)")

    print(f"\nTotal rows: {len(df)}")

    if "order_date" in df.columns:
        print(f"Date range: {df['order_date'].min()} → {df['order_date'].max()}")

    if "division" in df.columns:
        print(f"Divisions: {df['division'].unique()}")

    if "gross_margin_pct" in df.columns:
        print("\nMargin Stats:")
        print(df["gross_margin_pct"].describe())


# ── 4. MAIN ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    filepath = "data/nassau_candy_data.csv"

    raw_df = load_from_csv(filepath)

    clean_df = clean_nassau(raw_df)

    quality_report(clean_df)

    output_path = "data/nassau_candy_clean.csv"
    clean_df.to_csv(output_path, index=False)

    print(f"\n[INFO] Cleaned file saved at → {output_path}")