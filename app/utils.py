import pandas as pd


def load_gold_data(csv_path: str = "app/data/Gold Price.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {
        "date": "date",
        "close": "price",
        "price": "price",
        "open": "open",
        "high": "high",
        "low": "low",
        "volume": "volume",
        "chg%": "chg%",
        "change%": "chg%",
        "change_percent": "chg%",
        "change": "chg%",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "date" not in df.columns:
        raise KeyError("Expected a 'date' column. Please confirm the CSV includes it.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    for col in ["price", "open", "high", "low", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "chg%" in df.columns:
        df["chg%"] = (
            df["chg%"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df["chg%"] = pd.to_numeric(df["chg%"], errors="coerce")

    return df


def generate_sentence(row: pd.Series) -> str:
    date_str = pd.to_datetime(row["date"]).strftime("%B %d, %Y")
    price = _fmt_rupee(row.get("price"))
    high = _fmt_rupee(row.get("high"))
    low = _fmt_rupee(row.get("low"))
    open_ = _fmt_rupee(row.get("open"))
    chg = row.get("chg%")
    chg_str = f"{chg:.2f}%" if pd.notna(chg) else "n/a"

    parts = [f"On {date_str}, the closing price of gold was ₹{price}."]
    if pd.notna(high) and pd.notna(low) and pd.notna(open_):
        parts.append(f"The day's high was ₹{high}, low was ₹{low}, and it opened at ₹{open_}.")
    elif pd.notna(high) or pd.notna(low) or pd.notna(open_):
        extra = []
        if pd.notna(high): extra.append(f"high ₹{high}")
        if pd.notna(low): extra.append(f"low ₹{low}")
        if pd.notna(open_): extra.append(f"open ₹{open_}")
        parts.append("Available intraday info: " + ", ".join(extra) + ".")

    if "chg%" in row.index:
        parts.append(f"Change percentage was {chg_str}.")

    return " ".join(parts)


def _fmt_rupee(val) -> str | None:
    try:
        if pd.isna(val):
            return None
        val = float(val)
        return f"{int(val):,}" if val.is_integer() else f"{val:,.0f}"
    except Exception:
        return None


def summarize_monthly_stats(df: pd.DataFrame, price_col: str = "price") -> str:
    if df.empty:
        return "No data available for the specified month or year."

    monthly_avg = df.groupby(["year", "month"])[price_col].mean().reset_index()
    monthly_avg = monthly_avg.sort_values(["year", "month"])

    summary_lines = []
    for _, row in monthly_avg.iterrows():
        year, month, avg_price = int(row["year"]), int(row["month"]), round(row[price_col])
        summary_lines.append(f"In {month:02}/{year}, the average gold price was ₹{avg_price}.")

    return "\n".join(summary_lines)
