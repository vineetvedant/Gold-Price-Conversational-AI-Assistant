# app/model.py
import re
import httpx
import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from app.utils import load_gold_data, generate_sentence

OLLAMA_BASE_URL = "http://localhost:11434"
CHAT_MODEL = "deepseek-r1:1.5b"
PRICE_COL = "price"
NUDGE = (
    "Based on this data, many find that now is a good time to start. "
    "Would you be interested in purchasing some digital gold today?"
)

# Load and clean data
df = load_gold_data()
df.columns = [c.strip().lower() for c in df.columns]
df.rename(columns={"date": "date", "price": "price"}, inplace=True)

if "date" not in df.columns or PRICE_COL not in df.columns:
    raise KeyError("Expected columns 'date' and 'price' not found in dataset.")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df.dropna(subset=["date", PRICE_COL], inplace=True)
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

# RAG prep
def default_sentence(row):
    open_ = int(row['open']) if 'open' in row and pd.notna(row['open']) else None
    high  = int(row['high']) if 'high' in row and pd.notna(row['high']) else None
    low   = int(row['low'])  if 'low'  in row and pd.notna(row['low'])  else None
    close = int(row['price']) if pd.notna(row['price']) else None

    bits = [f"On {row['date'].strftime('%Y-%m-%d')}"]
    if open_ is not None: bits.append(f"opened at ₹{open_}")
    if high  is not None: bits.append(f"peaked at ₹{high}")
    if low   is not None: bits.append(f"dropped to ₹{low}")
    if close is not None: bits.append(f"closed at ₹{close}")
    chg = row.get("chg%")
    if pd.notna(chg): bits.append(f"with a {chg}% change")
    return ", ".join(bits) + "."

sentences = df.apply(default_sentence, axis=1).tolist()
ids = [f"id_{i}" for i in range(len(sentences))]

persist_dir = "app/vector_store"
chroma_client = chromadb.PersistentClient(path=persist_dir)

local_embedder = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="gold_price_local_all-MiniLM-L6-v2",
    embedding_function=local_embedder,
)

if collection.count() == 0:
    df_meta = df.copy()
    df_meta["date"] = df_meta["date"].dt.strftime("%Y-%m-%d")
    collection.add(
        documents=sentences,
        metadatas=df_meta.to_dict(orient="records"),
        ids=ids
    )

def chat_complete(system_msg: str, prompt: str) -> str:
    payload = {
        "model": CHAT_MODEL,
        "system": system_msg,
        "prompt": prompt,
        "stream": False
    }
    try:
        with httpx.Client(timeout=120) as client:
            r = client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        r.raise_for_status()
        answer = r.json().get("response", "").strip()
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

        if NUDGE.lower() not in answer.lower():
            answer += f"\n\n{NUDGE}"
        return answer
    except Exception as e:
        return f"[Chat Error] {e}"

def classify_query(q: str) -> str:
    ql = q.lower()
    if re.search(r"\b(best|cheapest|lowest)\b.*\b(month|time|year)\b", ql):
        return "best_month"
    elif re.search(r"\bprice\b.*\b(20\d{2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", ql):
        return "specific_price"
    elif any(w in ql for w in ["trend", "increase", "rise", "up", "performance"]):
        return "trend_analysis"
    elif any(w in ql for w in ["predict", "forecast", "future", "next year"]):
        return "prediction"
    elif any(k in ql for k in ["gold", "price", "rate", "mcx", "digital gold"]):
        return "generic_gold"
    else:
        return "off_topic"

def is_purchase_intent(user_query: str) -> bool:
    q = user_query.lower()
    buy_words = ["buy", "purchase", "invest", "order", "checkout", "pay"]
    gold_words = ["gold", "digital gold", "grams", "kg", "gms", "gram", "mcx"]
    has_money = any(sym in q for sym in ["\u20b9", "rs", "inr"]) or any(x in q for x in ["amount", "price"])
    return (any(w in q for w in buy_words) and any(w in q for w in gold_words)) or has_money

def get_response(user_query: str) -> str:
    query_type = classify_query(user_query)

    try:
        results = collection.query(
            query_texts=[user_query],
            n_results=5,
            include=["documents", "distances"]
        )
        context = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
    except Exception as e:
        print(f"[Vector Query Error] {e}")
        context, distances = [], []

    context_confidence = bool(distances) and distances[0] < 0.8
    has_context = bool(context)

    system_msg = (
        "You are a gold pricing analyst. Be detailed, use data, and never speculate beyond the data. "
        f"Always include the nudge: '{NUDGE}'"
    )

    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }

    if query_type == "specific_price":
        matched_years = [int(y) for y in re.findall(r"20\d{2}", user_query)]
        matched_months = [m.lower() for m in re.findall(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", user_query.lower())]

        filt_df = df.copy()
        if matched_years:
            filt_df = filt_df[filt_df['year'] == matched_years[0]]
        if matched_months:
            filt_df = filt_df[filt_df['month'] == month_map[matched_months[0]]]

        if not filt_df.empty:
            avg_price = round(filt_df[PRICE_COL].mean())
            month_name = matched_months[0].capitalize() if matched_months else ""
            year_val = matched_years[0] if matched_years else ""
            return f"In {month_name} {year_val}, the average gold price was ₹{avg_price} with daily highs and lows ranging across the month.\n\n{NUDGE}"
        else:
            return "Sorry, I couldn't find gold prices for that time period."

    elif query_type == "best_month":
        monthly_avg = df.groupby(["year", "month"])[PRICE_COL].mean().reset_index()
        best_row = monthly_avg.sort_values(PRICE_COL).iloc[0]
        year, month, price = int(best_row["year"]), int(best_row["month"]), round(best_row[PRICE_COL])
        month_name = pd.to_datetime(f"{year}-{month}-01").strftime("%B")
        return f"The lowest average monthly gold price was ₹{price:,} in {month_name} {year}.\n\n{NUDGE}"

    elif query_type == "trend_analysis":
        df_sorted = df.sort_values("date")
        if len(df_sorted) < 30:
            return f"Not enough data for a 30-day moving average.\n\n{NUDGE}"
        df_sorted['rolling'] = df_sorted[PRICE_COL].rolling(window=30).mean()
        start = df_sorted['rolling'].iloc[29]
        end = df_sorted['rolling'].iloc[-1]
        if pd.isna(start) or pd.isna(end):
            return f"Not enough data for a 30-day moving average.\n\n{NUDGE}"
        delta = round(((end - start) / start) * 100, 2)
        trend = "increased 📈" if delta > 0 else ("decreased 📉" if delta < 0 else "remained stable")
        prompt = f"From {df_sorted['date'].min().date()} to {df_sorted['date'].max().date()}, gold prices {trend} by {abs(delta)}% based on the 30-day moving average.\n\n{NUDGE}"
        return chat_complete(system_msg, prompt)

    elif query_type == "prediction":
        prompt = (
            "Gold prices are influenced by interest rates, inflation, and global markets. "
            "While future prices can't be guaranteed, historical stability suggests value preservation."
            f"\n\n{NUDGE}"
        )
        return chat_complete(system_msg, prompt)

    elif query_type == "generic_gold":
        if has_context and context_confidence:
            prompt = (
                "Here are a few recent gold price updates from the dataset:\n" + "\n".join(context) + f"\n\n{NUDGE}"
            )
            return chat_complete(system_msg, prompt)
        else:
            return f"Gold is widely regarded as a hedge against inflation and currency volatility. {NUDGE}"

    else:
        return chat_complete(system_msg, f"User asked: {user_query}\n\nOnly answer gold price queries.")

OR_BASE_URL = OLLAMA_BASE_URL
