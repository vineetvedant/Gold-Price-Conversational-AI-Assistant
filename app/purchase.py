# app/purchase.py
from fastapi import FastAPI, HTTPException, status, Header
from pydantic import BaseModel, Field
from uuid import uuid4
from decimal import Decimal, InvalidOperation, ROUND_DOWN, getcontext
from pathlib import Path
from typing import Optional, List
import csv
from datetime import datetime, timezone
import os
import threading

app = FastAPI(title="Gold Purchase API (API 2)", version="1.1.0")

# --- Configuration (env overrides keep this file unchanged) ---
DEFAULT_PRICE_PER_GRAM = Decimal(os.getenv("DEFAULT_PRICE_PER_GRAM", "6500.00"))  # INR/gram fallback
GOLD_PRICE_CSV = Path(os.getenv("GOLD_PRICE_CSV", "app/data/Gold Price.csv"))     # input price file (optional)
TRANSACTIONS_CSV = Path(os.getenv("TRANSACTIONS_CSV", "app/data/transactions.csv"))  # output "database"

# If your CSV price is per 10g (common in India), set PRICE_UNIT=10g; otherwise "gram"
PRICE_UNIT = os.getenv("PRICE_UNIT", "gram").lower()  # "gram" | "10g"

CSV_HEADERS = [
    "transaction_id",
    "user_id",
    "purchase_amount_inr",
    "gold_amount_grams",
    "price_per_gram",
    "timestamp",
    "status",
]

# better precision defaults for money-ish math
getcontext().prec = 28

# in-process write lock (good enough for a single-uvicorn process)
_WRITE_LOCK = threading.Lock()


# --- Models ---
class PurchaseRequest(BaseModel):
    userId: str = Field(..., min_length=1, description="User identifier")
    amountInr: Decimal = Field(..., gt=Decimal("0"), description="Amount to spend in INR")

    class Config:
        json_schema_extra = {"example": {"userId": "user_123", "amountInr": 10000}}


class TransactionDetails(BaseModel):
    transactionId: str
    userId: str
    amountSpentInr: Decimal
    goldPurchasedGrams: Decimal
    pricePerGram: Decimal
    timestamp: str  # ISO 8601 UTC


class PurchaseResponse(BaseModel):
    status: str
    message: str
    transactionDetails: TransactionDetails


# --- Helpers ---
def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_transactions_csv_exists() -> None:
    """Create the transactions CSV with header if missing or empty."""
    _ensure_parent(TRANSACTIONS_CSV)
    if not TRANSACTIONS_CSV.exists() or TRANSACTIONS_CSV.stat().st_size == 0:
        with TRANSACTIONS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)


def _try_parse_decimal(value: str) -> Optional[Decimal]:
    try:
        if value is None:
            return None
        # Strip common noise like % or commas if ever present
        value = str(value).replace(",", "").replace("%", "").strip()
        return Decimal(value)
    except (InvalidOperation, TypeError, ValueError):
        return None


def get_latest_gold_price_from_csv() -> Optional[Decimal]:
    """
    Reads GOLD_PRICE_CSV (if present) and returns the most recent price.
    Logic:
      1) If a 'date' column exists, pick the row with the MAX date.
      2) Otherwise, use the last non-empty numeric value encountered.
    Candidate price column names (case-insensitive): price_per_gram, price, inr_per_gram, priceinrpergram
    """
    if not GOLD_PRICE_CSV.exists():
        return None

    with GOLD_PRICE_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None

        fields_lower = [c.lower() for c in reader.fieldnames]
        date_key = None
        for c in reader.fieldnames:
            if c.lower() == "date":
                date_key = c
                break

        candidates_lower = {"price_per_gram", "price", "inr_per_gram", "priceinrpergram"}

        # Map actual field names which match candidates (preserve original key)
        price_keys = [c for c in reader.fieldnames if c.lower() in candidates_lower]

        if date_key:
            # Track best (latest) date row
            best_date: Optional[datetime] = None
            best_price: Optional[Decimal] = None
            for row in reader:
                # Parse date; tolerate many formats, fall back to string compare if needed
                raw_date = row.get(date_key)
                parsed_date: Optional[datetime] = None
                if raw_date:
                    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%b %d, %Y", "%d-%b-%Y"):
                        try:
                            parsed_date = datetime.strptime(raw_date.strip(), fmt)
                            break
                        except Exception:
                            continue
                # If no price keys, try last column
                price_val: Optional[Decimal] = None
                if price_keys:
                    for pk in price_keys:
                        price_val = _try_parse_decimal(row.get(pk))
                        if price_val is not None:
                            break
                else:
                    # last column fallback
                    last_col_key = reader.fieldnames[-1]
                    price_val = _try_parse_decimal(row.get(last_col_key))

                if price_val is None:
                    continue

                if parsed_date is not None:
                    if best_date is None or parsed_date > best_date:
                        best_date, best_price = parsed_date, price_val
                else:
                    # no usable date in this row; keep scanning
                    pass

            return best_price
        else:
            # No date column; take the last valid numeric encountered
            last_valid: Optional[Decimal] = None
            for row in reader:
                price_val: Optional[Decimal] = None
                if price_keys:
                    for pk in price_keys:
                        price_val = _try_parse_decimal(row.get(pk))
                        if price_val is not None:
                            last_valid = price_val
                            break
                else:
                    last_col_key = reader.fieldnames[-1]
                    price_val = _try_parse_decimal(row.get(last_col_key))
                    if price_val is not None:
                        last_valid = price_val
            return last_valid


def _normalize_to_price_per_gram(raw_price: Decimal) -> Decimal:
    """
    If your file is per 10g, convert to per-gram using PRICE_UNIT env.
    """
    if PRICE_UNIT in ("10g", "10gram", "10grams", "ten_grams"):
        return (raw_price / Decimal("10"))
    return raw_price


def determine_gold_price() -> Decimal:
    price = get_latest_gold_price_from_csv()
    if price is None:
        price = DEFAULT_PRICE_PER_GRAM
    return _normalize_to_price_per_gram(price)


def write_transaction_to_csv(
    transaction_id: str,
    user_id: str,
    purchase_amount_inr: Decimal,
    gold_amount_grams: Decimal,
    price_per_gram: Decimal,
    timestamp_iso: str,
    status_value: str = "completed",
) -> None:
    ensure_transactions_csv_exists()
    with _WRITE_LOCK:
        with TRANSACTIONS_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    transaction_id,
                    user_id,
                    f"{purchase_amount_inr:.2f}",
                    f"{gold_amount_grams:.4f}",
                    f"{price_per_gram:.2f}",
                    timestamp_iso,
                    status_value,
                ]
            )


# --- Endpoints ---
@app.post("/purchase", response_model=PurchaseResponse, status_code=status.HTTP_201_CREATED)
async def purchase(req: PurchaseRequest, x_idempotency_key: Optional[str] = Header(None)):
    """
    Completes a purchase and appends it to CSV.
    Optional: pass X-Idempotency-Key to the request header from your client if you want to
    dedupe at the client level (not stored here, but you can extend easily).
    """
    user_id = req.userId.strip()
    if not user_id:
        raise HTTPException(status_code=422, detail="userId must not be empty")

    price_per_gram = determine_gold_price()
    if price_per_gram <= 0:
        raise HTTPException(status_code=503, detail="Price unavailable")

    # Compute gold amount (round down to 4 dp)
    gold_amount = (req.amountInr / price_per_gram).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)

    # Basic guardrails (optional)
    if gold_amount <= 0:
        raise HTTPException(status_code=400, detail="Amount too small for a non-zero gold quantity")

    # Create transaction metadata
    transaction_id = str(uuid4())
    timestamp_iso = datetime.now(timezone.utc).isoformat()

    # Persist to CSV (append)
    write_transaction_to_csv(
        transaction_id=transaction_id,
        user_id=user_id,
        purchase_amount_inr=req.amountInr.quantize(Decimal("0.01"), rounding=ROUND_DOWN),
        gold_amount_grams=gold_amount,
        price_per_gram=price_per_gram.quantize(Decimal("0.01"), rounding=ROUND_DOWN),
        timestamp_iso=timestamp_iso,
        status_value="completed",
    )

    details = TransactionDetails(
        transactionId=transaction_id,
        userId=user_id,
        amountSpentInr=req.amountInr.quantize(Decimal("0.01"), rounding=ROUND_DOWN),
        goldPurchasedGrams=gold_amount,
        pricePerGram=price_per_gram.quantize(Decimal("0.01"), rounding=ROUND_DOWN),
        timestamp=timestamp_iso,
    )

    return PurchaseResponse(
        status="success",
        message="Gold purchase successful!",
        transactionDetails=details,
    )


@app.get("/price")
async def get_price():
    """Expose the effective price per gram the service will use."""
    p = determine_gold_price()
    return {
        "pricePerGram": f"{p:.2f}",
        "unit": "gram",
        "source": str(GOLD_PRICE_CSV) if GOLD_PRICE_CSV.exists() else "default",
        "assumedPriceUnit": PRICE_UNIT,
    }


@app.get("/health")
async def health():
    return {
        "ok": True,
        "transactionsCsv": str(TRANSACTIONS_CSV),
        "priceCsv": str(GOLD_PRICE_CSV),
        "priceUnit": PRICE_UNIT,
    }
