import sqlite3
import random
import string
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "billing.db"

# ---- Config defaults ----
DEFAULT_NUM_ACCOUNTS = 20
DEFAULT_MONTHS = ["202506", "202507", "202508", "202509", "202510", "202511"]  # keeps your existing months included
RANDOM_SEED = 42  # deterministic (same seed => same demo data)

# ---- Helpers ----
def _rand_digits(n: int) -> str:
    return "".join(random.choice(string.digits) for _ in range(n))

def _make_mtn(existing_mtns: set[str]) -> str:
    # SQLite stores MTN without '+', use 10 digits
    while True:
        mtn = "1" + _rand_digits(10)  # 11 digits like your example "12145550101"
        if mtn not in existing_mtns:
            return mtn

def _make_cust_id(i: int) -> str:
    return f"CUST{3000 + i}"

def _make_acc_no(i: int) -> str:
    return f"ACC{9000 + i}"

def _make_bill_id(mtn: str, bill_month: str) -> str:
    return f"BILL-{mtn[-4:]}-{bill_month}"

def _acct_scope(mtn: str, cust_id: str, bill_month: str) -> str:
    return f"{mtn}|{cust_id}|{bill_month}"

def _verbiage_for(event: str, bill_month: str) -> str:
    # short, realistic notes recruiters can search for
    notes = {
        "Plan Charge": "Monthly service plan base charge.",
        "Device Payment": "Monthly device installment for handset financing.",
        "Federal Tax": "Federal tax applied to eligible services.",
        "State Tax": "State tax applied based on service address.",
        "Surcharges": "Regulatory surcharges (e.g., 911, administrative recovery).",
        "Late Fee": "Late payment fee due to past-due balance.",
        "AutoPay Discount": "AutoPay discount applied for enrolled payment method.",
        "One-time Credit": "Courtesy credit applied to resolve a billing issue.",
        "International Charges": "International usage charges (calls/data) during travel.",
    }
    base = notes.get(event, "Billing line item.")
    return f"{base} (bill_month={bill_month})"

def _pick_events() -> list[tuple[str, float]]:
    """
    Returns list of (event_name, amount) for a synthetic bill.
    Total will be sum of these.
    """
    # Base recurring charges
    plan = random.choice([79.99, 89.99, 99.99, 109.99])
    device = random.choice([0.0, 22.91, 27.91, 34.84, 41.67])  # some customers have no device payment

    # Taxes & surcharges depend loosely on plan/device
    taxable_base = plan + device
    fed_tax = round(taxable_base * random.choice([0.00, 0.02, 0.03]), 2)  # sometimes 0
    state_tax = round(taxable_base * random.choice([0.00, 0.04, 0.05, 0.06]), 2)  # sometimes 0
    surcharges = round(random.uniform(6.50, 12.50), 2)

    items = [("Plan Charge", plan)]
    if device > 0:
        items.append(("Device Payment", device))

    # add taxes (only if > 0)
    if fed_tax > 0:
        items.append(("Federal Tax", fed_tax))
    if state_tax > 0:
        items.append(("State Tax", state_tax))

    items.append(("Surcharges", surcharges))

    # occasional fees/credits (to support queries like "late fee", "discount", "credit")
    roll = random.random()
    if roll < 0.18:
        items.append(("Late Fee", round(random.uniform(5.00, 15.00), 2)))
    elif roll < 0.30:
        items.append(("AutoPay Discount", -10.00))
    elif roll < 0.38:
        items.append(("One-time Credit", round(-random.uniform(5.00, 25.00), 2)))
    elif roll < 0.44:
        items.append(("International Charges", round(random.uniform(10.00, 55.00), 2)))

    return items

def seed_demo_data(num_accounts: int = DEFAULT_NUM_ACCOUNTS, months: list[str] = DEFAULT_MONTHS):
    random.seed(RANDOM_SEED)

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # existing accounts so we don't collide
    existing = cur.execute("SELECT DISTINCT mtn FROM billing_events").fetchall()
    existing_mtns = {r[0] for r in existing if r and r[0]}

    inserted_rows = 0
    inserted_bills = 0

    for i in range(num_accounts):
        mtn = _make_mtn(existing_mtns)
        existing_mtns.add(mtn)

        cust_id = _make_cust_id(i)
        acc_no = _make_acc_no(i)

        for bill_month in months:
            # Avoid inserting duplicate bill for same (mtn,cust_id,month)
            exists = cur.execute(
                """
                SELECT 1 FROM billing_events
                WHERE mtn=? AND cust_id=? AND bill_month=?
                LIMIT 1
                """,
                (mtn, cust_id, bill_month),
            ).fetchone()
            if exists:
                continue

            bill_id = _make_bill_id(mtn, bill_month)
            scope = _acct_scope(mtn, cust_id, bill_month)

            items = _pick_events()
            total = round(sum(amt for _, amt in items), 2)

            for event_name, charge in items:
                verbiage = _verbiage_for(event_name, bill_month)
                cur.execute(
                    """
                    INSERT INTO billing_events
                    (mtn, acc_no, cust_id, bill_id, event_name, charge, total_charges, bill_month, verbiage, acct_scope)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (mtn, acc_no, cust_id, bill_id, event_name, float(charge), float(total), bill_month, verbiage, scope),
                )
                inserted_rows += 1

            inserted_bills += 1

    con.commit()
    con.close()

    print("✅ Seed complete")
    print(f"Inserted bills: {inserted_bills}")
    print(f"Inserted rows : {inserted_rows}")
    print(f"DB           : {DB_PATH}")

if __name__ == "__main__":
    seed_demo_data()
