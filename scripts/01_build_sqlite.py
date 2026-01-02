import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("billing.db")
CSV_PATH = Path("data/billing_events_sample.csv")

# Your schema:
# mtn STRING
# acc_no STRING
# cust_id STRING
# bill_id STRING
# event_name STRING
# charge NUMERIC
# total_charges NUMERIC
# bill_month STRING
# verbiage STRING

def main():
    df = pd.read_csv(CSV_PATH)

    # Type normalization
    df["mtn"] = df["mtn"].astype(str)
    df["acc_no"] = df["acc_no"].astype(str)
    df["cust_id"] = df["cust_id"].astype(str)
    df["bill_id"] = df["bill_id"].astype(str)
    df["event_name"] = df["event_name"].astype(str)
    df["bill_month"] = df["bill_month"].astype(str)
    df["verbiage"] = df["verbiage"].fillna("").astype(str)

    # NUMERIC -> float for SQLite
    df["charge"] = df["charge"].astype(float)
    df["total_charges"] = df["total_charges"].astype(float)

    # helper for scoping/filtering like Matching Engine restricts
    df["acct_scope"] = df["mtn"] + "|" + df["cust_id"] + "|" + df["bill_month"]

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS billing_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mtn TEXT,
        acc_no TEXT,
        cust_id TEXT,
        bill_id TEXT,
        event_name TEXT,
        charge REAL,
        total_charges REAL,
        bill_month TEXT,
        verbiage TEXT,
        acct_scope TEXT
    );
    """)

    # indexes for fast lookups
    cur.execute("CREATE INDEX IF NOT EXISTS idx_scope ON billing_events (acct_scope);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bill ON billing_events (bill_id, bill_month);")

    # rebuild fresh each time for demo
    cur.execute("DELETE FROM billing_events;")

    df.to_sql("billing_events", con, if_exists="append", index=False)

    con.commit()
    con.close()

    print(f"✅ Built SQLite DB: {DB_PATH}")
    print(f"✅ Rows inserted: {len(df)}")

if __name__ == "__main__":
    main()
