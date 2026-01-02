import sqlite3
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]   # repo root
DB_PATH = BASE_DIR / "billing.db"
INDEX_PATH = BASE_DIR / "artifacts" / "billing.faiss"
IDS_PATH = BASE_DIR / "artifacts" / "billing_ids.npy"
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # fast + small

def row_to_text(r) -> str:
    # Keep it semantic: event + notes are most useful.
    # Month helps for "last month" retrieval; bill_id helps traceability.
    return (
        f"Bill month: {r['bill_month']}\n"
        f"Bill ID: {r['bill_id']}\n"
        f"Event: {r['event_name']}\n"
        f"Charge: {r['charge']}\n"
        f"Notes: {r['verbiage']}"
    )

def main():
    model = SentenceTransformer(EMBED_MODEL)

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    rows = cur.execute("""
      SELECT id, mtn, cust_id, bill_id, bill_month, event_name, charge, total_charges, verbiage, acct_scope
      FROM billing_events
      ORDER BY id
    """).fetchall()
    con.close()

    if not rows:
        raise RuntimeError("No rows found in SQLite. Run your SQLite build/seed script first.")

    texts = [row_to_text(r) for r in rows]
    ids = np.array([r["id"] for r in rows], dtype=np.int64)

    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim because vectors are normalized
    index.add(vectors)

    faiss.write_index(index, str(INDEX_PATH))
    np.save(str(IDS_PATH), ids)

    print(f"✅ FAISS index saved: {INDEX_PATH}")
    print(f"✅ ID map saved: {IDS_PATH}")
    print(f"✅ Total vectors: {index.ntotal}, dim: {dim}")

if __name__ == "__main__":
    main()
