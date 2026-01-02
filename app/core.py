import re
import sqlite3
import numpy as np
import faiss
import requests
from pathlib import Path
from sentence_transformers import SentenceTransformer

try:
    from app.prompts import SYSTEM_PROMPT
except Exception:
    from prompts import SYSTEM_PROMPT

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "billing.db"
INDEX_PATH = BASE_DIR / "artifacts" / "billing.faiss"
IDS_PATH = BASE_DIR / "artifacts" / "billing_ids.npy"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

_model = None
_index = None
_idmap = None

# Optional reranker (safe fallback)
_cross = None
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _normalize_mtn(mtn: str) -> str:
    m = str(mtn).strip()
    if m.startswith("+"):
        m = m[1:]
    return m


def load_assets():
    global _model, _index, _idmap, _cross
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    if _index is None:
        _index = faiss.read_index(str(INDEX_PATH))
    if _idmap is None:
        _idmap = np.load(str(IDS_PATH)).astype(np.int64)

    # CrossEncoder is optional; only load if available
    if _cross is None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            _cross = CrossEncoder(RERANK_MODEL)
        except Exception:
            _cross = False  # sentinel for "not available"


def get_available_months(mtn: str, cust_id: str) -> list[str]:
    mtn_norm = _normalize_mtn(mtn)
    cust_norm = str(cust_id).strip()

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    months = cur.execute(
        """
        SELECT DISTINCT bill_month
        FROM billing_events
        WHERE mtn = ? AND cust_id = ?
        ORDER BY bill_month DESC
        """,
        (mtn_norm, cust_norm),
    ).fetchall()
    con.close()

    return [m[0] for m in months]


def load_rows_for_month(mtn: str, cust_id: str, bill_month: str):
    mtn_norm = _normalize_mtn(mtn)
    cust_norm = str(cust_id).strip()
    month_norm = str(bill_month).strip()

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT *
        FROM billing_events
        WHERE mtn = ? AND cust_id = ? AND bill_month = ?
        """,
        (mtn_norm, cust_norm, month_norm),
    ).fetchall()
    con.close()
    return rows


def _aggregate(rows):
    total_charges = float(rows[0]["total_charges"])
    breakdown = {}
    for r in rows:
        ev = r["event_name"]
        breakdown[ev] = breakdown.get(ev, 0.0) + float(r["charge"])
    return total_charges, breakdown


def _top_items(breakdown: dict, n: int = 3):
    return sorted(breakdown.items(), key=lambda x: x[1], reverse=True)[:n]


def _fmt_money(x: float) -> str:
    return f"${x:.2f}"


def deterministic_compare(rows_curr, rows_prev, curr_month: str, prev_month: str) -> str:
    total_c, br_c = _aggregate(rows_curr)
    total_p, br_p = _aggregate(rows_prev)
    delta = total_c - total_p

    keys = set(br_c) | set(br_p)
    deltas = []
    for k in keys:
        c = br_c.get(k, 0.0)
        p = br_p.get(k, 0.0)
        deltas.append((k, c, p, c - p))
    deltas.sort(key=lambda x: abs(x[3]), reverse=True)

    lines = []
    lines.append(
        f"Total: {prev_month} {_fmt_money(total_p)} → {curr_month} {_fmt_money(total_c)} (Δ {_fmt_money(delta)})"
    )
    lines.append("")
    lines.append("Biggest changes:")
    for k, c, p, d in deltas[:5]:
        sign = "+" if d >= 0 else "-"
        lines.append(f"- {k}: {_fmt_money(p)} → {_fmt_money(c)} (Δ {sign}{_fmt_money(abs(d))})")
    return "\n".join(lines)


def deterministic_why_high(rows_curr) -> str:
    total, breakdown = _aggregate(rows_curr)
    top = _top_items(breakdown, n=4)

    lines = []
    lines.append(f"Total charges: {_fmt_money(total)}")
    lines.append("")
    lines.append("Top drivers this bill:")
    for k, v in top:
        lines.append(f"- {k}: {_fmt_money(v)}")
    lines.append("")
    lines.append(
        "If something (credits/discounts/extra fees) is not listed above, it is not present in the data for this bill."
    )
    return "\n".join(lines)


def build_structured_context(rows) -> str:
    bill_id = rows[0]["bill_id"]
    bill_month = rows[0]["bill_month"]
    total, breakdown = _aggregate(rows)

    lines = [
        f"bill_id: {bill_id}",
        f"bill_month: {bill_month}",
        f"total_charges: {total}",
        "line_items:",
    ]
    for ev, amt in breakdown.items():
        lines.append(f"  - event_name: {ev}, amount: {amt}")
    return "\n".join(lines)


def build_compare_context(rows_curr, rows_prev, curr_month: str, prev_month: str) -> str:
    total_c, br_c = _aggregate(rows_curr)
    total_p, br_p = _aggregate(rows_prev)

    all_keys = sorted(set(br_c.keys()) | set(br_p.keys()))
    lines = []
    lines.append(f"current_bill_month: {curr_month}, total_charges: {total_c}")
    lines.append(f"previous_bill_month: {prev_month}, total_charges: {total_p}")
    lines.append(f"delta_total: {total_c - total_p}")
    lines.append("line_item_deltas:")
    for k in all_keys:
        c = br_c.get(k, 0.0)
        p = br_p.get(k, 0.0)
        lines.append(f"  - event_name: {k}, current: {c}, previous: {p}, delta: {c - p}")
    return "\n".join(lines)


def semantic_retrieve(question: str, mtn: str, cust_id: str, top_k: int = 5, oversample: int = 80) -> list[dict]:
    """
    Returns list of dicts with fields: bill_month, event_name, charge, verbiage
    (Filtered to account)
    """
    load_assets()
    mtn_norm = _normalize_mtn(mtn)
    cust_norm = str(cust_id).strip()

    q_vec = _model.encode([question], normalize_embeddings=True).astype(np.float32)
    _, idxs = _index.search(q_vec, oversample)
    candidate_ids = _idmap[idxs[0]]

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    out = []
    for row_id in candidate_ids:
        r = cur.execute(
            "SELECT mtn, cust_id, bill_month, event_name, charge, verbiage FROM billing_events WHERE id = ?",
            (int(row_id),),
        ).fetchone()
        if not r:
            continue
        if str(r["mtn"]).strip() != mtn_norm or str(r["cust_id"]).strip() != cust_norm:
            continue

        out.append(
            {
                "bill_month": str(r["bill_month"]),
                "event_name": str(r["event_name"]),
                "charge": float(r["charge"]),
                "verbiage": str(r["verbiage"] or "").strip(),
            }
        )
        if len(out) >= top_k:
            break

    con.close()
    return out


def _ensure_fts(con: sqlite3.Connection) -> bool:
    """
    Try to ensure an FTS5 table exists for keyword search.
    If FTS5 isn't available, return False (safe fallback).
    """
    try:
        cur = con.cursor()
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS billing_events_fts
            USING fts5(event_name, verbiage, content='billing_events', content_rowid='id')
            """
        )
        # Build/rebuild index safely (no-op if already ok)
        try:
            cur.execute("INSERT INTO billing_events_fts(billing_events_fts) VALUES('rebuild')")
        except Exception:
            pass
        con.commit()
        return True
    except Exception:
        return False


def keyword_retrieve(question: str, mtn: str, cust_id: str, top_k: int = 8) -> list[dict]:
    """
    Keyword retrieval via SQLite FTS5 (if available).
    Returns list of dicts similar to semantic_retrieve.
    Safe fallback: returns [] if FTS isn't available.
    """
    mtn_norm = _normalize_mtn(mtn)
    cust_norm = str(cust_id).strip()

    q = str(question).strip()
    if not q:
        return []

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    if not _ensure_fts(con):
        con.close()
        return []

    terms = [t for t in q.replace('"', "").split() if len(t) > 2]
    if not terms:
        con.close()
        return []

    match_expr = " OR ".join(terms[:8])

    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT b.bill_month, b.event_name, b.charge, b.verbiage
        FROM billing_events_fts f
        JOIN billing_events b ON b.id = f.rowid
        WHERE f MATCH ?
          AND b.mtn = ? AND b.cust_id = ?
        LIMIT ?
        """,
        (match_expr, mtn_norm, cust_norm, int(top_k)),
    ).fetchall()
    con.close()

    out = []
    for r in rows:
        out.append(
            {
                "bill_month": str(r["bill_month"]),
                "event_name": str(r["event_name"]),
                "charge": float(r["charge"]),
                "verbiage": str(r["verbiage"] or "").strip(),
            }
        )
    return out


def hybrid_retrieve(question: str, mtn: str, cust_id: str, k: int = 6) -> list[dict]:
    """
    Hybrid: keyword + semantic -> merge -> optional rerank -> return top k.
    Safe: if keyword/ rerank not available, still works.
    """
    sem = semantic_retrieve(question, mtn, cust_id, top_k=10, oversample=80)
    kw = keyword_retrieve(question, mtn, cust_id, top_k=10)

    seen = set()
    cand = []
    for item in kw + sem:
        key = (item.get("bill_month"), item.get("event_name"), item.get("verbiage"))
        if key in seen:
            continue
        seen.add(key)
        cand.append(item)

    if not cand:
        return []

    load_assets()
    if _cross and _cross is not False:
        try:
            pairs = [(question, f"{c['event_name']}. {c['verbiage']}") for c in cand]
            scores = _cross.predict(pairs)
            order = np.argsort(-np.array(scores))
            cand = [cand[i] for i in order[:k]]
            return cand
        except Exception:
            pass

    return cand[:k]


def _call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 120},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    resp = data.get("response")
    return str(resp).strip() if resp else ""


# ---------------------------
# N-month compare helpers (NEW)
# ---------------------------
_NUM_WORDS = {"three": 3, "four": 4, "five": 5, "six": 6}


def _extract_n_months_request(q_low: str) -> int | None:
    """
    Detect requests like:
      - "compare my last 3 months bill"
      - "last 4 months"
      - "past five bills"
    Returns N if N >= 3 else None.
    """
    t = (q_low or "").lower().strip()

    m = re.search(r"(?:last|past|previous)\s+(\d+)\s+(?:month|months|bill|bills)", t)
    if m:
        try:
            n = int(m.group(1))
            return n if n >= 3 else None
        except Exception:
            pass

    m2 = re.search(r"compare\s+(?:my\s+)?(?:last\s+)?(\d+)\s+(?:month|months|bill|bills)", t)
    if m2:
        try:
            n = int(m2.group(1))
            return n if n >= 3 else None
        except Exception:
            pass

    for w, n in _NUM_WORDS.items():
        if re.search(rf"(?:last|past|previous)\s+{w}\s+(?:month|months|bill|bills)", t):
            return n

    return None


def _n_months_compare_summary(
    months_desc: list[str],
    totals: dict[str, float],
    breakdowns: dict[str, dict],
    top_k_changes: int = 8,
) -> str:
    """
    months_desc is DESC (newest first), e.g. ["202511","202510","202509"]
    """
    lines = ["Totals by month:"]
    for m in months_desc:
        lines.append(f"- {m}: {_fmt_money(float(totals[m]))}")

    lines.append("")
    lines.append("Month-over-month change:")
    for i in range(len(months_desc) - 1):
        newer = months_desc[i]
        older = months_desc[i + 1]
        d = float(totals[newer]) - float(totals[older])
        sign = "+" if d >= 0 else "-"
        lines.append(f"- {older} → {newer}: {sign}{_fmt_money(abs(d))}")

    latest = months_desc[0]
    oldest = months_desc[-1]
    br_latest = breakdowns[latest]
    br_oldest = breakdowns[oldest]

    keys = set(br_latest) | set(br_oldest)
    deltas = []
    for k in keys:
        p = float(br_oldest.get(k, 0.0))
        c = float(br_latest.get(k, 0.0))
        deltas.append((k, p, c, c - p))
    deltas.sort(key=lambda x: abs(x[3]), reverse=True)

    lines.append("")
    lines.append(f"Biggest line-item changes ({oldest} → {latest}):")
    for k, p, c, d in deltas[:top_k_changes]:
        sign = "+" if d >= 0 else "-"
        lines.append(f"- {k}: {_fmt_money(p)} → {_fmt_money(c)} (Δ {sign}{_fmt_money(abs(d))})")

    return "\n".join(lines)


# ---------------------------
# N-month breakdown helpers (NEW)
# ---------------------------
_BREAKDOWN_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}


def _parse_n_months_breakdown(q_low: str) -> int | None:
    """
    Detect breakdown requests like:
      - "breakdown my last 4 months bills"
      - "give me breakdown of last 4 months bill"
      - "break down for last four months"
      - "itemize last 3 bills"
    Returns N if N >= 2 else None.
    """
    t = (q_low or "").lower().strip()

    # numeric + months (optionally followed by "bill(s)")
    m = re.search(
        r"(?:break\s*down|breakdown|itemize)\s+"
        r"(?:of\s+|for\s+)?"
        r"(?:my\s+)?"
        r"(?:last|past|previous)\s+"
        r"(\d+)\s+"
        r"(?:month|months)"
        r"(?:\s+(?:bill|bills))?\b",
        t,
    )
    if m:
        try:
            n = int(m.group(1))
            return max(2, min(n, 12))
        except Exception:
            pass

    # numeric + bills (no "months" word)
    m2 = re.search(
        r"(?:break\s*down|breakdown|itemize)\s+"
        r"(?:of\s+|for\s+)?"
        r"(?:my\s+)?"
        r"(?:last|past|previous)\s+"
        r"(\d+)\s+"
        r"(?:bill|bills)\b",
        t,
    )
    if m2:
        try:
            n = int(m2.group(1))
            return max(2, min(n, 12))
        except Exception:
            pass

    # word-number + months (optionally followed by "bill(s)")
    m3 = re.search(
        r"(?:break\s*down|breakdown|itemize)\s+"
        r"(?:of\s+|for\s+)?"
        r"(?:my\s+)?"
        r"(?:last|past|previous)\s+"
        r"([a-z]+)\s+"
        r"(?:month|months)"
        r"(?:\s+(?:bill|bills))?\b",
        t,
    )
    if m3:
        w = m3.group(1).strip()
        if w in _BREAKDOWN_NUM_WORDS:
            n = _BREAKDOWN_NUM_WORDS[w]
            return max(2, min(n, 12))

    # word-number + bills (no "months" word)
    m4 = re.search(
        r"(?:break\s*down|breakdown|itemize)\s+"
        r"(?:of\s+|for\s+)?"
        r"(?:my\s+)?"
        r"(?:last|past|previous)\s+"
        r"([a-z]+)\s+"
        r"(?:bill|bills)\b",
        t,
    )
    if m4:
        w = m4.group(1).strip()
        if w in _BREAKDOWN_NUM_WORDS:
            n = _BREAKDOWN_NUM_WORDS[w]
            return max(2, min(n, 12))

    return None


def _month_breakdown_dict(mtn: str, cust_id: str, month: str) -> tuple[float, dict]:
    rows = load_rows_for_month(mtn, cust_id, month)
    if not rows:
        return 0.0, {}
    total, br = _aggregate(rows)
    return float(total), br


def deterministic_breakdown_last_n_months(
    mtn: str,
    cust_id: str,
    months_desc: list[str],
    n: int,
    top_items: int = 6,
    show_all_items: bool = False,
) -> str:
    """
    True 'breakdown' (NOT comparison):
      - each month: total + top items (or all)
    months_desc is DESC (latest->older). We display oldest->latest.
    """
    n = min(n, len(months_desc))
    window_desc = months_desc[:n]          # latest -> older
    window = list(reversed(window_desc))   # oldest -> latest

    lines = [f"🟧 Totals + line-items for last {n} months:"]
    lines.append("")

    for mo in window:
        total, br = _month_breakdown_dict(mtn, cust_id, mo)
        lines.append(f"{mo} — Total: {_fmt_money(total)}")

        if not br:
            lines.append("- No line items found.")
            lines.append("")
            continue

        items = sorted(br.items(), key=lambda x: x[1], reverse=True)
        if not show_all_items:
            items_to_show = items[:top_items]
        else:
            items_to_show = items

        for ev, amt in items_to_show:
            lines.append(f"- {ev}: {_fmt_money(float(amt))}")

        if not show_all_items and len(items) > top_items:
            lines.append(f"- …and {len(items) - top_items} more line items")

        lines.append("")

    return "\n".join(lines).strip()


def answer(
    question: str,
    mtn: str,
    cust_id: str,
    last_topic: str | None = None,
    last_mode: str | None = None,  # "current" | "previous" | "compare"
) -> tuple[str, str | None, str | None]:
    """
    Returns: (response_text, new_last_topic, new_last_mode)

    Enhancements (without breaking existing):
    - Prevents topic leakage across unrelated questions
    - Better follow-up detection
    - Compare supports "full/complete/all line items"
    - Adds hybrid retrieval (FTS5 + FAISS) for definition/explain style questions
    - NEW: If user asks for "last N months/bills" (N>=3), return an N-month comparison
    - NEW: If user asks "breakdown my last N months", return month-by-month line items (NOT comparison)
    """

    q = str(question).strip()
    q_low = q.lower()

    # NEW: detect N-month request early (existing)
    n_months_req = _extract_n_months_request(q_low)

    # NEW: detect N-month BREAKDOWN early (does not affect compare)
    n_months_breakdown = _parse_n_months_breakdown(q_low)

    # ---- greetings ----
    if q_low in {"hi", "hello", "hey", "yo"}:
        return (
            "Hi 👋 Try: 'Do I have federal tax this month?', 'Break down my bill', or 'Compare my last 2 bills (full comparison)'.",
            last_topic,
            last_mode,
        )

    months = get_available_months(mtn, cust_id)
    if not months:
        return ("I couldn't find any billing records for that MTN and customer ID.", last_topic, last_mode)

    latest = months[0]
    prev = months[1] if len(months) > 1 else None

    # ---- topic extraction ----
    topic_map = {
        # taxes
        "federal tax": "federal tax",
        "federal charges": "federal tax",
        "federal": "federal tax",
        "state tax": "state tax",
        "state charges": "state tax",
        "taxes": "tax",
        "tax": "tax",
        # surcharges / fees
        "late fee": "late fee",
        "late fees": "late fee",
        "late payment": "late fee",
        "payment fee": "payment fee",
        "returned payment": "returned payment",
        "fee": "fee",
        "fees": "fee",
        # common bill items
        "surcharge": "surcharge",
        "surcharges": "surcharge",
        "plan charge": "plan charge",
        "plan": "plan charge",
        "device payment": "device payment",
        "device": "device payment",
    }

    def extract_topic(text: str) -> str | None:
        for k in topic_map:
            if k in text:
                return topic_map[k]
        return None

    def is_followup(text: str) -> bool:
        t = text.strip().lower()
        followup_phrases = [
            "what about",
            "how about",
            "and last month",
            "and this month",
            "in last month",
            "in previous month",
            "previous month",
            "last month",
            "current month",
            "this month",
            "that one",
            "that",
            "those",
            "it",
            "same for",
            "also",
            "ok and",
        ]
        if len(t.split()) <= 6 and any(p in t for p in followup_phrases):
            return True
        if any(t.startswith(p) for p in ["what about", "how about", "and what about"]):
            return True
        return False

    topic_explicit = extract_topic(q_low)
    followup = is_followup(q_low)

    # ---- mode extraction ----
    mentions_prev = any(x in q_low for x in ["last month", "previous month", "prior month"])
    mentions_current = any(x in q_low for x in ["this month", "current month", "latest month", "this bill"])

    compare_intent = any(
        x in q_low
        for x in [
            "compare",
            "compared",
            "difference",
            "delta",
            "increase",
            "decrease",
            "higher than",
            "lower than",
            "what changed",
            "change from",
            "last 2 bills",
            "two bills",
        ]
    ) or (n_months_req is not None)

    if mentions_prev and not compare_intent:
        mode = "previous"
    elif compare_intent:
        mode = "compare"
    elif mentions_current:
        mode = "current"
    else:
        mode = (last_mode if followup else "current")

    # ---- intent extraction ----
    breakdown_intent = any(
        x in q_low
        for x in ["break down", "breakdown", "itemize", "line items", "line-item", "show charges", "charges list"]
    )

    why_high_intent = any(
        x in q_low
        for x in ["why is my bill high", "why is bill high", "bill high", "why so high", "why is my bill so high"]
    )

    total_intent = any(
        x in q_low
        for x in [
            "total bill",
            "total charges",
            "what is my total",
            "what's my total",
            "how much is my bill",
            "bill total",
        ]
    )

    explain_intent = any(
        x in q_low
        for x in ["what is", "what does", "meaning of", "explain", "why am i charged", "what is this charge"]
    )

    full_compare_intent = compare_intent and any(x in q_low for x in ["full", "complete", "all", "entire", "everything"])

    presence_intent = (
        any(
            x in q_low
            for x in [
                "do i have",
                "is there",
                "are there",
                "any ",
                "did i get",
                "have i been charged",
                "charged for",
                "was i charged",
                "do i see",
            ]
        )
        or ("?" in q_low and (topic_explicit is not None or followup))
    )

    amount_intent = any(
        x in q_low for x in ["how much", "amount", "total of", "what is the amount", "what's the amount"]
    )

    delta_intent = any(x in q_low for x in ["change", "delta", "increase", "decrease", "difference"]) and (
        mode == "compare"
    )

    # ---- topic carry rules (prevents leakage) ----
    fresh_intent = breakdown_intent or why_high_intent or compare_intent or total_intent or explain_intent
    if topic_explicit:
        topic = topic_explicit
    else:
        topic = (last_topic if (followup and not fresh_intent) else None)

    # ---- NEW: N-month breakdown short-circuit (month-by-month line items) ----
    # Only triggers for "breakdown my last N months..." patterns. It does NOT affect "compare".
    if n_months_breakdown is not None:
        n = n_months_breakdown
        if len(months) < 1:
            return ("I couldn't find any billing data for that account.", last_topic, last_mode)
        if n > len(months):
            n = len(months)
        return (
            deterministic_breakdown_last_n_months(
                mtn=mtn,
                cust_id=cust_id,
                months_desc=months,
                n=n,
                top_items=6,
                show_all_items=False,
            ),
            last_topic,
            last_mode,
        )

    # ---- helpers ----
    def month_rows(month: str):
        return load_rows_for_month(mtn, cust_id, month)

    def find_matches(breakdown: dict, topic_kw: str):
        t = (topic_kw or "").lower().strip()
        matches = []
        for ev, amt in breakdown.items():
            ev_low = str(ev).lower()

            if t == "tax":
                if "tax" in ev_low:
                    matches.append((ev, amt))
                continue

            if t == "fee":
                if "fee" in ev_low or "penalty" in ev_low:
                    matches.append((ev, amt))
                continue

            if t == "late fee":
                if ("late" in ev_low and "fee" in ev_low) or ("late" in ev_low and "penalty" in ev_low):
                    matches.append((ev, amt))
                continue

            if t and t in ev_low:
                matches.append((ev, amt))

        return matches

    def month_presence_summary(month: str, topic_kw: str) -> str:
        rows = month_rows(month)
        if not rows:
            return f"{month}: I could not load bill details."
        total, breakdown = _aggregate(rows)
        matches = find_matches(breakdown, topic_kw)
        if not matches:
            return f"{month}: No '{topic_kw}' line item found. Total: {_fmt_money(total)}."
        s = sum(a for _, a in matches)
        names = ", ".join(ev for ev, _ in matches)
        return f"{month}: Yes — {names}: {_fmt_money(s)}. Total: {_fmt_money(total)}."

    def explain_from_verbiage() -> str:
        hits = hybrid_retrieve(q, mtn, cust_id, k=5)
        if not hits:
            return "I don’t see an explanation note for that in the available bill data."
        lines = ["Here’s what the bill notes say (grounded):"]
        for h in hits[:3]:
            vb = (h.get("verbiage") or "").strip()
            if not vb:
                continue
            lines.append(f"- {h['bill_month']} | {h['event_name']}: {vb}")
        return "\n".join(lines)

    # ---- deterministic actions ----

    # A) Compare mode
    if mode == "compare":
        # NEW: N-month compare (N >= 3) — does not change existing 2-month behavior
        if n_months_req is not None:
            if len(months) < 2:
                return (f"I only have one month of data ({latest}), so I can't compare.", last_topic, "compare")

            n = n_months_req
            if n > len(months):
                n = len(months)

            months_to_use = months[:n]  # DESC: newest -> oldest

            totals: dict[str, float] = {}
            breakdowns: dict[str, dict] = {}
            for m in months_to_use:
                rows_m = month_rows(m)
                if not rows_m:
                    return (f"I couldn't load bill details for {m}.", last_topic, "compare")
                tot, br = _aggregate(rows_m)
                totals[m] = float(tot)
                breakdowns[m] = br

            # If user asked a specific topic along with N-month compare, show topic trend
            if topic and (presence_intent or amount_intent or delta_intent):
                lines = [f"{topic.title()} by month:"]
                per_month = {}
                for m in months_to_use:
                    matches = find_matches(breakdowns[m], topic)
                    s = sum(a for _, a in matches) if matches else 0.0
                    per_month[m] = float(s)
                    lines.append(f"- {m}: {_fmt_money(s)}")

                lines.append("")
                lines.append("Month-over-month change:")
                for i in range(len(months_to_use) - 1):
                    newer = months_to_use[i]
                    older = months_to_use[i + 1]
                    d = per_month[newer] - per_month[older]
                    sign = "+" if d >= 0 else "-"
                    lines.append(f"- {older} → {newer}: {sign}{_fmt_money(abs(d))}")

                return ("\n".join(lines), topic, "compare")

            return (_n_months_compare_summary(months_to_use, totals, breakdowns), topic, "compare")

        # ---- existing 2-month compare logic (unchanged) ----
        if not prev:
            return (
                f"I only have one month of data ({latest}), so I can't compare to last month.",
                last_topic,
                "compare",
            )

        rows_c = month_rows(latest)
        rows_p = month_rows(prev)
        if not rows_c or not rows_p:
            return (f"I cannot compare because I am missing data for {latest} or {prev}.", last_topic, "compare")

        total_c, br_c = _aggregate(rows_c)
        total_p, br_p = _aggregate(rows_p)
        delta_total = total_c - total_p

        # Topic-specific compare (e.g., "federal tax change?")
        if topic and (presence_intent or amount_intent or delta_intent):
            c_matches = find_matches(br_c, topic)
            p_matches = find_matches(br_p, topic)
            c_sum = sum(a for _, a in c_matches)
            p_sum = sum(a for _, a in p_matches)
            d = c_sum - p_sum

            if not c_matches and not p_matches:
                return (f"{topic.title()} is not present in {prev} or {latest}.", topic, "compare")

            return (
                f"{topic.title()} change: {prev} {_fmt_money(p_sum)} -> {latest} {_fmt_money(c_sum)} (Δ {_fmt_money(d)}). "
                f"Total bill change: Δ {_fmt_money(delta_total)}.",
                topic,
                "compare",
            )

        # Full compare: ALL line items (including unchanged)
        if full_compare_intent:
            keys = sorted(set(br_c) | set(br_p))
            lines = [
                f"Total: {prev} {_fmt_money(total_p)} -> {latest} {_fmt_money(total_c)} (Δ {_fmt_money(delta_total)})"
            ]
            lines.append("Line item changes:")
            for k in keys:
                p_amt = br_p.get(k, 0.0)
                c_amt = br_c.get(k, 0.0)
                d = c_amt - p_amt
                sign = "+" if d >= 0 else "-"
                lines.append(f"- {k}: {_fmt_money(p_amt)} -> {_fmt_money(c_amt)} (Δ {sign}{_fmt_money(abs(d))})")
            return ("\n".join(lines), topic, "compare")

        # Otherwise: compact compare summary (top changes)
        keys = set(br_c) | set(br_p)
        deltas = []
        for k in keys:
            c_amt = br_c.get(k, 0.0)
            p_amt = br_p.get(k, 0.0)
            deltas.append((k, p_amt, c_amt, c_amt - p_amt))
        deltas.sort(key=lambda x: abs(x[3]), reverse=True)

        lines = [f"Total: {prev} {_fmt_money(total_p)} -> {latest} {_fmt_money(total_c)} (Δ {_fmt_money(delta_total)})"]
        lines.append("Top changes:")
        for k, p_amt, c_amt, d in deltas[:6]:
            sign = "+" if d >= 0 else "-"
            lines.append(f"- {k}: {_fmt_money(p_amt)} -> {_fmt_money(c_amt)} (Δ {sign}{_fmt_money(abs(d))})")

        return ("\n".join(lines), topic, "compare")

    # B) Previous month only
    if mode == "previous":
        if not prev:
            return (f"I only have one month of data ({latest}).", last_topic, "previous")

        if topic and (presence_intent or amount_intent):
            return (month_presence_summary(prev, topic), topic, "previous")

        rows = month_rows(prev)
        if not rows:
            return (f"I could not load bill details for {prev}.", last_topic, "previous")

        total, breakdown = _aggregate(rows)

        if total_intent:
            return (f"{prev} total: {_fmt_money(total)}.", last_topic, "previous")

        if breakdown_intent:
            items = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)[:10]
            lines = [f"{prev} total: {_fmt_money(total)}", "Line items:"]
            for ev, amt in items:
                lines.append(f"- {ev}: {_fmt_money(amt)}")
            return ("\n".join(lines), last_topic, "previous")

        if why_high_intent:
            return (deterministic_why_high(rows), last_topic, "previous")

        if explain_intent:
            return (explain_from_verbiage(), last_topic, "previous")

        return (f"{prev} total: {_fmt_money(total)}. Try: 'Break down last month' or 'Compare to this month'.", last_topic, "previous")

    # C) Current/latest month
    rows = month_rows(latest)
    if not rows:
        return (f"I could not load bill details for {latest}.", last_topic, "current")

    total, breakdown = _aggregate(rows)

    if total_intent:
        return (f"{latest} total: {_fmt_money(total)}.", last_topic, "current")

    if topic and (presence_intent or amount_intent):
        return (month_presence_summary(latest, topic), topic, "current")

    if breakdown_intent:
        items = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)[:12]
        lines = [f"{latest} total: {_fmt_money(total)}", "Line items:"]
        for ev, amt in items:
            lines.append(f"- {ev}: {_fmt_money(amt)}")
        return ("\n".join(lines), last_topic, "current")

    if why_high_intent:
        return (deterministic_why_high(rows), last_topic, "current")

    if explain_intent:
        return (explain_from_verbiage(), last_topic, "current")

    # D) Fall back (short)
    return (
        f"For {latest}, total is {_fmt_money(total)}. Try: 'Why is my bill high?', 'Break down my bill', or 'Compare my last 2 bills (full comparison)'.",
        last_topic,
        "current",
    )
