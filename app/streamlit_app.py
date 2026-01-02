import re
import streamlit as st
import sqlite3
from pathlib import Path
import pandas as pd

from core import answer  # must return (reply, last_topic, last_mode)

# --- Paths (repo root) ---
# Keeping DB path relative to repo root makes it easy to run locally or in a demo container.
BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "billing.db"


# ---------------------------
# SQLite helpers
# ---------------------------
def _norm_mtn(mtn: str) -> str:
    # DB stores MTN without "+", so normalize here too (users often paste +1...).
    return str(mtn or "").strip().lstrip("+")


def _norm_cust(cust_id: str) -> str:
    # Keep it minimal — just strip whitespace so lookups don’t fail on copy/paste.
    return str(cust_id or "").strip()


def _con():
    # Single place to open a connection; callers are responsible for closing.
    # (We keep it simple here; sqlite perf is fine for demo scale.)
    return sqlite3.connect(DB_PATH)


def load_sample_accounts(limit: int = 25) -> pd.DataFrame:
    """
    Distinct accounts (mtn, cust_id) so recruiters can try different inputs.
    """
    con = _con()
    q = """
    SELECT
      mtn,
      cust_id,
      COUNT(DISTINCT bill_month) AS months_count,
      GROUP_CONCAT(DISTINCT bill_month) AS months_available,
      MAX(bill_month) AS latest_month
    FROM billing_events
    GROUP BY mtn, cust_id
    ORDER BY latest_month DESC
    LIMIT ?
    """
    df = pd.read_sql_query(q, con, params=(limit,))
    con.close()

    if not df.empty and "months_available" in df.columns:

        def _sort_months(s):
            if s is None:
                return ""
            parts = [p.strip() for p in str(s).split(",") if p.strip()]
            parts = sorted(set(parts), reverse=True)
            return ", ".join(parts)

        # UI polish: month list ordering is noisy otherwise.
        df["months_available"] = df["months_available"].apply(_sort_months)

    return df


def account_months(mtn: str, cust_id: str) -> list[str]:
    # Returns bill months for the account in DESC order (latest first).
    con = _con()
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT DISTINCT bill_month
        FROM billing_events
        WHERE mtn = ? AND cust_id = ?
        ORDER BY bill_month DESC
        """,
        (_norm_mtn(mtn), _norm_cust(cust_id)),
    ).fetchall()
    con.close()
    return [r[0] for r in rows]


def month_total(mtn: str, cust_id: str, bill_month: str) -> float | None:
    # Total is repeated across events, so MAX() is a quick way to fetch it reliably.
    con = _con()
    cur = con.cursor()
    r = cur.execute(
        """
        SELECT MAX(total_charges)
        FROM billing_events
        WHERE mtn = ? AND cust_id = ? AND bill_month = ?
        """,
        (_norm_mtn(mtn), _norm_cust(cust_id), str(bill_month).strip()),
    ).fetchone()
    con.close()
    return float(r[0]) if r and r[0] is not None else None


def month_breakdown(mtn: str, cust_id: str, bill_month: str) -> pd.DataFrame:
    # Aggregates charges per event_name for a given month.
    con = _con()
    df = pd.read_sql_query(
        """
        SELECT event_name, SUM(charge) AS amount
        FROM billing_events
        WHERE mtn = ? AND cust_id = ? AND bill_month = ?
        GROUP BY event_name
        ORDER BY amount DESC
        """,
        con,
        params=(_norm_mtn(mtn), _norm_cust(cust_id), str(bill_month).strip()),
    )
    con.close()
    if not df.empty:
        df["amount"] = df["amount"].astype(float)
    return df


def compare_table(mtn: str, cust_id: str, curr_month: str, prev_month: str) -> pd.DataFrame:
    # Builds a simple prev-vs-current table at line-item level.
    # Note: Outer join + fillna lets us catch adds/removals (0 -> X or X -> 0).
    df_c = month_breakdown(mtn, cust_id, curr_month).rename(columns={"amount": "current"})
    df_p = month_breakdown(mtn, cust_id, prev_month).rename(columns={"amount": "previous"})

    df = pd.merge(df_p, df_c, on="event_name", how="outer").fillna(0.0)
    df["delta"] = df["current"] - df["previous"]
    df = df.sort_values(by="current", ascending=False).reset_index(drop=True)
    return df


def n_month_totals(mtn: str, cust_id: str, months_sel: list[str]) -> pd.DataFrame:
    """
    Return totals by month for selected months (chronological order in UI).
    """
    rows = []
    for m in months_sel:
        t = month_total(mtn, cust_id, m)
        rows.append({"bill_month": m, "total": float(t) if t is not None else 0.0})
    df = pd.DataFrame(rows)
    return df


def n_month_breakdowns(mtn: str, cust_id: str, months_sel: list[str]) -> pd.DataFrame:
    """
    Return a long table: month, event_name, amount
    """
    frames = []
    for m in months_sel:
        df = month_breakdown(mtn, cust_id, m)
        if df.empty:
            continue
        df = df.copy()
        df["bill_month"] = m
        frames.append(df[["bill_month", "event_name", "amount"]])
    if not frames:
        return pd.DataFrame(columns=["bill_month", "event_name", "amount"])
    return pd.concat(frames, ignore_index=True)


def account_event_names(mtn: str, cust_id: str) -> set[str]:
    # Used by the Verify tab to detect claims that don't exist for this account at all.
    # (Helpful when a bot says "International Roaming" but this account never had it.)
    con = _con()
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT DISTINCT event_name
        FROM billing_events
        WHERE mtn = ? AND cust_id = ?
        """,
        (_norm_mtn(mtn), _norm_cust(cust_id)),
    ).fetchall()
    con.close()
    return {str(r[0]).strip() for r in rows if r and r[0] is not None}


# ---------------------------
# Verify helpers (audit-style grounding)
# ---------------------------
# We keep this intentionally lightweight: it's not a perfect "fact checker",
# just a sanity audit to show the bot isn't making up bill line items.
_MONTH_RE = re.compile(r"\b20\d{4}\b")


def _extract_months_from_text(text: str, valid_months: list[str]) -> list[str]:
    # Pull YYYYMM mentions from the model reply (keeps only months that exist for the account).
    if not text:
        return []
    found = _MONTH_RE.findall(text)
    if not found:
        return []
    valid = set(valid_months)
    out = []
    seen = set()
    for m in found:
        if m in valid and m not in seen:
            out.append(m)
            seen.add(m)
    return out


def _infer_evidence_months(last_mode: str | None, question: str, reply: str, months_desc: list[str]) -> list[str]:
    """
    Pick the month(s) we should use as evidence for the current response.
    - Prefer explicit month mentions in the reply.
    - Otherwise fall back to the mode + available months.
    """
    # If the bot already printed YYYYMM, trust that and validate against those months.
    explicit = _extract_months_from_text(reply or "", months_desc)
    if explicit:
        return explicit

    if not months_desc:
        return []

    if last_mode == "previous":
        # "previous" means "last month" in our UX; if only one month exists, just use it.
        return [months_desc[1]] if len(months_desc) > 1 else [months_desc[0]]

    if last_mode == "compare":
        # Default to latest vs previous when we don't have explicit month labels in the response.
        if len(months_desc) >= 2:
            return [months_desc[1], months_desc[0]]
        return [months_desc[0]]

    # Current/latest
    return [months_desc[0]]


def _norm_name(s: str) -> str:
    # Verifier normalization: don't fail a match due to spacing/casing differences.
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# These are *not* bill line items; they show up in responses as headings.
# If we treat them as "claims", we'd inflate hallucination rate for no reason.
_EXCLUDE_CLAIMS = {
    "total",
    "total charges",
    "top changes",
    "biggest changes",
    "line item changes",
    "top drivers this bill",
    "top drivers",
    "totals by month",
    "month-over-month change",
    "month over month change",
    "line items",
    "line item",
}


def _extract_claimed_items(reply: str) -> list[str]:
    """
    Extract candidate line-item names from the model reply.

    We only look for "Name: <number>" shapes. That catches:
      - Plan Charge: 99.99
      - State Tax: 6.00
    and ignores generic sentences.

    (This is intentionally heuristic; good enough for demo-style grounding.)
    """
    if not reply:
        return []

    # Captures "Name: 12.34" even if embedded inside a sentence.
    # Example: "202511: Yes — State Tax: 6.00. Total: 115.50."
    pattern = re.compile(r"([A-Za-z][A-Za-z0-9 /&()\-]+)\s*:\s*\$?-?\d+(?:\.\d+)?")

    out = []
    seen = set()

    for line in reply.splitlines():
        for m in pattern.finditer(line):
            raw = m.group(1).strip()
            key = _norm_name(raw)

            if not raw:
                continue
            if key in _EXCLUDE_CLAIMS:
                continue
            if _MONTH_RE.fullmatch(raw):
                continue

            if raw not in seen:
                out.append(raw)
                seen.add(raw)

    return out


def _breakdown_map(df: pd.DataFrame) -> dict[str, float]:
    # Convenience map for quick membership checks and later numeric comparisons if needed.
    if df is None or df.empty:
        return {}
    out = {}
    for _, r in df.iterrows():
        out[str(r["event_name"]).strip()] = float(r["amount"])
    return out


# Kept around in case we want to expand numeric checks (amount-level grounding) later.
_MONEY_RE = re.compile(r"\$?\s*(-?\d+(?:\.\d+)?)")


def _safe_float(x: str) -> float | None:
    # Small helper so parsing doesn't blow up the app on unexpected text.
    try:
        return float(str(x).strip())
    except Exception:
        return None


def _extract_all_month_mentions(text: str) -> list[str]:
    # Useful for catching responses that reference months the account doesn't even have.
    if not text:
        return []
    out = []
    seen = set()
    for m in _MONTH_RE.findall(text):
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def _extract_total_claims(reply: str, evidence_months: list[str]) -> list[dict]:
    """
    Best-effort extraction of total claims from the reply.
    Returns list of {bill_month, total} where bill_month may be inferred for single-month answers.
    """
    if not reply:
        return []

    claims = []

    # Compare-style: Total: 202510 165.02 -> 202511 115.50 (Δ ...)
    # Note: supports both "→" and "->" since terminals/fonts vary.
    cmp_pat = re.compile(
        r"Total\s*:\s*(20\d{4})\s*\$?\s*(-?\d+(?:\.\d+)?)\s*.*?(?:→|->)\s*(20\d{4})\s*\$?\s*(-?\d+(?:\.\d+)?)",
        re.IGNORECASE | re.DOTALL,
    )
    m = cmp_pat.search(reply)
    if m:
        m1, t1, m2, t2 = m.group(1), m.group(2), m.group(3), m.group(4)
        f1, f2 = _safe_float(t1), _safe_float(t2)
        if f1 is not None:
            claims.append({"bill_month": m1, "total": f1})
        if f2 is not None:
            claims.append({"bill_month": m2, "total": f2})
        return claims

    # Line-by-line: "... 202511 ... Total: 115.50"
    line_pat = re.compile(r"(20\d{4}).*?\bTotal\b\s*:\s*\$?\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
    for line in reply.splitlines():
        mm = line_pat.search(line)
        if mm:
            bm = mm.group(1)
            tv = _safe_float(mm.group(2))
            if tv is not None:
                claims.append({"bill_month": bm, "total": tv})

    if claims:
        return claims

    # Single-month: "Total charges: $115.50" (infer month from evidence if unambiguous)
    single_pat = re.compile(r"Total\s+charges\s*:\s*\$?\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
    ms = single_pat.search(reply)
    if ms and len(evidence_months) == 1:
        tv = _safe_float(ms.group(1))
        if tv is not None:
            claims.append({"bill_month": evidence_months[0], "total": tv})

    return claims


def _extract_delta_claim(reply: str) -> float | None:
    if not reply:
        return None
    # Handles "(Δ $-49.52)" and "(Δ -49.52)".
    pat = re.compile(r"\(\s*Δ\s*\$?\s*(-?\d+(?:\.\d+)?)\s*\)", re.IGNORECASE)
    m = pat.search(reply)
    if not m:
        return None
    return _safe_float(m.group(1))


# ---------------------------
# Session state
# ---------------------------
def init_state():
    # Seed the chat with a helpful assistant message.
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome to **Orange Mobile**.\n\n"
                    "Try:\n"
                    "- Do I have federal tax this month?\n"
                    "- What about last month?\n"
                    "- Break down my bill\n"
                    "- Compare my last 2 bills (full comparison)\n"
                ),
            }
        ]

    # Default demo account values (can be changed in sidebar).
    if "mtn" not in st.session_state:
        st.session_state.mtn = "+12145550101"
    if "cust_id" not in st.session_state:
        st.session_state.cust_id = "CUST2001"

    # Controls whether the sample accounts panel is visible.
    if "show_accounts" not in st.session_state:
        st.session_state.show_accounts = False

    # Conversation memory used by core.answer() for follow-ups.
    if "last_topic" not in st.session_state:
        st.session_state.last_topic = None
    if "last_mode" not in st.session_state:
        st.session_state.last_mode = None

    # Audit/verification context (last turn only).
    # We only keep the latest turn to avoid bloating session state.
    if "last_question" not in st.session_state:
        st.session_state.last_question = None
    if "last_reply" not in st.session_state:
        st.session_state.last_reply = None
    if "last_evidence_months" not in st.session_state:
        st.session_state.last_evidence_months = []


def clear_chat_and_memory():
    # Reset conversation and follow-up context to initial state.
    st.session_state.messages = []
    st.session_state.last_topic = None
    st.session_state.last_mode = None
    st.session_state.last_question = None
    st.session_state.last_reply = None
    st.session_state.last_evidence_months = []
    init_state()


init_state()

st.set_page_config(
    page_title="Orange Mobile | Bill Inquiry Bot",
    page_icon="🟧",
    layout="wide",
)

# ---------------------------
# CSS (enterprise portal style)
# ---------------------------
# This is just scoped CSS to make the demo look like a telco portal.
# Streamlit class names can change across versions, so keep it non-fragile.
st.markdown(
    """
<style>
.stApp { background: #f8fafc; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; }

.om-header {
  background: #0b0b0c;
  border: 1px solid rgba(255,255,255,0.08);
  padding: 18px 18px;         /* extra height so header doesn't get clipped */
  border-radius: 14px;
  color: #ffffff;
  box-shadow: 0 10px 28px rgba(0,0,0,0.20);
  margin-bottom: 18px;        /* push content below header a bit */
}
.om-brand { font-weight: 800; font-size: 18px; letter-spacing: 0.2px; }
.om-sub { font-size: 12px; opacity: 0.85; margin-top: 6px; }  /* more breathing room */
.om-hi { text-align:right; font-weight: 800; font-size: 14px; }
.om-meta { text-align:right; font-size: 12px; opacity: 0.85; margin-top: 6px; }

.om-card {
  background: #ffffff;
  border: 1px solid rgba(15, 23, 42, 0.10);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
}

/* KPI tiles used in Billing Overview */
.om-kpi {
  border: 1px solid rgba(15, 23, 42, 0.10);
  border-radius: 14px;
  padding: 14px 16px;
  background: #ffffff;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  min-height: 84px;
}
.om-kpi-label { font-size: 12px; opacity: 0.75; }
.om-kpi-month { font-size: 18px; font-weight: 800; margin-top: 2px; }
.om-kpi-amt { font-size: 14px; font-weight: 700; opacity: 0.85; }
.om-kpi-sub { font-size: 12px; opacity: 0.75; margin-top: 2px; }

h1 { font-size: 26px !important; }
h2 { font-size: 18px !important; }
h3 { font-size: 16px !important; }
p, li { font-size: 14px !important; }

[data-testid="stDataFrame"] * { font-size: 13px !important; }

section[data-testid="stSidebar"] {
  background: #ffffff;
  border-right: 1px solid rgba(15, 23, 42, 0.10);
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Header
# ---------------------------
hl, hr = st.columns([2, 1], vertical_alignment="center")

with hl:
    st.markdown(
        """
<div class="om-header">
  <div class="om-brand">🟧 Orange Mobile  |  Bill Inquiry Bot</div>
  <div class="om-sub">Offline demo • SQLite + FAISS + deterministic billing logic</div>
</div>
""",
        unsafe_allow_html=True,
    )

with hr:
    cust_display = _norm_cust(st.session_state.cust_id)
    mtn_display = st.session_state.mtn.strip()
    st.markdown(
        f"""
<div class="om-header">
  <div class="om-hi">Hi, {cust_display}</div>
  <div class="om-meta">MTN: {mtn_display}</div>
</div>
""",
        unsafe_allow_html=True,
    )

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Account")
    st.caption("Use these fields like a telecom portal login context.")

    # Input fields for the account context used by all tabs.
    mtn = st.text_input("MTN", value=st.session_state.mtn)
    cust_id = st.text_input("Customer ID", value=st.session_state.cust_id)
    st.session_state.mtn = mtn
    st.session_state.cust_id = cust_id

    st.divider()

    if st.button("🧹 Clear chat"):
        clear_chat_and_memory()
        st.rerun()

    if st.button("👥 Recruiter Playground (Sample accounts)"):
        st.session_state.show_accounts = not st.session_state.show_accounts

    if st.session_state.show_accounts:
        st.caption("Pick an account to try:")
        df_acct = load_sample_accounts(limit=25)
        if df_acct.empty:
            st.info("No accounts found in billing.db.")
        else:
            st.dataframe(
                df_acct[["mtn", "cust_id", "months_count", "latest_month"]],
                width="stretch",
                height=220,
                hide_index=True,
            )

            st.caption("Click to auto-fill:")
            for i, row in df_acct.iterrows():
                label = f"{row['cust_id']} | MTN={row['mtn']} | latest={row['latest_month']}"
                if st.button(label, key=f"use_{i}"):
                    # Switching accounts should reset follow-up context to avoid topic leakage.
                    st.session_state.mtn = str(row["mtn"])
                    st.session_state.cust_id = str(row["cust_id"])
                    st.session_state.last_topic = None
                    st.session_state.last_mode = None
                    st.session_state.last_question = None
                    st.session_state.last_reply = None
                    st.session_state.last_evidence_months = []
                    st.success("Account updated.")
                    st.rerun()

# ---------------------------
# Main tabs
# ---------------------------
tabs = st.tabs(["💬 Chat", "📌 Billing Overview", "📊 Compare", "✅ Verify"])

# ---- Chat (conversation on top, input at bottom)
with tabs[0]:
    st.markdown('<div class="om-card">', unsafe_allow_html=True)
    st.subheader("Conversation")

    chat_box = st.container(height=520)
    with chat_box:
        for msg in st.session_state.messages:
            avatar = "👤" if msg["role"] == "user" else "🟧"
            with st.chat_message(msg["role"], avatar=avatar):
                st.write(msg["content"])

    user_text = st.chat_input("Message Orange Mobile…")

    if user_text:
        # Persist the user message first, then compute assistant reply.
        st.session_state.messages.append({"role": "user", "content": user_text})

        bot_reply, new_topic, new_mode = answer(
            user_text,
            mtn=st.session_state.mtn,
            cust_id=st.session_state.cust_id,
            last_topic=st.session_state.last_topic,
            last_mode=st.session_state.last_mode,
        )
        st.session_state.last_topic = new_topic
        st.session_state.last_mode = new_mode

        # Save last turn artifacts for audit/verification.
        # We only store the latest Q/A because recruiters usually click around quickly.
        mtn0 = st.session_state.mtn
        cust0 = st.session_state.cust_id
        months_desc = account_months(mtn0, cust0)
        st.session_state.last_question = user_text
        st.session_state.last_reply = bot_reply
        st.session_state.last_evidence_months = _infer_evidence_months(new_mode, user_text, bot_reply, months_desc)

        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ---- Billing overview
with tabs[1]:
    st.markdown('<div class="om-card">', unsafe_allow_html=True)
    st.subheader("Billing Overview")

    mtn0 = st.session_state.mtn
    cust0 = st.session_state.cust_id

    months = account_months(mtn0, cust0)
    if not months:
        st.info("No billing data found for this account.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        latest = months[0]
        prev = months[1] if len(months) > 1 else None

        total_latest = month_total(mtn0, cust0, latest)
        total_prev = month_total(mtn0, cust0, prev) if prev else None

        # Overview tiles: month label + amount in the same tile (no metric arrows).
        k1, k2, k3 = st.columns(3)

        latest_amt = f"${total_latest:.2f}" if total_latest is not None else "—"
        prev_amt = f"${total_prev:.2f}" if (prev and total_prev is not None) else "—"

        with k1:
            st.markdown(
                f"""
<div class="om-kpi">
  <div>
    <div class="om-kpi-label">Current month</div>
    <div class="om-kpi-month">{latest}</div>
  </div>
  <div class="om-kpi-amt">{latest_amt}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        with k2:
            st.markdown(
                f"""
<div class="om-kpi">
  <div>
    <div class="om-kpi-label">Previous month</div>
    <div class="om-kpi-month">{prev or "—"}</div>
  </div>
  <div class="om-kpi-amt">{prev_amt}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        with k3:
            # Keep change visible, but avoid Streamlit metric delta styling/arrows.
            if prev and total_prev is not None and total_latest is not None:
                delta = total_latest - total_prev
                delta_txt = f"${delta:+.2f}"
            else:
                delta_txt = "—"

            st.markdown(
                f"""
<div class="om-kpi">
  <div>
    <div class="om-kpi-label">Change vs prev</div>
    <div class="om-kpi-month">{delta_txt}</div>
    <div class="om-kpi-sub">Latest vs previous bill</div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.divider()

        df_bd = month_breakdown(mtn0, cust0, latest)
        if df_bd.empty:
            st.info("No line items found for current month.")
        else:
            st.write("**Top drivers (current month)**")
            st.dataframe(df_bd.head(8), width="stretch", height=260, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ---- Compare (Enhanced: N months, totals trend, top changing items, only changed toggle)
with tabs[2]:
    st.markdown('<div class="om-card">', unsafe_allow_html=True)
    st.subheader("Compare Bills (N months)")

    mtn0 = st.session_state.mtn
    cust0 = st.session_state.cust_id
    months = account_months(mtn0, cust0)

    if len(months) < 2:
        st.info("Need at least 2 months of data to compare.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Bound the window so the UI stays responsive even if accounts have long history.
        max_n = min(len(months), 12)

        if max_n <= 2:
            # Avoid Streamlit slider min/max collision when only 2 months exist.
            n = 2
            st.info("Only 2 months available for this account — comparing 2 months.")
        else:
            n = st.slider("Months to compare", min_value=2, max_value=max_n, value=min(3, max_n), step=1)

        # Default to the latest N months; user can override via multiselect.
        default_sel = months[:n]
        default_sel_chrono = list(reversed(default_sel))  # oldest -> newest for readability

        months_sel = st.multiselect(
            "Select months (optional override)",
            options=list(reversed(months)),  # oldest -> newest in dropdown
            default=default_sel_chrono,
        )

        # Normalize selected months to chronological order (YYYYMM sorts correctly).
        months_sel = [m for m in months_sel if m in months]
        months_sel = sorted(months_sel)

        if len(months_sel) < 2:
            st.warning("Select at least 2 months to compare.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # --- Totals trend ---
            df_tot = n_month_totals(mtn0, cust0, months_sel)
            df_tot = df_tot.sort_values("bill_month").reset_index(drop=True)

            st.write("**Totals trend**")
            st.dataframe(df_tot, width="stretch", height=200, hide_index=True)

            # Quick trend visualization; uses Streamlit defaults.
            st.line_chart(df_tot.set_index("bill_month")["total"])

            st.divider()

            # --- Top changing line items across the window ---
            df_long = n_month_breakdowns(mtn0, cust0, months_sel)

            if df_long.empty:
                st.info("No line items found for the selected months.")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                pivot = (
                    df_long.pivot_table(
                        index="event_name",
                        columns="bill_month",
                        values="amount",
                        aggfunc="sum",
                        fill_value=0.0,
                    )
                    .reset_index()
                )

                month_cols = [m for m in months_sel if m in pivot.columns]
                if len(month_cols) >= 2:
                    pivot["max_amount"] = pivot[month_cols].max(axis=1)
                    pivot["min_amount"] = pivot[month_cols].min(axis=1)
                    pivot["range"] = pivot["max_amount"] - pivot["min_amount"]

                    # Display table includes per-month amounts plus a simple variability signal ("range").
                    df_change = pivot[["event_name"] + month_cols + ["range"]].copy()

                    only_changed = st.toggle("Only changed items (hide rows with range = 0)", value=True)
                    if only_changed:
                        df_change = df_change[df_change["range"] != 0.0]

                    df_change = df_change.sort_values("range", ascending=False).reset_index(drop=True)

                    top_k = st.slider("Top changing items to show", min_value=5, max_value=50, value=15, step=5)

                    st.write("**Top changing line items (by range across selected months)**")
                    st.dataframe(df_change.head(top_k), width="stretch", height=520, hide_index=True)

                    st.caption("Range = max(month_amount) - min(month_amount) within the selected window.")
                else:
                    st.info("Need at least 2 months selected to compute changes.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- Verify (grounding + hallucination checks)
with tabs[3]:
    st.markdown('<div class="om-card">', unsafe_allow_html=True)
    st.subheader("Verify Chatbot Response")

    mtn0 = st.session_state.mtn
    cust0 = st.session_state.cust_id
    months_desc = account_months(mtn0, cust0)

    last_q = st.session_state.last_question
    last_r = st.session_state.last_reply

    if not last_q or not last_r:
        st.info("Ask something in the Chat tab first. This view will show the supporting bill evidence.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        evidence_months = st.session_state.last_evidence_months or _infer_evidence_months(
            st.session_state.last_mode, last_q, last_r, months_desc
        )

        # Evidence header
        mode_txt = st.session_state.last_mode or "current"
        st.write(f"**Detected mode:** `{mode_txt}`")
        st.write(f"**Evidence month(s):** {', '.join(evidence_months) if evidence_months else '—'}")

        with st.expander("Last question + response", expanded=False):
            st.write("**User**")
            st.write(last_q)
            st.write("**Assistant**")
            st.write(last_r)

        st.divider()

        # Load evidence breakdowns for the month(s) we're validating against.
        evidence_frames = []
        breakdowns = {}
        for m in evidence_months:
            df_m = month_breakdown(mtn0, cust0, m)
            breakdowns[m] = _breakdown_map(df_m)
            if not df_m.empty:
                df_show = df_m.copy()
                df_show.insert(0, "bill_month", m)
                evidence_frames.append(df_show[["bill_month", "event_name", "amount"]])

        if not evidence_frames:
            st.warning("No supporting line-item data found for the inferred evidence month(s).")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            df_evidence = pd.concat(evidence_frames, ignore_index=True)
            st.write("**Bill evidence (from SQLite)**")
            st.dataframe(df_evidence, width="stretch", height=420, hide_index=True)

            st.divider()

            # Pull out "line items" the bot claimed and verify they're actually in the evidence months.
            claimed = _extract_claimed_items(last_r)

            all_events_raw = account_event_names(mtn0, cust0)
            all_events_norm = {_norm_name(x) for x in all_events_raw}

            allowed_raw = set()
            for m in evidence_months:
                allowed_raw |= set(breakdowns.get(m, {}).keys())
            allowed_norm = {_norm_name(x) for x in allowed_raw}

            rows = []
            for name in claimed:
                name_norm = _norm_name(name)
                status = "OK"
                if name_norm not in allowed_norm:
                    status = "Not in evidence months"
                    if name_norm not in all_events_norm:
                        status = "Not in account data"
                rows.append({"claimed_item": name, "status": status})

            df_claims = pd.DataFrame(rows)

            # Extra verification signals that are easy to explain in interviews:
            # - Month validity: did the bot reference months that aren't even available?
            # - Total grounding: did it quote the right total for the month?
            # - Delta grounding: for compare mode, is the Δ correct?
            mentioned_months = _extract_all_month_mentions(last_r)
            invalid_months = [m for m in mentioned_months if m not in set(months_desc)]
            invalid_month_rate = (len(invalid_months) / max(1, len(mentioned_months))) * 100.0

            total_claims = _extract_total_claims(last_r, evidence_months)
            total_checks = []
            for tc in total_claims:
                bm = tc["bill_month"]
                ev_total = month_total(mtn0, cust0, bm)
                if ev_total is None:
                    total_checks.append(
                        {
                            "bill_month": bm,
                            "claimed_total": tc["total"],
                            "evidence_total": None,
                            "abs_error": None,
                            "status": "No evidence total",
                        }
                    )
                else:
                    err = abs(float(tc["total"]) - float(ev_total))
                    ok = err <= 0.01  # cents-level tolerance is fine for demo data
                    total_checks.append(
                        {
                            "bill_month": bm,
                            "claimed_total": float(tc["total"]),
                            "evidence_total": float(ev_total),
                            "abs_error": float(err),
                            "status": "OK" if ok else "Mismatch",
                        }
                    )

            delta_claim = _extract_delta_claim(last_r)
            delta_check = None
            if mode_txt == "compare" and len(evidence_months) >= 2:
                old_m, new_m = evidence_months[0], evidence_months[1]  # [prev, latest]
                t_old = month_total(mtn0, cust0, old_m)
                t_new = month_total(mtn0, cust0, new_m)
                if t_old is not None and t_new is not None:
                    expected_delta = float(t_new) - float(t_old)
                    if delta_claim is None:
                        delta_check = {
                            "claimed_delta": None,
                            "expected_delta": expected_delta,
                            "abs_error": None,
                            "status": "No delta in reply",
                        }
                    else:
                        derr = abs(float(delta_claim) - expected_delta)
                        delta_check = {
                            "claimed_delta": float(delta_claim),
                            "expected_delta": expected_delta,
                            "abs_error": float(derr),
                            "status": "OK" if derr <= 0.01 else "Mismatch",
                        }

            if df_claims.empty:
                st.info("No line-item claims detected in the last response (nothing to validate).")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                # Hallucination rate here is: "claimed items not found in evidence months".
                # This is a pragmatic metric, not a formal NLP evaluation.
                bad = df_claims[df_claims["status"] != "OK"]
                hallu_rate = (len(bad) / max(1, len(df_claims))) * 100.0
                grounded_precision = 100.0 - hallu_rate

                # Quick KPI row so the verifier isn't just a wall of tables.
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Grounded precision", f"{grounded_precision:.1f}%")
                m2.metric("Hallucination rate", f"{hallu_rate:.1f}%")
                m3.metric("Invalid month rate", f"{invalid_month_rate:.1f}%")
                m4.metric("Claimed items", f"{len(df_claims)}")

                if len(bad) == 0:
                    st.success(f"Grounding check passed. Hallucination rate: {hallu_rate:.1f}%")
                else:
                    st.warning(f"Grounding check found issues. Hallucination rate: {hallu_rate:.1f}%")

                if invalid_months:
                    st.caption(f"Months mentioned but not present for this account: {', '.join(invalid_months)}")

                st.write("**Claims extracted from response**")
                st.dataframe(df_claims, width="stretch", height=260, hide_index=True)

                if len(bad) > 0:
                    st.caption(
                        "Items marked 'Not in evidence months' were mentioned by the bot but are not present in the selected bill breakdown."
                    )

                # Total grounding results (if we found totals to validate)
                if total_checks:
                    st.divider()
                    st.write("**Total grounding (reply vs SQLite)**")
                    st.dataframe(pd.DataFrame(total_checks), width="stretch", height=220, hide_index=True)

                # Delta grounding for compare mode
                if delta_check:
                    st.divider()
                    st.write("**Delta grounding (compare mode)**")
                    st.dataframe(pd.DataFrame([delta_check]), width="stretch", height=120, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)
