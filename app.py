"""
LedgerBridge AI — Streamlit Web App

User flow:
  1. Upload two ledger files
  2. AI proposes ROLE (buyer/seller) + column mappings → user reviews/confirms
  3. Enter opening balances
  4. Process & download reports

This file owns ALL UI. Business logic lives in the other modules; this file
just wires them together with Streamlit and a custom visual layer.
"""

import base64
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv

from config import MAPPABLE_FIELDS, ROLES, DEFAULT_AMOUNT_TOLERANCE
from ingest import load_ledger, list_sheets
from mapper import analyze_ledger, save_confirmed_analysis
from standardize import standardize
from reconcile import reconcile
from report import write_report, write_standardized_ledger
from insights import generate_insights
from tds_reconciliation import classify_tds_entries, apply_tds_reclassification
from cost_tracker import CostTracker

load_dotenv()

st.set_page_config(
    page_title="LedgerBridge AI",
    page_icon="assets/logo_mark_small.png" if Path("assets/logo_mark_small.png").exists() else "📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — premium light-mode visual layer
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #FAFAF7;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: #1A1A1A;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1280px;
    }
    #MainMenu, footer, header { visibility: hidden; }

    .lb-brand {
        display: flex; align-items: center; gap: 14px;
        padding-bottom: 8px; margin-bottom: 6px;
    }
    .lb-brand-mark { height: 44px; width: auto; }
    .lb-brand-text {
        font-size: 26px; font-weight: 700; color: #1F3A6B;
        letter-spacing: -0.5px; line-height: 1;
    }
    .lb-brand-text .ai-tag {
        color: #1E6DD8; font-weight: 600; font-size: 18px;
        margin-left: 6px; letter-spacing: 0;
    }
    .lb-tagline {
        color: #666; font-size: 13px; margin-top: -2px;
        margin-bottom: 16px; font-weight: 400; letter-spacing: 0.2px;
    }

    .lb-session-meter { display: none; }  /* deprecated — no longer used */

    .lb-stepper {
        display: flex; gap: 8px;
        margin: 8px 0 28px 0; padding: 0;
    }
    .lb-step {
        flex: 1; padding: 14px 18px; border-radius: 10px;
        font-size: 13px; font-weight: 500;
        display: flex; align-items: center; gap: 10px;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }
    .lb-step-done {
        background: #F0F7F2; color: #2D6A4F; border-color: #C8E0CF;
    }
    .lb-step-current {
        background: #FFFFFF; color: #1F3A6B; border-color: #1E6DD8;
        box-shadow: 0 2px 6px rgba(30, 109, 216, 0.12); font-weight: 600;
    }
    .lb-step-pending {
        background: #F4F4F1; color: #9CA3AF; border-color: #E5E7EB;
    }
    .lb-step-icon {
        width: 22px; height: 22px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 11px; font-weight: 700; flex-shrink: 0;
    }
    .lb-step-done .lb-step-icon    { background: #2D6A4F; color: white; }
    .lb-step-current .lb-step-icon { background: #1E6DD8; color: white; }
    .lb-step-pending .lb-step-icon { background: #D1D5DB; color: white; }

    h1, h2, h3, h4 {
        color: #1A1A1A; font-family: 'Inter', sans-serif !important;
        font-weight: 700; letter-spacing: -0.3px;
    }
    h1 { font-size: 28px !important; margin-bottom: 4px !important; }
    h2 { font-size: 22px !important; margin-top: 24px !important; margin-bottom: 8px !important; }
    h3 { font-size: 16px !important; color: #374151 !important; font-weight: 600 !important; }

    .stButton > button {
        border-radius: 8px; font-weight: 500; padding: 10px 22px;
        transition: all 0.15s ease; border: 1px solid #D1D5DB;
        background: white; color: #374151;
        font-family: 'Inter', sans-serif !important;
    }
    .stButton > button:hover { background: #F9FAFB; border-color: #9CA3AF; }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #1F3A6B 0%, #1E6DD8 100%);
        color: white; border: none; font-weight: 600;
        box-shadow: 0 2px 4px rgba(31, 58, 107, 0.18);
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #163059 0%, #1857B8 100%);
        box-shadow: 0 4px 8px rgba(31, 58, 107, 0.24);
        transform: translateY(-1px);
    }
    .stDownloadButton > button {
        border-radius: 8px; font-weight: 500; padding: 12px 22px;
        font-family: 'Inter', sans-serif !important;
    }

    [data-testid="stFileUploader"] {
        background: #FFFFFF; border: 1px dashed #C7D2DD;
        border-radius: 10px; padding: 8px;
    }
    [data-testid="stFileUploader"]:hover { border-color: #1E6DD8; background: #FAFCFF; }

    .stSelectbox > div > div, .stNumberInput > div > div > input {
        border-radius: 8px !important; border-color: #E5E7EB !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stSelectbox > div > div:hover, .stNumberInput > div > div > input:hover {
        border-color: #C7D2DD !important;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E8E6E0;
        border-radius: 10px; padding: 18px 20px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        transition: all 0.15s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: #C7D2DD; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
    }
    [data-testid="stMetricLabel"] {
        font-size: 12px !important; font-weight: 500 !important;
        color: #6B7280 !important; text-transform: uppercase; letter-spacing: 0.4px;
    }
    [data-testid="stMetricValue"] {
        font-size: 26px !important; font-weight: 700 !important;
        color: #1F3A6B !important;
        font-family: 'Inter', sans-serif !important;
    }

    .lb-banner {
        padding: 18px 24px; border-radius: 12px; margin: 18px 0;
        display: flex; align-items: center; gap: 16px;
        font-size: 15px; font-weight: 500;
        border-left: 4px solid;
    }
    .lb-banner-icon { font-size: 24px; flex-shrink: 0; }
    .lb-banner-content { flex: 1; }
    .lb-banner-title { font-weight: 700; font-size: 16px; margin-bottom: 2px; }
    .lb-banner-sub { font-size: 13px; font-weight: 400; opacity: 0.85; }
    .lb-banner-success { background: #F0F7F2; color: #1F4E3A; border-color: #2D6A4F; }
    .lb-banner-warn    { background: #FEF7E6; color: #7C5E12; border-color: #B7791F; }
    .lb-banner-error   { background: #FCEDED; color: #6B1F1F; border-color: #9C2C2C; }
    .lb-banner-info    { background: #EEF4FB; color: #1F3A6B; border-color: #1E6DD8; }

    [data-testid="stExpander"] {
        border: 1px solid #E8E6E0 !important;
        border-radius: 10px !important;
        background: #FFFFFF !important;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    [data-testid="stExpander"] summary {
        font-weight: 600 !important; color: #1F3A6B !important;
        padding: 14px 18px !important; font-size: 14px !important;
    }
    [data-testid="stExpander"] summary:hover { background: #FAFCFF; }

    .stDataFrame {
        border: 1px solid #E8E6E0; border-radius: 8px; overflow: hidden;
    }

    .stCaption, [data-testid="stCaptionContainer"] { color: #6B7280; font-size: 12.5px; }

    .stRadio > div { gap: 8px; flex-direction: row !important; }
    .stRadio label { font-weight: 500 !important; }

    .lb-spacer-sm { margin-top: 12px; margin-bottom: 12px; }
    .lb-spacer-md { margin-top: 22px; margin-bottom: 22px; }

    .lb-conf {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 3px 10px; border-radius: 999px;
        font-size: 11px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .lb-conf-high   { background: #F0F7F2; color: #2D6A4F; }
    .lb-conf-medium { background: #FEF7E6; color: #7C5E12; }
    .lb-conf-low    { background: #FCEDED; color: #6B1F1F; }
    .lb-conf-na     { background: #F3F4F6; color: #6B7280; }

    hr { border: none; border-top: 1px solid #E8E6E0; margin: 24px 0; }

    .lb-ledger-title {
        font-size: 17px; font-weight: 700; color: #1F3A6B;
        margin-bottom: 4px;
        display: flex; align-items: center; gap: 8px;
    }
    .lb-ledger-meta {
        font-size: 12px; color: #6B7280;
        margin-bottom: 14px; padding-bottom: 12px;
        border-bottom: 1px solid #F0EFEB;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# UI render helpers
# ═══════════════════════════════════════════════════════════════════════════

def _logo_b64(path: str) -> str:
    if not Path(path).exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def render_header():
    logo_b64 = _logo_b64("assets/logo_mark_small.png")
    logo_html = (
        f'<img src="data:image/png;base64,{logo_b64}" class="lb-brand-mark" alt="LedgerBridge AI"/>'
        if logo_b64 else ""
    )
    st.markdown(
        f"""
        <div class="lb-brand">
            {logo_html}
            <div>
                <div class="lb-brand-text">LedgerBridge<span class="ai-tag">AI</span></div>
                <div class="lb-tagline">AI-powered ledger reconciliation</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stepper(current: int):
    labels = ["Upload", "Confirm Mapping", "Opening Balances", "Results"]
    parts = []
    for i, label in enumerate(labels, start=1):
        if i < current:
            cls, icon = "lb-step-done", "✓"
        elif i == current:
            cls, icon = "lb-step-current", str(i)
        else:
            cls, icon = "lb-step-pending", str(i)
        parts.append(
            f'<div class="lb-step {cls}">'
            f'<span class="lb-step-icon">{icon}</span>'
            f'<span>{label}</span>'
            f'</div>'
        )
    st.markdown(f'<div class="lb-stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


def banner(kind: str, title: str, sub: str = "", icon: str = ""):
    default_icons = {"success": "✓", "warn": "⚠", "error": "✗", "info": "ℹ"}
    icon = icon or default_icons.get(kind, "")
    sub_html = f'<div class="lb-banner-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="lb-banner lb-banner-{kind}">
            <div class="lb-banner-icon">{icon}</div>
            <div class="lb-banner-content">
                <div class="lb-banner-title">{title}</div>
                {sub_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def conf_badge(conf: str) -> str:
    conf = (conf or "n/a").lower()
    if conf == "high":   return '<span class="lb-conf lb-conf-high">● High</span>'
    if conf == "medium": return '<span class="lb-conf lb-conf-medium">● Medium</span>'
    if conf == "low":    return '<span class="lb-conf lb-conf-low">● Low</span>'
    return '<span class="lb-conf lb-conf-na">○ N/A</span>'


# ═══════════════════════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "step": 1,
        "our_df": None, "their_df": None,
        "our_analysis": None, "their_analysis": None,
        "result": None, "ai_insights": "",
        "tds_result": None,
        "cost_tracker": None,
        "report_paths": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ═══════════════════════════════════════════════════════════════════════════
# Generic helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_client() -> Anthropic | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        banner("error", "API key missing",
               "Set ANTHROPIC_API_KEY in your .env file before running.")
        return None
    return Anthropic(api_key=key)


def save_uploaded_file(uploaded) -> str:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        return tmp.name


def reset():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_state()
    st.rerun()


def _safe_write(write_fn, default_path):
    try:
        return write_fn(default_path)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_path = default_path.with_name(f"{default_path.stem}_{ts}{default_path.suffix}")
        return write_fn(new_path)


# ═══════════════════════════════════════════════════════════════════════════
# Page chrome
# ═══════════════════════════════════════════════════════════════════════════

render_header()
render_stepper(st.session_state.step)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Upload
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.step == 1:
    st.markdown("## Step 1 — Upload Ledger Files")
    st.markdown(
        '<p style="color:#6B7280; margin-top:-4px; margin-bottom:24px;">'
        'Upload both ledger files. CSV and Excel (.xlsx, .xls) are supported.'
        '</p>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="lb-ledger-title">📂 Our Ledger</div>', unsafe_allow_html=True)
        st.markdown('<div class="lb-ledger-meta">Your books</div>', unsafe_allow_html=True)
        our_file = st.file_uploader("Drop your CSV / XLSX / XLS here", type=["csv", "xlsx", "xls"], key="our_upload", label_visibility="collapsed")
    with c2:
        st.markdown('<div class="lb-ledger-title">📁 Their Ledger</div>', unsafe_allow_html=True)
        st.markdown('<div class="lb-ledger-meta">Counterparty\'s books</div>', unsafe_allow_html=True)
        their_file = st.file_uploader("Drop your CSV / XLSX / XLS here", type=["csv", "xlsx", "xls"], key="their_upload", label_visibility="collapsed")

    if our_file and their_file:
        # Prevent Out-Of-Memory (OOM) crashes by enforcing a 25MB limit
        MAX_MB = 25
        if (our_file.size > MAX_MB * 1024 * 1024) or (their_file.size > MAX_MB * 1024 * 1024):
            banner("error", "File too large", f"One or both files exceed the {MAX_MB}MB limit. Please split them to prevent memory crashes.")
            st.stop()

        st.markdown('<div class="lb-spacer-md"></div>', unsafe_allow_html=True)
        if st.button("Analyze →", type="primary", use_container_width=True):
            with st.spinner("Reading files and asking Claude to detect role + map columns..."):
                our_path = save_uploaded_file(our_file)
                their_path = save_uploaded_file(their_file)

                our_sheets = list_sheets(our_path)
                their_sheets = list_sheets(their_path)
                our_sheet = our_sheets[0] if our_sheets else None
                their_sheet = their_sheets[0] if their_sheets else None

                st.session_state.our_df = load_ledger(our_path, our_sheet)
                st.session_state.their_df = load_ledger(their_path, their_sheet)

                client = get_client()
                if client is None:
                    st.stop()

                tracker = CostTracker()
                st.session_state.cost_tracker = tracker

                st.session_state.our_analysis = analyze_ledger(
                    st.session_state.our_df, client,
                    cost_tracker=tracker, step_name="Map Our Ledger",
                )
                st.session_state.their_analysis = analyze_ledger(
                    st.session_state.their_df, client,
                    cost_tracker=tracker, step_name="Map Their Ledger",
                )

                from standardize import detect_role_from_data
                for analysis_key, df_key in [("our_analysis", "our_df"), ("their_analysis", "their_df")]:
                    a = st.session_state[analysis_key]
                    if a.get("role_confidence") in ("low", "n/a", None):
                        role, reason = detect_role_from_data(
                            st.session_state[df_key], a.get("mapping", {})
                        )
                        a["role"] = role
                        a["role_confidence"] = "medium"
                        a["role_reasoning"] = f"rule-based: {reason}"
                        st.session_state[analysis_key] = a

                st.session_state.step = 2
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Confirm role + mappings
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    st.markdown("## Step 2 — Confirm Role & Column Mappings")
    st.markdown(
        '<p style="color:#6B7280; margin-top:-4px; margin-bottom:8px;">'
        '<strong>Role</strong> determines how Gross Amount is computed from Debit/Credit. '
        '<strong>Mappings</strong> tell the engine which column means what. '
        'This is the critical accuracy step — review carefully before confirming.'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="lb-spacer-sm"></div>', unsafe_allow_html=True)

    def render_editor(df, analysis, label):
        st.markdown(f'<div class="lb-ledger-title">{label}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="lb-ledger-meta">{len(df)} rows · {len(df.columns)} columns: '
            f'{", ".join(list(df.columns)[:6])}{"..." if len(df.columns) > 6 else ""}</div>',
            unsafe_allow_html=True,
        )

        current_role = analysis.get("role", "buyer")
        role_conf = analysis.get("role_confidence", "n/a")
        role_reason = analysis.get("role_reasoning", "")

        st.markdown(
            f'<div style="margin-bottom:6px;">'
            f'<strong style="color:#1F3A6B;">Role</strong> &nbsp; {conf_badge(role_conf)}'
            f'</div>'
            f'<div style="font-size:12px; color:#6B7280; margin-bottom:8px; font-style:italic;">{role_reason}</div>',
            unsafe_allow_html=True,
        )
        new_role = st.radio(
            f"{label} role",
            ROLES,
            index=ROLES.index(current_role) if current_role in ROLES else 0,
            horizontal=True,
            key=f"{label}_role",
            label_visibility="collapsed",
        )

        st.markdown('<hr style="margin:18px 0;">', unsafe_allow_html=True)
        st.markdown('<div style="font-weight:600; color:#1F3A6B; margin-bottom:10px;">Column mappings</div>', unsafe_allow_html=True)

        mapping = analysis.get("mapping", {})
        options = ["(not present)"] + list(df.columns)
        new_mapping = {}

        for field in MAPPABLE_FIELDS:
            entry = mapping.get(field, {})
            src = entry.get("source")
            conf = entry.get("confidence", "n/a")

            idx = options.index(src) if (src in df.columns) else 0

            c1, c2, c3 = st.columns([2, 3, 1.3])
            with c1:
                st.markdown(f'<div style="padding-top:8px; font-weight:500; color:#374151;">{field}</div>', unsafe_allow_html=True)
            with c2:
                choice = st.selectbox(
                    f"Source for {field}",
                    options,
                    index=idx,
                    key=f"{label}_{field}",
                    label_visibility="collapsed",
                )
            with c3:
                st.markdown(f'<div style="padding-top:8px;">{conf_badge(conf)}</div>', unsafe_allow_html=True)

            new_mapping[field] = (
                {"source": None, "confidence": "n/a"}
                if choice == "(not present)"
                else {"source": choice, "confidence": conf}
            )

        return {
            "role": new_role,
            "role_confidence": role_conf,
            "role_reasoning": role_reason,
            "mapping": new_mapping,
        }

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.session_state.our_analysis = render_editor(
            st.session_state.our_df, st.session_state.our_analysis, "Our Ledger"
        )
    with c2:
        st.session_state.their_analysis = render_editor(
            st.session_state.their_df, st.session_state.their_analysis, "Their Ledger"
        )

    st.markdown('<div class="lb-spacer-md"></div>', unsafe_allow_html=True)
    nav = st.columns([1, 3, 1])
    with nav[0]:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with nav[2]:
        if st.button("Confirm →", type="primary", use_container_width=True):
            save_confirmed_analysis(st.session_state.our_df, st.session_state.our_analysis)
            save_confirmed_analysis(st.session_state.their_df, st.session_state.their_analysis)
            st.session_state.step = 3
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Opening balances & settings
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    st.markdown("## Step 3 — Opening Balances & Settings")
    st.markdown(
        '<p style="color:#6B7280; margin-top:-4px; margin-bottom:24px;">'
        'Optional but recommended. Opening balances enable the closing-balance walk and the RECONCILED banner.'
        '</p>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        opening_ours = st.number_input("Opening balance (Our books)", value=0.0, step=1000.0, format="%.2f")
    with c2:
        opening_theirs = st.number_input("Opening balance (Their books)", value=0.0, step=1000.0, format="%.2f")

    st.markdown('<div class="lb-spacer-sm"></div>', unsafe_allow_html=True)

    tolerance = st.number_input(
        "Amount tolerance (₹)",
        value=DEFAULT_AMOUNT_TOLERANCE,
        step=0.5,
        help="Amounts within this difference are treated as matched (handles rounding).",
    )
    enable_ai = st.checkbox(
        "Generate AI insights for the Summary sheet",
        value=True,
        help="One additional Claude API call. Adds plain-English explanation of exceptions.",
    )

    st.markdown('<div class="lb-spacer-md"></div>', unsafe_allow_html=True)
    nav = st.columns([1, 3, 1])
    with nav[0]:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with nav[2]:
        if st.button("Run Reconciliation →", type="primary", use_container_width=True):
            with st.spinner("Standardizing, reconciling, generating report..."):
                our_a = st.session_state.our_analysis
                their_a = st.session_state.their_analysis

                ours_std = standardize(st.session_state.our_df, our_a["mapping"], role=our_a["role"])
                theirs_std = standardize(st.session_state.their_df, their_a["mapping"], role=their_a["role"])

                result = reconcile(
                    ours_std, theirs_std,
                    amount_tolerance=tolerance,
                    opening_balance_ours=opening_ours,
                    opening_balance_theirs=opening_theirs,
                )

                tds_result = classify_tds_entries(
                    missing_in_theirs=result.missing_in_theirs,
                    missing_in_ours=result.missing_in_ours,
                    our_full_ledger=result.our_ledger,
                    their_full_ledger=result.their_ledger,
                    tolerance=tolerance,
                )
                our_final, their_final = apply_tds_reclassification(
                    result.our_ledger, result.their_ledger, tds_result
                )

                ai_text = ""
                if enable_ai:
                    client = get_client()
                    if client:
                        ai_text = generate_insights(
                            result, client,
                            cost_tracker=st.session_state.cost_tracker,
                        )

                out_dir = Path("outputs")
                out_dir.mkdir(exist_ok=True)
                rec = _safe_write(
                    lambda p: write_report(
                        result, p, ai_insights=ai_text,
                        tds_result=tds_result,
                        cost_tracker=st.session_state.cost_tracker,
                    ),
                    out_dir / "reconciliation_report.xlsx",
                )
                ours_xlsx = _safe_write(
                    lambda p: write_standardized_ledger(our_final, p, "Our Standardized Ledger"),
                    out_dir / "standardized_our_books.xlsx",
                )
                theirs_xlsx = _safe_write(
                    lambda p: write_standardized_ledger(their_final, p, "Their Standardized Ledger"),
                    out_dir / "standardized_their_books.xlsx",
                )

                st.session_state.result = result
                st.session_state.tds_result = tds_result
                st.session_state.ai_insights = ai_text
                st.session_state.report_paths = {"report": rec, "ours": ours_xlsx, "theirs": theirs_xlsx}
                st.session_state.step = 4
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Results dashboard
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    st.markdown("## Step 4 — Results")
    result = st.session_state.result
    s = result.summary

    or_ = st.session_state.our_analysis
    tr_ = st.session_state.their_analysis
    st.markdown(
        f'<p style="color:#6B7280; margin-top:-4px; margin-bottom:18px; font-size:13px;">'
        f'Roles used: Our ledger = <strong style="color:#1F3A6B;">{or_["role"]}</strong> · '
        f'Their ledger = <strong style="color:#1F3A6B;">{tr_["role"]}</strong>'
        f'</p>',
        unsafe_allow_html=True,
    )

    # Adjust the Missing counts to reflect TDS-reclassified rows — keeps the
    # KPI tiles consistent with both the Excel report's Summary sheet and the
    # individual exception tables below.
    tds_result_pre = st.session_state.get("tds_result")
    tds_removed_theirs = len(tds_result_pre.removed_from_missing_theirs) if tds_result_pre else 0
    tds_removed_ours   = len(tds_result_pre.removed_from_missing_ours)   if tds_result_pre else 0
    adj_missing_theirs = s["missing_in_theirs"] - tds_removed_theirs
    adj_missing_ours   = s["missing_in_ours"]   - tds_removed_ours

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("L1 Matches", s["matched_l1"], help="Date + Ref + Amount all align")
    c2.metric("L2 Matches (Timing)", s["matched_l2"])
    c3.metric("Amount Mismatches", s["amount_mismatches"])
    c4.metric("Missing in Theirs", adj_missing_theirs,
              help="In our books but not in theirs (TDS journal entries excluded — see TDS Reconciliation)")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("L3 Matches (Review)", s["matched_l3"])
    c6.metric("Missing in Ours", adj_missing_ours,
              help="In their books but not in ours (TDS journal entries excluded — see TDS Reconciliation)")
    c7.metric("Our Records", s["total_our_records"])
    c8.metric("Their Records", s["total_their_records"])

    st.markdown('<div class="lb-spacer-sm"></div>', unsafe_allow_html=True)

    if s["reconciled"]:
        banner("success",
               "RECONCILED",
               f"Residual ₹{s['residual']:,.2f} is within tolerance. Books match.")
    else:
        banner("error",
               "NOT RECONCILED",
               f"Residual: ₹{s['residual']:,.2f}. See Amount Mismatches sheet for details.")

    with st.expander("📊 Closing Balance Walk", expanded=True):
        walk = pd.DataFrame([
            ("Opening Balance",     s["opening_balance_ours"], s["opening_balance_theirs"]),
            ("Sum of transactions", s["sum_our_transactions"], s["sum_their_transactions"]),
            ("Closing Balance",     s["closing_balance_ours"], s["closing_balance_theirs"]),
        ], columns=["", "Ours (₹)", "Theirs (₹)"])
        st.dataframe(walk.style.format({"Ours (₹)": "{:,.2f}", "Theirs (₹)": "{:,.2f}"}), use_container_width=True, hide_index=True)

        cb1, cb2, cb3 = st.columns(3)
        cb1.metric("Difference (Ours − Theirs)", f"₹{s['difference']:,.2f}")
        cb2.metric("Reconciling items", f"₹{s['reconciling_item']:,.2f}")
        cb3.metric("Residual", f"₹{s['residual']:,.2f}")

    tds_result = st.session_state.get("tds_result")
    if tds_result is not None and tds_result.overall_status != "NO_TDS_ACTIVITY":
        with st.expander("🧾 TDS Reconciliation", expanded=True):
            status = tds_result.overall_status
            msg = tds_result.status_message
            if status == "MATCHED":
                banner("success", "TDS Matched", msg, icon="✓")
            elif status == "PARTIAL":
                banner("warn", "Partial TDS Posting", msg, icon="⚠")
            elif status == "EXCESS":
                banner("error", "Excess TDS Journal", msg, icon="✗")
            else:
                banner("info", "TDS Unverified", msg, icon="ℹ")

            tds_totals = pd.DataFrame([
                ("TDS column total (sum of TDS Amount)",
                 tds_result.our_tds_column_total, tds_result.their_tds_column_total),
                ("TDS journal entries (description-flagged)",
                 tds_result.our_tds_journal_total, tds_result.their_tds_journal_total),
            ], columns=["", "Ours (₹)", "Theirs (₹)"])
            st.dataframe(
                tds_totals.style.format({"Ours (₹)": "{:,.2f}", "Theirs (₹)": "{:,.2f}"}),
                use_container_width=True, hide_index=True,
            )

            if not tds_result.flagged_entries.empty:
                st.markdown('<div style="font-weight:600; margin-top:12px; margin-bottom:6px; color:#1F3A6B;">Individual TDS journal entries detected:</div>', unsafe_allow_html=True)
                st.dataframe(tds_result.flagged_entries, use_container_width=True, hide_index=True)

    if st.session_state.ai_insights:
        with st.expander("🤖 AI Insights", expanded=True):
            st.write(st.session_state.ai_insights)

    if not result.amount_mismatches.empty:
        with st.expander(f"⚠ Amount Mismatches ({len(result.amount_mismatches)})"):
            st.dataframe(result.amount_mismatches, use_container_width=True, hide_index=True)

    miss_theirs = result.missing_in_theirs
    miss_ours = result.missing_in_ours
    if tds_result is not None:
        if tds_result.removed_from_missing_theirs:
            miss_theirs = miss_theirs.drop(index=tds_result.removed_from_missing_theirs, errors="ignore").reset_index(drop=True)
        if tds_result.removed_from_missing_ours:
            miss_ours = miss_ours.drop(index=tds_result.removed_from_missing_ours, errors="ignore").reset_index(drop=True)

    if not miss_theirs.empty:
        with st.expander(f"➡ Missing in Their Books ({len(miss_theirs)})"):
            st.dataframe(miss_theirs, use_container_width=True, hide_index=True)

    if not miss_ours.empty:
        with st.expander(f"⬅ Missing in Our Books ({len(miss_ours)})"):
            st.dataframe(miss_ours, use_container_width=True, hide_index=True)

    if not result.timing_differences.empty:
        with st.expander(f"⏱ Timing Differences ({len(result.timing_differences)})"):
            st.dataframe(result.timing_differences, use_container_width=True, hide_index=True)

    st.markdown('<div class="lb-spacer-md"></div>', unsafe_allow_html=True)
    st.markdown('### Download Reports')
    paths = st.session_state.report_paths
    d1, d2, d3 = st.columns(3, gap="medium")
    with d1:
        with open(paths["report"], "rb") as f:
            st.download_button(
                "📊 Reconciliation Report",
                data=f,
                file_name="reconciliation_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary",
            )
    with d2:
        with open(paths["ours"], "rb") as f:
            st.download_button(
                "📁 Standardized Our Books",
                data=f,
                file_name="standardized_our_books.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    with d3:
        with open(paths["theirs"], "rb") as f:
            st.download_button(
                "📁 Standardized Their Books",
                data=f,
                file_name="standardized_their_books.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    st.markdown('<div class="lb-spacer-md"></div>', unsafe_allow_html=True)

    # ─── Run details (token usage + API cost) ─────────────────────────────
    tracker = st.session_state.get("cost_tracker")
    if tracker is not None and tracker.records:
        with st.expander("🧾 Run Details — token usage and API cost", expanded=False):
            totals = tracker.total()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total cost", f"₹{totals['cost_inr']:.2f}", help=f"≈ ${totals['cost_usd']:.4f} USD")
            c2.metric("Input tokens", f"{totals['input_tokens']:,}")
            c3.metric("Output tokens", f"{totals['output_tokens']:,}")
            c4.metric("API calls", totals["calls"], help=f"{totals['cache_hits']} skipped (cached)")

            st.markdown('<div style="font-weight:600; margin-top:14px; margin-bottom:6px; color:#1F3A6B;">Per-step breakdown:</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(tracker.summary_rows()), use_container_width=True, hide_index=True)

            if totals["cache_hits"] > 0:
                banner("info",
                       f"{totals['cache_hits']} step(s) used cached mappings",
                       "Future reconciliations of the same file formats reuse the cache automatically — no API charge.",
                       icon="ℹ")

    st.markdown('<div class="lb-spacer-md"></div>', unsafe_allow_html=True)
    if st.button("🔄 Start a new reconciliation", use_container_width=False):
        reset()
