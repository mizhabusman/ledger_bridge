"""
LedgerBridge AI — Ingest Module

Reads CSV / XLSX / XLS files. Handles:
- Encoding detection
- Header row detection (skipping junk preamble rows)
- Multi-sheet workbooks
- Ragged CSV rows (preamble with fewer columns than data rows)
- Tally multi-row XLS exports (collapses blocks, extracts hidden invoice refs)
- .xls files that are actually .xlsx (detected by magic bytes)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# Magic bytes that identify file types regardless of extension.
XLSX_MAGIC = b"PK\x03\x04"
XLS_MAGIC  = b"\xD0\xCF\x11\xE0"

# Extracts "Our Inv No : 25-26/GS-140" from Tally sub-row narration cells
_INV_RE = re.compile(r'Our Inv No\s*:\s*([\w/\-]+)', re.IGNORECASE)

# Detect leading ISO dates (YYYY-MM-DD / YYYY/MM/DD). Must match standardize.clean_date
# so main-row detection here uses the SAME dayfirst convention as canonical parsing.
_ISO_DATE_RE = re.compile(r"^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}")

# Words that identify summary rows to skip entirely
_SKIP_WORDS = ['Opening Balance', 'Grand Total', 'Closing Balance', 'Account Closed']


# ─────────────────────────── file-type detection ────────────────────────────

def detect_file_type(file_path: str | Path) -> str:
    path = Path(file_path)
    with open(path, "rb") as f:
        head = f.read(8)
    if head.startswith(XLSX_MAGIC):
        return "xlsx"
    if head.startswith(XLS_MAGIC):
        return "xls"
    return "csv"


def list_sheets(file_path: str | Path) -> list[str]:
    ftype = detect_file_type(file_path)
    if ftype == "csv":
        return []
    engine = "xlrd" if ftype == "xls" else "openpyxl"
    try:
        return pd.ExcelFile(file_path, engine=engine).sheet_names
    except Exception:
        return pd.ExcelFile(file_path, engine="openpyxl").sheet_names


# ─────────────────────────── raw reading ────────────────────────────────────

def _read_raw(file_path: str | Path, sheet_name: str | None = None) -> pd.DataFrame:
    ftype = detect_file_type(file_path)

    if ftype == "csv":
        last_error = None
        for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    max_fields = max((len(line.split(",")) for line in f), default=1)
                col_names = [f"col_{i}" for i in range(max_fields)]
                return pd.read_csv(
                    file_path,
                    header=None,
                    names=col_names,
                    dtype=str,
                    encoding=enc,
                    keep_default_na=False,
                    engine="python",
                    on_bad_lines="skip",
                )
            except UnicodeDecodeError as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                continue
        raise ValueError(f"Could not decode CSV: {file_path}. Last error: {last_error}")

    if ftype == "xls":
        try:
            return pd.read_excel(
                file_path, sheet_name=sheet_name or 0,
                header=None, dtype=str, keep_default_na=False, engine="xlrd",
            )
        except Exception:
            return pd.read_excel(
                file_path, sheet_name=sheet_name or 0,
                header=None, dtype=str, keep_default_na=False, engine="openpyxl",
            )

    return pd.read_excel(
        file_path, sheet_name=sheet_name or 0,
        header=None, dtype=str, keep_default_na=False, engine="openpyxl",
    )


# ─────────────────────────── Tally multi-row detection & parsing ─────────────

def _is_tally_multirow(raw: pd.DataFrame) -> bool:
    """
    Detect Tally multi-row format by checking for a column that contains
    mostly 'r' or 'a' values (Tally's reconciliation marker).
    """
    for col in raw.columns:
        vals = raw[col].astype(str).str.strip()
        non_empty = vals[vals != ""]
        if len(non_empty) > 3:
            ra_count = non_empty.isin(["r", "a"]).sum()
            if ra_count / len(non_empty) > 0.5:
                return True
    return False


def _parse_tally_multirow(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a Tally multi-row export into one flat row per transaction.

    Tally XLS structure (columns by position after preamble):
      col 0 = Date           (only on main rows)
      col 1 = Vch No         (only on main rows)
      col 2 = Source/blank
      col 3 = Particulars    (description — only on main rows)
      col 4 = (blank on main rows, used for sub-row amounts)
      col 5 = Debit          (net amount received — invoices)
      col 6 = Credit         (gross amount paid out — payments/CNs)
      col 7 = New Ref / r/a flag
      sub-rows: col3 contains "Our Inv No : 25-26/GS-140..." or tax lines

    Approach: a main row has a date in col0 AND a voucher number in col1.
    Everything until the next dated row is a sub-row belonging to that block.
    """
    # Find the header row so we know actual column count and positions
    header_idx = _find_tally_header(raw)

    # Read actual header labels
    headers = [str(raw.iloc[header_idx, c]).strip() for c in range(raw.shape[1])]

    # Identify key column positions by header label
    col_date   = _find_col_idx(headers, ["date"])
    col_vch    = _find_col_idx(headers, ["vch", "voucher no"])
    col_partic = _find_col_idx(headers, ["particular", "narration", "description"])
    col_debit  = _find_col_idx(headers, ["debit"])
    col_credit = _find_col_idx(headers, ["credit"])

    # Defaults if not found
    if col_date   is None: col_date   = 0
    if col_vch    is None: col_vch    = 1
    if col_partic is None: col_partic = 3
    if col_debit  is None: col_debit  = 5
    if col_credit is None: col_credit = 6

    records = []
    current = None

    for i in range(header_idx + 1, len(raw)):
        row = raw.iloc[i]

        def cell(c):
            try:
                v = str(row.iloc[c]).strip()
                return "" if v.lower() in ("nan", "none") else v
            except Exception:
                return ""

        date_val = cell(col_date)
        vch_val  = cell(col_vch)
        partic   = cell(col_partic)
        debit    = cell(col_debit)
        credit   = cell(col_credit)

        # Skip summary rows
        if any(s.lower() in partic.lower() for s in _SKIP_WORDS):
            continue
        if any(s.lower() in date_val.lower() for s in _SKIP_WORDS):
            continue

        # Main transaction row: has both date and voucher number
        if _is_valid_date(date_val) and vch_val:
            if current:
                records.append(current)

            # Determine amount and sign:
            # Debit = they recorded money coming IN (invoice raised on buyer)
            # Credit = they recorded money going OUT (payment received, CN issued)
            # From reconciliation perspective we want a SIGNED gross:
            #   Invoice (SV) = positive (buyer owes them)
            #   Credit Note (CN) = negative (reduces what buyer owes)
            #   Bank Receipt (BR) = negative (money received, reduces balance)
            #   Journal (JV) = depends on which column has value
            gross_raw = debit if debit else credit
            # Determine sign: if amount came from Credit column, it's money
            # flowing back (CN or payment), treat as negative
            amount_is_credit = (not debit) and bool(credit)

            current = {
                "Date":        date_val,
                "Vch No":      vch_val,
                "Particulars": partic,
                "Debit":       debit,
                "Credit":      credit,
                "Gross Raw":   gross_raw,
                "Is Credit":   amount_is_credit,
                "Invoice Ref": "",
            }
        elif current is not None:
            # Sub-row: look for "Our Inv No :" pattern in every cell
            for c_idx in range(raw.shape[1]):
                text = str(row.iloc[c_idx])
                m = _INV_RE.search(text)
                if m:
                    ref = m.group(1)
                    # Strip _x000D_ encoding artifact and anything after
                    ref = re.sub(r'_x000D_.*', '', ref).strip()
                    if ref:
                        current["Invoice Ref"] = ref
                    break

    if current:
        records.append(current)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Build a clean output with the columns our standardize.py expects
    # We expose both Debit and Credit separately so the mapping screen
    # can show them, plus a pre-computed "Gross Amount" column that handles sign
    return df[["Date", "Vch No", "Particulars", "Debit", "Credit", "Invoice Ref"]]


def _find_tally_header(raw: pd.DataFrame) -> int:
    """Find the row that contains 'Date' and 'Vch No' — the real header."""
    for i in range(min(15, len(raw))):
        row_vals = [str(v).strip().lower() for v in raw.iloc[i]]
        if "date" in row_vals and any("vch" in v or "voucher" in v for v in row_vals):
            return i
    # Fallback to generic header detection
    return detect_header_row(raw)


def _find_col_idx(headers: list[str], keywords: list[str]) -> int | None:
    """Find column index by matching header label against keywords."""
    for i, h in enumerate(headers):
        hl = h.lower().strip()
        if any(k in hl for k in keywords):
            return i
    return None


def _is_valid_date(s: str) -> bool:
    if not s or s.lower() in ("nan", ""):
        return False
    # ISO dates auto-detect; everything else is dayfirst (Indian DD-MM-YYYY),
    # matching standardize.clean_date so both stages agree on ambiguous dates.
    dayfirst = not bool(_ISO_DATE_RE.match(s))
    try:
        pd.to_datetime(s, dayfirst=dayfirst)
        return True
    except Exception:
        return False


# ─────────────────────────── header detection (flat files) ──────────────────

def detect_header_row(raw: pd.DataFrame, max_check: int = 20) -> int:
    best_row, best_score = 0, -1
    for i in range(min(max_check, len(raw))):
        row = raw.iloc[i]
        non_empty = sum(1 for v in row if str(v).strip())
        if non_empty < 2:
            continue
        text_like = sum(
            1 for v in row
            if str(v).strip()
            and not _looks_numeric(v)
            and len(str(v).strip()) <= 40
        )
        score = text_like * 2 + non_empty
        if score > best_score:
            best_score, best_row = score, i
    return best_row


def _looks_numeric(v) -> bool:
    s = str(v).strip().replace(",", "").replace("₹", "").replace("(", "-").replace(")", "")
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


# ─────────────────────────── main entry point ───────────────────────────────

def load_ledger(file_path: str | Path, sheet_name: str | None = None) -> pd.DataFrame:
    """
    Load a ledger file and return a clean flat DataFrame.
    Handles Tally multi-row exports by collapsing them automatically.
    All values returned as strings — cleaning happens in standardize.py.
    """
    raw = _read_raw(file_path, sheet_name=sheet_name)

    # Tally multi-row detection and parsing
    if _is_tally_multirow(raw):
        return _parse_tally_multirow(raw)

    # Standard flat file: detect header row, slice, clean
    header_idx = detect_header_row(raw)
    headers = [str(v).strip() for v in raw.iloc[header_idx]]
    df = raw.iloc[header_idx + 1:].copy()
    df.columns = headers

    # Drop columns with no header
    df = df.loc[:, [c for c in df.columns if c]]

    # Drop fully-empty rows
    df = df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)].reset_index(drop=True)

    return df


def get_column_fingerprint(df: pd.DataFrame) -> str:
    cols = sorted(str(c).strip().lower() for c in df.columns)
    return "|".join(cols)