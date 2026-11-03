"""Shared data-input helpers: upload a CSV or paste cells / a link from Google Sheets."""
import re

import pandas as pd
import streamlit as st

import core as core
import rounds


@st.cache_data(show_spinner=False)
def load_csv(data: bytes) -> pd.DataFrame:
    return core.read_csv_bytes(data)


@st.cache_data(show_spinner="Reading pasted data…", ttl=300)
def load_pasted(text: str) -> pd.DataFrame:
    t = text.strip()
    if t.startswith("http") and "docs.google.com/spreadsheets" in t and "\n" not in t:
        return core.read_csv_bytes(rounds.fetch_gsheet(t))  # a sheet link was pasted
    return core.read_pasted_text(text)


def data_input(label: str, key: str):
    """Upload a CSV, or paste cells / a sheet link from Google Sheets. Returns a DataFrame or None."""
    st.markdown(f"**{label}**")
    src = st.radio(label, ["Upload CSV", "Paste from Google Sheets"], horizontal=True,
                   key=f"{key}_src", label_visibility="collapsed")
    if src == "Upload CSV":
        up = st.file_uploader(label, type="csv", key=key, label_visibility="collapsed")
        return load_csv(up.getvalue()) if up is not None else None

    text = st.text_area(
        label, key=f"{key}_paste", height=160, label_visibility="collapsed",
        placeholder="In Google Sheets, select the cells INCLUDING the header row → Ctrl+C → paste here.\n"
                    "(You can also paste the sheet link if it's shared as 'Anyone with the link → Viewer'.)",
    )
    if not text.strip():
        return None
    try:
        df = load_pasted(text)
    except Exception as e:  # noqa: BLE001
        st.error(f"Couldn't read that: {e}")
        return None
    st.caption(f"✓ {len(df):,} rows · {len(df.columns)} columns read")
    if any(re.fullmatch(r"\+?\d[\d\s-]{8,}", str(c)) for c in df.columns):
        st.warning("The first row looks like data, not headers — copy the header row too.")
    return df
