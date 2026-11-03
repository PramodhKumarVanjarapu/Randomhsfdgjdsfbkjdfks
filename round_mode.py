"""Mode 2 — Round funnel from a company's tracking sheet (Google Sheet URL or CSV)."""
import re

import altair as alt
import pandas as pd
import streamlit as st

import core
import rounds
from inputs import data_input, load_pasted
import pool_view

NONE = "— none —"
CAT_ORDER = ["Intensive", "Academy", "External"]


@st.cache_data(show_spinner="Fetching Google Sheet…", ttl=300)
def _fetch(url: str) -> bytes:
    return rounds.fetch_gsheet(url)


def _pick(label, df, default, key):
    options = [NONE] + list(df.columns)
    idx = options.index(default) if default in options else 0
    v = st.selectbox(label, options, index=idx, key=key)
    return None if v == NONE else v


def _label(c):
    return str(c).replace("\n", " ").strip()


def render(pools, tr_sum, tr_rule, academy_kw, intensive_kw):
    st.title("🪜 Company Round Funnel")
    company = st.text_input("Company name", placeholder="e.g. Travel Unbounded", key="rf_company")

    # ---------------- source ----------------
    src = st.radio(
        "Company sheet source",
        ["Google Sheet URL", "Paste from Google Sheets", "Upload CSV"],
        horizontal=True, key="rf_src",
    )
    sheet = None
    if src == "Google Sheet URL":
        u1, u2 = st.columns([5, 1])
        url = u1.text_input("Google Sheet link (open the right tab first so the link has its gid)", key="rf_url")
        if u2.button("↻ Refresh", help="Re-fetch the sheet (cached for 5 min)"):
            _fetch.clear()
        if url:
            try:
                sheet = core.read_csv_bytes(_fetch(url))
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
                st.stop()
    elif src == "Paste from Google Sheets":
        text = st.text_area(
            "Company sheet cells", key="rf_paste", height=180, label_visibility="collapsed",
            placeholder="In Google Sheets, select the cells INCLUDING the header row → Ctrl+C → paste here.",
        )
        if text.strip():
            try:
                sheet = load_pasted(text)
            except Exception as e:  # noqa: BLE001
                st.error(f"Couldn't read that: {e}")
                st.stop()
            st.caption(f"✓ {len(sheet):,} rows · {len(sheet.columns)} columns read")
            if any(re.fullmatch(r"\+?\d[\d\s-]{8,}", str(c)) for c in sheet.columns):
                st.warning("The first row looks like data, not headers — copy the header row too.")
    else:
        up = st.file_uploader("Company sheet CSV", type="csv", key="rf_csv")
        sheet = core.read_csv_bytes(up.getvalue()) if up else None

    apps = data_input("Optional · Applications with Product (if the sheet has no Product column)", "rf_apps")
    if sheet is None:
        st.info("Add the company's sheet: paste its link, paste the cells, or upload a CSV.")
        st.stop()

    sheet = sheet.reset_index(drop=True)
    det_s = core.detect_columns(sheet)
    det_a = core.detect_columns(apps) if apps is not None else {}

    # ---------------- settings ----------------
    with st.expander("Columns & status words", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Candidate columns in the sheet**")
            s_cols = {
                "name": _pick("Name", sheet, det_s["name"], "rf_name"),
                "mobile": _pick("Mobile", sheet, det_s["mobile"], "rf_mobile"),
                "email": _pick("Email", sheet, det_s["email"], "rf_email"),
                "uid": _pick("User ID", sheet, det_s["uid"], "rf_uid"),
                "resume": _pick("Resume", sheet, det_s["resume"], "rf_resume"),
                "product": _pick("Product (if the sheet has one)", sheet, det_s["product"], "rf_product"),
            }
            if apps is not None:
                st.markdown("**Applications file**")
                a_cols = {
                    "product": _pick("Product", apps, det_a["product"], "rf_a_product"),
                    "mobile": _pick("Mobile", apps, det_a["mobile"], "rf_a_mobile"),
                    "name": _pick("Name", apps, det_a["name"], "rf_a_name"),
                    "uid": _pick("User ID", apps, det_a["uid"], "rf_a_uid"),
                    "email": _pick("Email", apps, det_a["email"], "rf_a_email"),
                }
        with c2:
            st.markdown("**Extra status words** (comma-separated, matched inside the cell text)")
            extra_pass = [w.strip().lower() for w in st.text_input("Count as Cleared", "", key="rf_pass").split(",") if w.strip()]
            extra_fail = [w.strip().lower() for w in st.text_input("Count as Rejected", "", key="rf_fail").split(",") if w.strip()]

    detected = rounds.detect_round_columns(sheet)
    round_cols = st.multiselect(
        "Rounds in funnel order (auto-detected — add, remove or re-order)",
        options=list(sheet.columns),
        default=detected,
        format_func=_label,
        key="rf_rounds",
    )
    if not round_cols:
        st.warning("No round columns found. Pick them in the box above.")
        st.stop()

    # ---------------- candidates + product ----------------
    join_s = s_cols["mobile"] or s_cols["email"]
    if s_cols["product"] or apps is None or not join_s:
        # Product from the sheet itself (or none -> everyone External)
        prepared = core.prepare_company(sheet, s_cols, None, {}, None, None, academy_kw, intensive_kw)
        prepared["Resume"] = sheet[s_cols["resume"]] if s_cols["resume"] else None
        product_note = "Product from the sheet." if s_cols["product"] else "No Product source — everyone counts as External. Upload the applications CSV to split by product."
    else:
        # Sheet is the base, Product looked up from the applications file by mobile
        join_a = a_cols["mobile"] if s_cols["mobile"] else a_cols["email"]
        prepared = core.prepare_company(
            apps, a_cols, sheet, {"mobile": s_cols["mobile"], "resume": s_cols["resume"]},
            join_a, join_s, academy_kw, intensive_kw,
        )
        if s_cols["email"]:  # keep the sheet's own email for bucket matching
            prepared["_email"] = prepared["_email"].combine_first(sheet[s_cols["email"]].map(core.norm_email).astype(object))
        found = int(prepared["_app_found"].sum())
        product_note = f"Product from the applications file — matched {found} of {len(prepared)} sheet rows; the rest count as External."
    matched = core.match_company(prepared, pools, tr_sum)

    stages_all = pd.DataFrame(
        {c: sheet[c].map(lambda v: rounds.classify(v, extra_pass, extra_fail)) for c in round_cols},
        index=sheet.index,
    )

    # One row per candidate: keep the row with the most round information
    filled = (stages_all != "").sum(axis=1)
    keep = filled.sort_values(ascending=False, kind="stable").index
    keep = matched.loc[keep].drop_duplicates("_person").index.sort_values()
    cand = matched.loc[keep].copy()
    stages = stages_all.loc[keep]
    raw_rounds = sheet.loc[keep, round_cols]
    total = len(cand)

    # ---------------- headline ----------------
    st.subheader(f"{company + ' — ' if company else ''}Summary")
    st.caption(product_note)
    n = lambda m: int(m.sum())  # noqa: E731
    final_col = round_cols[-1]
    r1 = st.columns(5)
    r1[0].metric("Total applications", f"{len(sheet):,}")
    r1[1].metric("Unique candidates", f"{total:,}")
    r1[2].metric("Intensive", n(cand["Category"] == "Intensive"))
    r1[3].metric("Academy", n(cand["Category"] == "Academy"))
    r1[4].metric("External", n(cand["Category"] == "External"))
    r2 = st.columns(6)
    r2[0].metric("Bucket students", n(cand["In Bucket"]))
    r2[1].metric("Assessment cleared", n(cand["In Assessment Pool"]))
    r2[2].metric("TR Cleared", n(cand["TR Cleared"]), help=f"Rule: {tr_rule}")
    r2[3].metric(f"Cleared {_label(round_cols[0])}", n(stages[round_cols[0]] == "Cleared"))
    r2[4].metric(f"Cleared {_label(final_col)}", n(stages[final_col] == "Cleared"))
    r2[5].metric("Still in process", n(stages.isin(["Pending"]).any(axis=1)))
    pool_view.render_pool_metrics(pools, tr_sum, cand)
    if len(sheet) != total:
        st.caption(f"{len(sheet) - total} duplicate rows merged (same mobile / email).")

    tab_f, tab_p, tab_pool, tab_c, tab_s, tab_q = st.tabs(
        ["Round funnel", "Funnel by product", "Pool coverage", "Candidates", "Status mapping", "Data checks"]
    )
    with tab_pool:
        pool_view.render_pool_tab(pools, tr_sum, tr_rule, cand, company)

    # ---------------- funnel ----------------
    funnel = rounds.funnel_table(stages, round_cols, total)
    with tab_f:
        chart_df = pd.concat(
            [pd.DataFrame({"Round": ["Applied"], "Cleared": [total]}), funnel[["Round", "Cleared"]]]
        )
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                y=alt.Y("Round:N", sort=None, title=None),
                x=alt.X("Cleared:Q", title="Candidates cleared"),
                tooltip=["Round", "Cleared"],
            )
            .properties(height=40 * len(chart_df) + 20)
        )
        labels = chart.mark_text(align="left", dx=4).encode(text="Cleared:Q")
        st.altair_chart(chart + labels, width="stretch")
        st.dataframe(funnel, hide_index=True, width="stretch")

    # ---------------- by product ----------------
    with tab_p:
        cats = [c for c in CAT_ORDER if c in set(cand["Category"])]
        extra = ["Bucket students", "Assessment cleared", "TR Cleared"]
        tbl = pd.DataFrame(0, index=["Applied"] + extra + [_label(c) for c in round_cols], columns=cats)
        for cat in cats:
            m = cand["Category"] == cat
            tbl.loc["Applied", cat] = n(m)
            tbl.loc["Bucket students", cat] = n(m & cand["In Bucket"])
            tbl.loc["Assessment cleared", cat] = n(m & cand["In Assessment Pool"])
            tbl.loc["TR Cleared", cat] = n(m & cand["TR Cleared"])
            for c in round_cols:
                tbl.loc[_label(c), cat] = n(m & (stages[c] == "Cleared"))
        tbl["Total"] = tbl.sum(axis=1)
        st.markdown("**By product: pool status, then each round cleared**")
        st.dataframe(tbl.astype(int), width="stretch")

        bucket_lbl = cand["Bucket"].fillna("Not in bucket")
        btbl = pd.DataFrame({"Applied": bucket_lbl.value_counts()})
        for c in round_cols:
            btbl[_label(c)] = bucket_lbl[stages[c] == "Cleared"].value_counts()
        order = [b for b in pools["bucket_values"] if b in btbl.index] + (["Not in bucket"] if "Not in bucket" in btbl.index else [])
        st.markdown("**Cleared each round, by bucket**")
        st.dataframe(btbl.reindex(order).fillna(0).astype(int), width="stretch")

        summary = pd.DataFrame(
            [{"Company": company, "Total applications": len(sheet), "Unique candidates": total,
              **{c: n(cand["Category"] == c) for c in CAT_ORDER},
              "Bucket students": n(cand["In Bucket"]), "Assessment cleared": n(cand["In Assessment Pool"]),
              "TR Cleared": n(cand["TR Cleared"]),
              "Total in buckets (pool)": len(pools["bucket"]), "Total assessment cleared (pool)": len(tr_sum),
              **{f"{_label(c)} cleared": n(stages[c] == "Cleared") for c in round_cols}}]
        )
        st.download_button(
            "⬇️ Download funnel summary CSV",
            summary.to_csv(index=False).encode("utf-8"),
            file_name=f"{(company or 'company').replace(' ', '_')}_funnel_summary.csv",
            mime="text/csv",
        )

    # ---------------- candidates ----------------
    with tab_c:
        view = pd.DataFrame(
            {
                "Name": cand["Name"], "Mobile": cand["Mobile"], "Category": cand["Category"],
                "Bucket": cand["Bucket"], "TR Cleared": cand["TR Cleared"],
                **{_label(c): raw_rounds[c] for c in round_cols},
                "Furthest cleared": rounds.furthest_cleared(stages, round_cols),
                "Latest outcome": rounds.current_outcome(stages, round_cols),
                "Resume": cand["Resume"],
            }
        )
        f1, f2, f3 = st.columns(3)
        sel_cat = f1.multiselect("Category", CAT_ORDER, default=CAT_ORDER, key="rf_fcat")
        cleared_opts = ["Any"] + [_label(c) for c in round_cols]
        sel_round = f2.selectbox("Cleared round", cleared_opts, key="rf_fround")
        search = f3.text_input("Search name / mobile", key="rf_search")
        v = view[view["Category"].isin(sel_cat)]
        if sel_round != "Any":
            col = round_cols[cleared_opts.index(sel_round) - 1]
            v = v[stages.loc[v.index, col] == "Cleared"]
        if search:
            s = search.strip().lower()
            v = v[v["Name"].fillna("").str.lower().str.contains(s, regex=False) | v["Mobile"].fillna("").str.contains(s, regex=False)]
        st.caption(f"{len(v):,} candidates")
        st.dataframe(v, hide_index=True, width="stretch",
                     column_config={"Resume": st.column_config.LinkColumn("Resume", display_text="Open")})
        st.download_button(
            "⬇️ Download these candidates (CSV)", v.to_csv(index=False).encode("utf-8"),
            file_name=f"{(company or 'company').replace(' ', '_')}_round_candidates.csv", mime="text/csv",
        )

    # ---------------- status mapping ----------------
    with tab_s:
        st.caption("Every value found in the round columns and how it was counted. Fix any wrong ones with the extra status words above.")
        rows = []
        for c in round_cols:
            vc = sheet[c].dropna().astype(str).str.strip()
            vc = vc[vc != ""].value_counts()
            for val, cnt in vc.items():
                rows.append({"Round": _label(c), "Value": val, "Counted as": rounds.classify(val, extra_pass, extra_fail) or "(blank)", "Rows": cnt})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        others = [r for r in rows if r["Counted as"] == "Other"]
        if others:
            st.warning(f"{len(others)} value(s) weren't recognised and are counted as 'Other'.")

    # ---------------- data checks ----------------
    with tab_q:
        ooo = rounds.out_of_order(stages, round_cols)
        skipped = [_label(c) for c in sheet.columns if c not in round_cols]
        st.table(pd.DataFrame({
            "Check": ["Candidates without a usable mobile", "Duplicate rows merged",
                      "Candidates with a later-round status while an earlier round isn't Cleared"],
            "Count": [n(cand["Mobile"].isna()), len(sheet) - total, n(ooo)],
        }))
        if n(ooo):
            st.markdown("**Out-of-order candidates** (usually a status that wasn't updated)")
            st.dataframe(view.loc[ooo[ooo].index], hide_index=True, width="stretch")
        st.caption("Columns not used as rounds: " + ", ".join(skipped))
