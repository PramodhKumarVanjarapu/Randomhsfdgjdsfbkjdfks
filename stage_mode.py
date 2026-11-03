"""Mode 4 — Student stages / guidance pool.

Input: a CSV with Job ID, Org ID, Order, Company Name, Company Sheet (link).
Reads every company sheet, maps each company's round columns onto standard stages
(Screening -> TR1 -> TR2+ -> HR / MR -> Final / Offer), and gives student-level stats so you
can filter e.g. "cleared at least TR1 in any company" into a guidance pool.
"""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st

import core
import rounds
from inputs import data_input
from multi_company import fetch_all, link_gid, pool_masks, sheet_id

NONE = "— none —"
STAGES = ["Screening", "TR1", "TR2+", "HR / MR", "Final / Offer"]
IGNORE = "(ignore)"
_S_IDX = {s: i for i, s in enumerate(STAGES)}
_ST_RANK = {"Cleared": 0, "Rejected": 1, "Pending": 2, "No-show": 3, "Dropped": 4}
_HIRED_RX = re.compile(r"hired|\boffer|joined|onboard")

# Checked in order; first match wins
_STAGE_RULES = [
    ("Final / Offer", r"\bfinal\b|\boffer|hired|\bjoin|selection status|\bresult\b|onboard|\bbgv\b|verification"),
    ("TR2+", r"\b(tr|technical|tech|l|interview)\s*(round|interview)?\s*[-_]?\s*[2-9]\b|\btr\s*[2-9]\b|"
             r"\bl[2-9]\b|\bround\s*[2-9]\b"),
    ("Screening", r"screen|shortlist|assessment|assignment|\btest\b|aptitude|online|\boa\b|hackathon|\bgd\b|"
                  r"group discussion|written|mcq|case study|telephonic|communication|introductory|ai interview|"
                  r"registration|attend|invite|\btask\b|coding (test|assessment|challenge)"),
    ("HR / MR", r"\bhr\b|\bmr\b|manager|founder|\bceo\b|\bcto\b|\bcoo\b|director|culture|cultural|fitment|"
                r"salary|\bvp\b|behaviou?ral|values"),
    ("TR1", r"\btr\b|\btr\s*1\b|technical|\btech\b|\bl1\b|interview|coding|machine test|pair programming|"
            r"system design|\bdsa\b|panel|f2f|face to face|client|customer|\bround\s*1\b|practical|project round|"
            r"presentation"),
]
_STAGE_RULES = [(s, re.compile(p)) for s, p in _STAGE_RULES]


def guess_stage(col) -> str:
    n = rounds.norm_name(col)
    for stage, rx in _STAGE_RULES:
        if rx.search(n):
            return stage
    return IGNORE


def _pick(label, df, default, key):
    options = [NONE] + list(df.columns)
    v = st.selectbox(label, options, index=options.index(default) if default in options else 0, key=key)
    return None if v == NONE else v


# --------------------------------------------------------------------------- #
# Per-tab: one row per candidate row with the best status per stage
# --------------------------------------------------------------------------- #
def tab_rows(df, colmap: dict, pools, tr_sum, academy_kw, intensive_kw):
    """colmap: {column: stage}. Returns rows with Mobile/pool info + one column per stage."""
    df = df.reset_index(drop=True)
    det = core.detect_columns(df)
    cols = {k: det[k] for k in ("name", "mobile", "email", "uid", "product")}
    prepared = core.prepare_company(df, cols, None, {}, None, None, academy_kw, intensive_kw)
    has_id = prepared[["_uid", "_mob", "_email"]].notna().any(axis=1)
    m = core.match_company(prepared, pools, tr_sum)

    out = pd.DataFrame({
        "_person": m["_person"], "Name": m["Name"], "Mobile": m["Mobile"], "Email": m["_email"],
        "In Bucket": m["In Bucket"], "Bucket": m["Bucket"], "Cycle": m["Cycle"],
        "In Assessment Pool": m["In Assessment Pool"], "TR Cleared": m["TR Cleared"],
        "TR1 Status": m["TR1 Status"], "TR2 Status": m["TR2 Status"],
    })
    used = [c for c, s in colmap.items() if s in _S_IDX]
    for s in STAGES:
        out[s] = ""
    hired = pd.Series(False, index=df.index)
    for c in used:
        stg = df[c].map(rounds.classify)
        raw = df[c].fillna("").astype(str).str.strip().str.lower()
        hired |= (stg == "Cleared") & raw.str.contains(_HIRED_RX)
        s = colmap[c]
        better = stg.map(lambda v: _ST_RANK.get(v, 9)) < out[s].map(lambda v: _ST_RANK.get(v, 9))
        out.loc[better, s] = stg[better]
    out.loc[hired, "Final / Offer"] = "Cleared"
    out["_used"] = [frozenset(_S_IDX[colmap[c]] for c in used) | (frozenset([4]) if h else frozenset())
                    for h in hired]
    out["Rounds"] = df[used].apply(
        lambda r: " | ".join(f"{str(c).strip()}: {v}" for c, v in r.items() if str(v).strip() not in ("", "nan")),
        axis=1) if used else ""
    return out[has_id.values]


def summarise(r: pd.DataFrame) -> pd.DataFrame:
    """Adds Reached / Highest cleared / Outcome columns to per student x company rows."""
    st_idx = lambda row, keys: [i for i, s in enumerate(STAGES) if row[s] in keys]  # noqa: E731
    reached, cleared, outcome, rej_at = [], [], [], []
    for _, row in r.iterrows():
        any_ = st_idx(row, _ST_RANK.keys())
        clr = st_idx(row, ("Cleared",))
        rj = st_idx(row, ("Rejected",))
        top = max(any_) if any_ else -1
        prev = [i for i in row["_used"] if i < top]  # reaching a stage means the company's previous stage was cleared
        hc = max((max(clr) if clr else -1), max(prev) if prev else -1)
        reached.append(STAGES[top] if top >= 0 else "")
        cleared.append(hc)
        if row["Final / Offer"] == "Cleared":
            outcome.append("Hired / Offer"); rej_at.append("")
        elif rj:
            outcome.append("Rejected"); rej_at.append(STAGES[max(rj)])
        elif top >= 0 and row[STAGES[top]] == "Pending":
            outcome.append("In process"); rej_at.append("")
        elif top >= 0 and row[STAGES[top]] in ("No-show", "Dropped"):
            outcome.append("No-show / Dropped"); rej_at.append("")
        elif top >= 0:
            outcome.append("In process"); rej_at.append("")
        else:
            outcome.append("Not started"); rej_at.append("")
    r = r.copy()
    r["Reached"] = reached
    r["_hc"] = cleared
    r["Highest cleared"] = [STAGES[i] if i >= 0 else "" for i in cleared]
    r["Outcome"] = outcome
    r["Rejected at"] = rej_at
    return r


def student_view(r: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """One row per student across companies. threshold = stage index for 'cleared at least' (-1 = any)."""
    r = r.copy()
    r["_sid"] = r["Mobile"].where(r["Mobile"].fillna("") != "", "p:" + r["_person"].astype(str))
    first = lambda s: next((v for v in s if isinstance(v, str) and v), "")  # noqa: E731
    rows = []
    for sid, d in r.groupby("_sid", sort=False):
        top = int(d["_hc"].max())
        where = d.loc[d["_hc"] == top, "Company"] if top >= 0 else pd.Series(dtype=object)
        rej = d[d["Outcome"] == "Rejected"]
        hired = d[d["Outcome"] == "Hired / Offer"]
        stuck = rej["Rejected at"].value_counts()
        if len(hired):
            flag = "Hired"
        elif len(stuck):
            s, n = stuck.index[0], int(stuck.iloc[0])
            flag = (f"Cleared {STAGES[top]}, " if top >= 0 else "") + f"rejected at {s}" + (f" ({n}×)" if n > 1 else "")
        elif (d["Outcome"] == "In process").any():
            flag = "In process at " + first(d.loc[d["Outcome"] == "In process", "Reached"])
        else:
            flag = first(d["Outcome"])
        rows.append({
            "Name": first(d["Name"]), "Mobile": first(d["Mobile"]), "Email": first(d["Email"]),
            "Bucket": first(d["Bucket"]), "Cycle": first(d["Cycle"]),
            "In Bucket": bool(d["In Bucket"].any()), "In Assessment Pool": bool(d["In Assessment Pool"].any()),
            "TR Cleared": bool(d["TR Cleared"].any()),
            "Companies": d["Company"].nunique(),
            "Companies reached " + (STAGES[threshold] + "+" if threshold >= 0 else "any stage"):
                d.loc[d["_hc"] >= threshold, "Company"].nunique() if threshold >= 0 else d["Company"].nunique(),
            "Highest stage cleared": STAGES[top] if top >= 0 else "",
            "Where": ", ".join(dict.fromkeys(where)),
            "Rejected at": ", ".join(f"{c} – {s}" for c, s in zip(rej["Company"], rej["Rejected at"])),
            "Hired at (Job IDs)": ", ".join(dict.fromkeys(hired["Job ID"])),
            "Guidance flag": flag,
            "_top": top, "_hired": len(hired) > 0, "_rej": len(rej) > 0,
            "_inproc": bool((d["Outcome"] == "In process").any()),
            "_n_at": int((d["_hc"] >= threshold).sum()) if threshold >= 0 else len(d),
        })
    return pd.DataFrame(rows)


def stage_funnel(r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (comp, job), d in r.groupby(["Company", "Job ID"], sort=False):
        used = frozenset().union(*d["_used"])
        for i, s in enumerate(STAGES):
            if i not in used:
                continue
            col = d[s]
            reached = int(((col != "") | (d["_hc"] >= i)).sum())  # implied if a later stage was reached
            if not reached:
                continue
            cleared = int((d["_hc"] >= i).sum())
            rows.append({"Company": comp, "Job ID": job, "Stage": s, "Reached": reached, "Cleared": cleared,
                         "Rejected": int((col == "Rejected").sum()), "Pending": int((col == "Pending").sum()),
                         "No-show / Dropped": int(col.isin(["No-show", "Dropped"]).sum()),
                         "Cleared % of reached": round(100 * cleared / reached, 1)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def render(pools, tr_sum, tr_rule, academy_kw, intensive_kw):
    st.title("🎯 Student Stages — Guidance Pool")
    st.caption("Reads every company sheet, maps each company's rounds to standard stages "
               "(Screening → TR1 → TR2+ → HR / MR → Final / Offer) and lets you filter students by how far they got.")

    i1, i2 = st.columns(2)
    with i1:
        comp_in = data_input("1 · Companies (Job ID, Org ID, Order, Company Name, Company Sheet)", "sm_in")
    with i2:
        apps = data_input("2 · Optional · Applications (only needed when several Job IDs share one sheet)", "sm_apps")
    if comp_in is None:
        st.info("Add the companies CSV.")
        st.stop()

    cols = list(comp_in.columns)
    with st.expander("Columns", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            job_col = _pick("Job ID", comp_in, core.find_col(comp_in, ["jobid"]), "sm_job")
            org_col = _pick("Org ID", comp_in, core.find_col(comp_in, ["orgid"]), "sm_org")
            ord_col = _pick("Order", comp_in, core.find_col(comp_in, ["order"]), "sm_order")
        with c2:
            comp_col = _pick("Company Name", comp_in, core.find_col(comp_in, ["companyname", "company"],
                                                                    exclude=("sheet", "link", "id")), "sm_comp")
            link_col = _pick("Company Sheet (link)", comp_in,
                             core.find_col(comp_in, ["companysheet", "sheetlink", "sheet"], exclude=("hubspot",)),
                             "sm_link")
            if apps is not None:
                a_det = core.detect_columns(apps)
                a_mob = _pick("Applications · Phone", apps, a_det["mobile"], "sm_a_mob")
                a_job = _pick("Applications · Job ID", apps, core.find_col(apps, ["jobid"]), "sm_a_job")
    if not (comp_col and link_col):
        st.warning("Pick the Company Name and Company Sheet columns.")
        st.stop()
    all_tabs = st.toggle("Read every tab of each sheet", value=True, key="sm_alltabs")

    t = comp_in.copy()
    t["_comp"] = t[comp_col].fillna("").astype(str).str.strip()
    t["_link"] = t[link_col].fillna("").astype(str).str.strip()
    t["_job"] = t[job_col].fillna("").astype(str).str.strip() if job_col else ""
    t["_org"] = t[org_col].fillna("").astype(str).str.strip() if org_col else ""
    t["_order"] = pd.to_numeric(t[ord_col], errors="coerce") if ord_col else range(len(t))
    no_link = t[~t["_link"].str.contains("docs.google.com/spreadsheets", regex=False)]
    t = t.drop(no_link.index)
    t["_sheet"] = [sheet_id(l) if all_tabs else f"{sheet_id(l)}#{link_gid(l)}" for l in t["_link"]]
    order = t.groupby("_comp")["_order"].min().sort_values(kind="stable")
    sheets = t.drop_duplicates(["_comp", "_sheet"])
    sheet_jobs = t.groupby(["_comp", "_sheet"])["_job"].agg(lambda s: [j for j in dict.fromkeys(s) if j]).to_dict()
    org_of = dict(zip(t["_job"], t["_org"]))

    b1, b2 = st.columns([1, 5])
    if b1.button("▶ Fetch & analyse", type="primary", key="sm_go_btn"):
        st.session_state["sm_go"] = True
    if b2.button("↻ Refresh sheets", key="sm_refresh"):
        fetch_all.clear()
    if not st.session_state.get("sm_go"):
        st.caption(f"{len(sheets)} sheet(s) · {t['_comp'].nunique()} companies"
                   + (f" · {len(no_link)} row(s) without a sheet link" if len(no_link) else ""))
        st.stop()

    jobs = tuple((c, l, all_tabs) for c, l in zip(sheets["_comp"], sheets["_link"]))
    with st.spinner(f"Reading {len(jobs)} sheet(s)…"):
        tabs = fetch_all(jobs)
    key_of = {(c, l): k for c, l, k in zip(sheets["_comp"], sheets["_link"], sheets["_sheet"])}

    # ---------------- round -> stage mapping ----------------
    map_rows, log, usable = [], [], []
    for tb in tabs:
        e = {"Company": tb["Company"], "Tab": tb["Tab"], "Rows": 0, "Status": "", "Link": tb["Link"]}
        if tb["df"] is None:
            e["Status"] = "❌ " + tb["Note"]
        else:
            df = tb["df"]
            e["Rows"] = len(df)
            det = core.detect_columns(df)
            rc = rounds.detect_round_columns(df)
            if not (det["mobile"] or det["email"] or det["uid"]):
                e["Status"] = "skipped — no mobile / email column"
            elif not rc:
                e["Status"] = "skipped — no round columns"
            else:
                e["Status"] = "✓ used"
                usable.append(tb)
                map_rows += [{"Company": tb["Company"], "Tab": tb["Tab"], "Column": rounds.norm_name(c),
                              "_col": c, "Stage": guess_stage(c)} for c in rc]
        log.append(e)
    mp = pd.DataFrame(map_rows, columns=["Company", "Tab", "Column", "_col", "Stage"])
    with st.expander(f"Round → stage mapping ({int((mp['Stage'] == IGNORE).sum())} column(s) ignored)", expanded=False):
        st.caption("Auto-mapped from the column names. Fix any wrong ones; '(ignore)' columns aren't counted.")
        ed = st.data_editor(
            mp.drop(columns="_col"), hide_index=True, width="stretch", disabled=["Company", "Tab", "Column"],
            key=f"sm_map_{len(mp)}_{hash(tuple(mp['Column']))}",
            column_config={"Stage": st.column_config.SelectboxColumn("Stage", options=STAGES + [IGNORE])},
        )
    mp["Stage"] = ed["Stage"].values

    # ---------------- build student x company rows ----------------
    applied = {}
    if apps is not None and a_mob and a_job:
        aa = pd.DataFrame({"m": apps[a_mob].map(core.norm_mobile), "j": apps[a_job].fillna("").astype(str).str.strip()})
        applied = aa.dropna().groupby("m")["j"].agg(set).to_dict()
    parts = []
    for tb in usable:
        colmap = {r["_col"]: r["Stage"] for _, r in mp[(mp["Company"] == tb["Company"]) & (mp["Tab"] == tb["Tab"])].iterrows()}
        rr = tab_rows(tb["df"], colmap, pools, tr_sum, academy_kw, intensive_kw)
        sj = sheet_jobs.get((tb["Company"], key_of.get((tb["Company"], tb["Link"]))), [])
        jl = []
        for mob in rr["Mobile"]:
            mine = [j for j in sj if j in applied.get(mob, ())] if len(sj) > 1 else []
            jl.append(mine or ([", ".join(sj)] if sj else [""]))
        parts.append(rr.assign(Company=tb["Company"], Tab=tb["Tab"], **{"Job ID": jl}).explode("Job ID"))
    if not parts:
        st.error("No candidate tabs could be read.")
        st.dataframe(pd.DataFrame(log), hide_index=True, width="stretch")
        st.stop()
    rows = pd.concat(parts, ignore_index=True)
    rows["_key"] = rows["Mobile"].where(rows["Mobile"].fillna("") != "", "p:" + rows["_person"].astype(str))

    # One row per student x company x Job ID: best status per stage across tabs
    agg = {c: "first" for c in rows.columns if c not in ("Company", "Job ID", "_key", *STAGES, "Tab", "Rounds")}
    agg.update({s: (lambda x: min(x, key=lambda v: _ST_RANK.get(v, 9))) for s in STAGES})
    agg.update({"Tab": lambda x: ", ".join(dict.fromkeys(x)), "Rounds": lambda x: " || ".join(dict.fromkeys(v for v in x if v))})
    agg.update({c: "any" for c in ("In Bucket", "In Assessment Pool", "TR Cleared")})
    agg["_used"] = lambda x: frozenset().union(*x)
    sc = rows.groupby(["Company", "Job ID", "_key"], sort=False).agg(agg).reset_index().drop(columns="_key")
    sc = summarise(sc)
    sc["Org ID"] = sc["Job ID"].map(lambda j: ", ".join(dict.fromkeys(org_of.get(x.strip(), "") for x in str(j).split(","))).strip(", "))
    sc["_order"] = sc["Company"].map(order)

    # ---------------- filters ----------------
    st.subheader("Guidance pool")
    f1, f2, f3, f4 = st.columns(4)
    thr_lbl = f1.selectbox("Cleared at least", ["Any stage"] + STAGES, index=2, key="sm_thr")
    thr = -1 if thr_lbl == "Any stage" else _S_IDX[thr_lbl]
    min_n = f2.number_input("…in at least N companies", 1, 50, 1, key="sm_n")
    masks = pool_masks(sc, pools["bucket_values"])
    grp = f3.selectbox("Pool group", list(masks), key="sm_grp")
    outs = ["Hired", "Rejected (not hired)", "In process (not hired)", "Other"]
    s_out = f4.multiselect("Student outcome", outs, default=outs, key="sm_out")
    comps = list(order.index)
    s_comp = st.multiselect("Only these companies (blank = all)", comps, key="sm_comps")

    base = sc[masks[grp]]
    if s_comp:
        base = base[base["Company"].isin(s_comp)]
    stu = student_view(base, thr) if len(base) else pd.DataFrame()
    if len(stu):
        stu = stu[(stu["_n_at"] >= min_n) & ((stu["_top"] >= thr) if thr >= 0 else True)]
        kind = pd.Series("Other", index=stu.index)
        kind[stu["_inproc"]] = "In process (not hired)"
        kind[stu["_rej"]] = "Rejected (not hired)"
        kind[stu["_hired"]] = "Hired"
        stu = stu[kind.isin(s_out)]
        stu = stu.sort_values(["_top", "_n_at"], ascending=False)

    k = st.columns(4)
    k[0].metric("Students in pool", f"{len(stu):,}")
    k[1].metric("Hired", int(stu["_hired"].sum()) if len(stu) else 0)
    k[2].metric("Rejected, not hired", int((stu["_rej"] & ~stu["_hired"]).sum()) if len(stu) else 0)
    k[3].metric("Companies covered", base["Company"].nunique())

    tab_p, tab_d, tab_f, tab_m = st.tabs(["Guidance pool", "Student × company stages", "Stage funnel per Job ID",
                                          "Sheets read"])
    with tab_p:
        show = stu.drop(columns=[c for c in stu.columns if c.startswith("_")]) if len(stu) else stu
        st.dataframe(show, hide_index=True, width="stretch")
        st.download_button("⬇️ Download guidance pool (CSV)", show.to_csv(index=False).encode("utf-8"),
                           file_name=f"guidance_pool_{re.sub(r'[^A-Za-z0-9]+', '_', thr_lbl)}.csv",
                           mime="text/csv", key="sm_dl_pool")
        st.caption("A stage counts as cleared if it's marked Cleared, or if the student reached a later stage. "
                   "Guidance flag = where they most often get rejected.")

    with tab_d:
        keep = set(stu["Mobile"]) if len(stu) else set()
        only = st.checkbox("Only students in the guidance pool", value=True, key="sm_only")
        d = base[base["Mobile"].isin(keep)] if only else base
        d = d.sort_values(["_order", "Company", "Name"], kind="stable")
        dcols = ["Name", "Mobile", "Bucket", "TR Cleared", "Company", "Job ID", "Org ID", *STAGES,
                 "Highest cleared", "Outcome", "Rejected at", "Tab", "Rounds"]
        st.dataframe(d[dcols], hide_index=True, width="stretch")
        st.download_button("⬇️ Download (CSV)", d[dcols].to_csv(index=False).encode("utf-8"),
                           file_name="student_company_stages.csv", mime="text/csv", key="sm_dl_det")

    with tab_f:
        fn = stage_funnel(base.sort_values("_order", kind="stable"))
        st.dataframe(fn, hide_index=True, width="stretch")
        if len(fn):
            st.download_button("⬇️ Download (CSV)", fn.to_csv(index=False).encode("utf-8"),
                               file_name="stage_funnel_per_job.csv", mime="text/csv", key="sm_dl_fn")
        st.caption(f"Pool group: {grp}. Reached includes students who got to a later stage.")

    with tab_m:
        st.dataframe(pd.DataFrame(log), hide_index=True, width="stretch",
                     column_config={"Link": st.column_config.LinkColumn("Link", display_text="Open")})
        if len(no_link):
            st.markdown("**Rows without a company sheet link**")
            st.dataframe(comp_in.loc[no_link.index], hide_index=True, width="stretch")
        st.caption(f"TR Cleared rule: {tr_rule}.")