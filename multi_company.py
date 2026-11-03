"""Mode 3 — All companies.

Reads the tracker (one row per company / JD with a 'Company Sheet Link'), fetches every
company sheet that is shared as 'Anyone with the link -> Viewer', matches each candidate
against the static pools and counts, per company:
    attending (in the sheet)  ->  interviewed  ->  rejected / hired / in process.
"""
from __future__ import annotations

import html
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st

import core
import rounds
from inputs import data_input

NONE = "— none —"
NOT_IN_SHEET = "Not in company sheet"
SHEET_OUTCOMES = ["Hired / Offer", "In process", "Rejected", "No-show / Dropped", "Not interviewed yet"]
OUTCOMES = SHEET_OUTCOMES + [NOT_IN_SHEET]
_OUTCOME_RANK = {o: i for i, o in enumerate(OUTCOMES)}  # best first, used when a person appears more than once
CAT_ORDER = ["Intensive", "Academy", "External", "Not in applications"]
_CAT_RANK = {"Intensive": 0, "Academy": 1, "External": 2}
_HIRED_RX = re.compile(r"hired|\boffer|joined|onboard")
_HOLD_RX = re.compile(r"\bhold\b")


# --------------------------------------------------------------------------- #
# Google Sheets: list tabs + fetch one tab (plain functions, safe to run in threads)
# --------------------------------------------------------------------------- #
def sheet_id(url: str):
    m = re.search(r"/spreadsheets/d/([\w-]+)", str(url))
    return m.group(1) if m and m.group(1) != "e" else None


def link_gid(url: str):
    m = re.search(r"[#?&]gid=(\d+)", str(url))
    return m.group(1) if m else None


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def list_tabs(sid: str) -> list[tuple[str, str]]:
    """[(tab name, gid), ...] read from the sheet's public htmlview page. [] if it can't be read."""
    try:
        text = _get(f"https://docs.google.com/spreadsheets/d/{sid}/htmlview").decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return []
    tabs = []
    for gid, name in re.findall(r'sheet-button-(\d+)"[^>]*>\s*<a[^>]*>(.*?)</a>', text, flags=re.S):
        tabs.append((html.unescape(re.sub(r"<[^>]+>", "", name)).strip(), gid))
    if not tabs:  # older page layout: items.push({name: "JD-1", ..., gid: "123"})
        for name, gid in re.findall(r'name:\s*"((?:[^"\\]|\\.)*)"[^}]*?gid:\s*"(\d+)"', text):
            tabs.append((name.encode().decode("unicode_escape"), gid))
    return list(dict.fromkeys(tabs))


def fetch_tab(sid: str, gid: str | None) -> pd.DataFrame:
    url = f"https://docs.google.com/spreadsheets/d/{sid}/edit" + (f"#gid={gid}" if gid else "")
    return core.read_csv_bytes(rounds.fetch_gsheet(url))


def _fetch_job(job):
    """job = (company, link, all_tabs) -> list of tab results."""
    company, link, all_tabs = job
    sid = sheet_id(link)
    if not sid:
        return [{"Company": company, "Link": link, "Tab": "", "gid": None, "df": None,
                 "Note": "Not a Google Sheets link"}]
    tabs = list_tabs(sid) if all_tabs else []
    note = ""
    if not tabs:
        g = link_gid(link)
        tabs = [(f"gid {g}" if g else "first tab", g)]
        if all_tabs:
            note = "Couldn't list the tabs — read only the linked tab. "
    out = []
    for name, gid in tabs:
        try:
            out.append({"Company": company, "Link": link, "Tab": name, "gid": gid,
                        "df": fetch_tab(sid, gid), "Note": note})
        except Exception as e:  # noqa: BLE001
            out.append({"Company": company, "Link": link, "Tab": name, "gid": gid, "df": None,
                        "Note": note + str(e)})
    return out


@st.cache_data(show_spinner=False, ttl=600)
def fetch_all(jobs: tuple, workers: int = 8) -> list[dict]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return [tab for res in ex.map(_fetch_job, jobs) for tab in res]


# --------------------------------------------------------------------------- #
# Analysis of one tab (no Streamlit here, so it can be tested on its own)
# --------------------------------------------------------------------------- #
def analyse_tab(df: pd.DataFrame, pools, tr_sum, academy_kw, intensive_kw):
    """One row per candidate row in the tab, or (None, reason) if the tab isn't a candidate list."""
    df = df.reset_index(drop=True)
    round_cols = rounds.detect_round_columns(df)
    det = core.detect_columns(df)
    if not (det["mobile"] or det["email"] or det["uid"]):
        return None, "skipped — no mobile / email column"
    if not round_cols:
        return None, "skipped — no round / status columns"

    cols = {k: det[k] for k in ("name", "mobile", "email", "uid", "product")}
    prepared = core.prepare_company(df, cols, None, {}, None, None, academy_kw, intensive_kw)
    has_id = prepared[["_uid", "_mob", "_email"]].notna().any(axis=1)
    m = core.match_company(prepared, pools, tr_sum)

    raw = df[round_cols].apply(lambda s: s.fillna("").astype(str).str.strip().str.lower())
    stages = pd.DataFrame({c: df[c].map(rounds.classify) for c in round_cols}, index=df.index)

    interviewed = stages.isin(["Cleared", "Rejected"]).any(axis=1) | raw.apply(
        lambda s: s.str.contains(_HOLD_RX)).any(axis=1)
    rejected = (stages == "Rejected").any(axis=1)
    hired = ((stages == "Cleared") & raw.apply(lambda s: s.str.contains(_HIRED_RX))).any(axis=1)
    noshow = stages.isin(["No-show", "Dropped"]).any(axis=1)

    outcome = pd.Series("Not interviewed yet", index=df.index, dtype=object)
    outcome[noshow & ~interviewed] = "No-show / Dropped"
    outcome[interviewed] = "In process"
    outcome[rejected] = "Rejected"
    outcome[hired] = "Hired / Offer"

    out = pd.DataFrame({
        "_person": m["_person"],
        "Name": m["Name"],
        "Mobile": m["Mobile"],
        "Email": m["_email"],
        "Product": m["Product"],
        "Category": m["Category"],
        "In Bucket": m["In Bucket"],
        "Bucket": m["Bucket"],
        "Cycle": m["Cycle"],
        "In Assessment Pool": m["In Assessment Pool"],
        "TR Cleared": m["TR Cleared"],
        "TR1 Status": m["TR1 Status"],
        "TR2 Status": m["TR2 Status"],
        "Interviewed": interviewed,
        "Outcome": outcome,
        "Rounds": df[round_cols].apply(
            lambda r: " | ".join(f"{_lbl(c)}: {v}" for c, v in r.items() if str(v).strip() not in ("", "nan")),
            axis=1),
        "Resume": df[det["resume"]] if det["resume"] else None,
    })
    other = sorted({v for c in round_cols for v, st_ in zip(raw[c], stages[c]) if st_ == "Other"})
    info = {"round_cols": ", ".join(_lbl(c) for c in round_cols), "no_id": int((~has_id).sum()), "other": other}
    return out[has_id.values], info


def _lbl(c):
    return str(c).replace("\n", " ").strip()


def _nk(x) -> str:
    """Company-name key: 'AdSocial.ai' -> 'adsocialai'."""
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def _person_key(df: pd.DataFrame) -> pd.Series:
    """Mobile first (so applications and sheet rows meet), else the existing person id."""
    mob = df["Mobile"].map(lambda v: f"m:{v}" if isinstance(v, str) and v else None)
    return mob.combine_first(df["_person"])


# --------------------------------------------------------------------------- #
# Applications file (all companies)
# --------------------------------------------------------------------------- #
def detect_app_columns(apps: pd.DataFrame) -> dict:
    det = core.detect_columns(apps)
    return {
        "company": core.find_col(apps, ["companyname", "company"], exclude=("id", "link")),
        "job": core.find_col(apps, ["jobid", "jobiduuid"]),
        "name": det["name"], "mobile": det["mobile"], "email": det["email"], "uid": det["uid"],
        "product": det["product"], "resume": det["resume"],
    }


def prepare_apps(apps, cols, tracker, comp_col, job_col, name_cols, pools, tr_sum, academy_kw, intensive_kw):
    """One row per application, with the tracker's company name and pool flags."""
    a = apps.reset_index(drop=True)
    raw_comp = a[cols["company"]].fillna("").astype(str).str.strip() if cols["company"] else pd.Series("", index=a.index)

    # Company: tracker company via Job ID first, then via any tracker name column, else the name as given
    company = pd.Series(None, index=a.index, dtype=object)
    jd = pd.Series(None, index=a.index, dtype=object)
    if job_col and cols["job"]:
        jt = tracker[[job_col, comp_col, "_JD"]].dropna(subset=[job_col])
        jkey = a[cols["job"]].map(core.norm_text)
        jmap = dict(zip(jt[job_col].map(core.norm_text), jt[comp_col].astype(str).str.strip()))
        company = jkey.map(jmap).astype(object)
        jd = jkey.map(dict(zip(jt[job_col].map(core.norm_text), jt["_JD"]))).astype(object)
    nmap = {}
    for c in [comp_col] + [c for c in name_cols if c != comp_col]:
        for k, v in zip(tracker[c], tracker[comp_col]):
            if isinstance(k, str) and k.strip() and isinstance(v, str):
                nmap.setdefault(_nk(k), v.strip())
    company = company.combine_first(raw_comp.map(lambda x: nmap.get(_nk(x), x)))

    pc = {k: cols.get(k) for k in ("name", "mobile", "email", "uid", "product")}
    prepared = core.prepare_company(a, pc, None, {}, None, None, academy_kw, intensive_kw)
    m = core.match_company(prepared, pools, tr_sum)
    out = pd.DataFrame({
        "Company": company, "JD": jd.fillna("Unknown JD"), "JD source": jd.notna().map({True: "Job ID", False: ""}),
        "_person": m["_person"], "Name": m["Name"], "Mobile": m["Mobile"], "Email": m["_email"],
        "Product": m["Product"], "Category": m["Category"],
        "In Bucket": m["In Bucket"], "Bucket": m["Bucket"], "Cycle": m["Cycle"],
        "In Assessment Pool": m["In Assessment Pool"], "TR Cleared": m["TR Cleared"],
        "TR1 Status": m["TR1 Status"], "TR2 Status": m["TR2 Status"],
        "Resume": a[cols["resume"]] if cols.get("resume") else None,
        "Job ID": a[cols["job"]] if cols.get("job") else None,
    })
    out = out[out["Company"].fillna("").astype(str).str.strip() != ""]
    return out.assign(Applied=True, **{"In Sheet": False, "Interviewed": False, "Outcome": NOT_IN_SHEET,
                                       "Tab": "", "Rounds": ""})


# --------------------------------------------------------------------------- #
# Combine applications + sheet rows -> one row per company x person
# --------------------------------------------------------------------------- #
def merge_people(rows: pd.DataFrame, has_apps: bool, per_jd: bool = False) -> pd.DataFrame:
    """One row per company (or company x JD) x person."""
    keys = ["Company", "JD", "_key"] if per_jd else ["Company", "_key"]
    rows = rows.copy()
    rows["_key"] = _person_key(rows)
    rows["_rank"] = rows["Outcome"].map(_OUTCOME_RANK)
    # Applications carry the product; prefer the strongest product if someone applied twice
    rows["_cat_rank"] = rows["Category"].map(_CAT_RANK).where(rows["Applied"], 9)
    rows["Category"] = rows["Category"].where(rows["Applied"]) if has_apps else rows["Category"]
    rows = rows.sort_values(["_rank", "_cat_rank"], kind="stable")
    skip = (*keys, "_rank", "_cat_rank", "_person")
    agg = {c: "first" for c in rows.columns if c not in skip}
    agg.update({c: "any" for c in ("Applied", "In Sheet", "Interviewed", "In Bucket", "In Assessment Pool", "TR Cleared")})
    agg["Tab"] = lambda s: ", ".join(dict.fromkeys(v for v in s if v))
    agg["Rounds"] = lambda s: " || ".join(dict.fromkeys(v for v in s if v))
    agg["JD source"] = lambda s: next((v for v in s if isinstance(v, str) and v), "")  # sheet row's source first
    if not per_jd:
        agg["JD"] = lambda s: ", ".join(sorted(dict.fromkeys(v for v in s if isinstance(v, str) and v)))
    g = rows.groupby(keys, sort=False)
    out = g.agg(agg)
    out.insert(0, "Applications", g["Applied"].sum().astype(int))
    out = out.reset_index()
    if has_apps:  # product from the applications, strongest first; sheet-only people have none
        best = rows[rows["Applied"]].sort_values("_cat_rank").groupby(keys)["Category"].first()
        out["Category"] = [best.get(t) for t in zip(*(out[k] for k in keys))]
        out["Category"] = out["Category"].fillna("Not in applications")
    else:
        out["Category"] = out["Category"].fillna("External")
    out = out.drop(columns="_key")
    return out


def assign_sheet_jd(sheet_rows: pd.DataFrame, app_rows) -> pd.DataFrame:
    """Job ID -> student (applications) -> their row in the company sheet (by mobile).
    Someone who applied to several Job IDs of the company gets their sheet status under each of them.
    Sheet candidates with no application go to 'Not in applications'."""
    r = sheet_rows.copy()
    r["_key"] = _person_key(r)
    applied = {}
    if app_rows is not None and len(app_rows):
        a = app_rows.assign(_key=_person_key(app_rows))
        a = a[a["JD"] != "Unknown JD"]
        applied = a.groupby(["Company", "_key"])["JD"].agg(lambda s: list(dict.fromkeys(s))).to_dict()
    r["JD"] = [applied.get((c, k)) or ["Not in applications"] for c, k in zip(r["Company"], r["_key"])]
    r["JD source"] = ["Job ID" if (c, k) in applied else "" for c, k in zip(r["Company"], r["_key"])]
    return r.explode("JD").drop(columns="_key").reset_index(drop=True)


UNIT_SEP = "\x1f"


def split_units(df: pd.DataFrame, job_ids: dict) -> pd.DataFrame:
    """'Company<sep>JD' in the Company column -> Company | JD | Job ID columns."""
    if df is None or not len(df) or "Company" not in df:
        return df
    parts = df["Company"].astype(str).str.split(UNIT_SEP, n=1, expand=True).reindex(columns=[0, 1])
    out = df.drop(columns="Company")
    out.insert(0, "Job ID", [job_ids.get((c, j), "") if isinstance(j, str) else "" for c, j in zip(parts[0], parts[1])])
    out.insert(0, "JD", parts[1].fillna(""))
    out.insert(0, "Company", parts[0])
    return out


def student_table(cand_jd: pd.DataFrame, job_ids: dict) -> pd.DataFrame:
    """One row per unique student across all companies (per-JD rows in, students out)."""
    c = cand_jd.copy()
    c["_sid"] = c["Mobile"].where(c["Mobile"].fillna("") != "", "e:" + c["Email"].fillna("")) \
        .where(lambda x: x != "e:", "n:" + c["Name"].fillna("").str.lower().str.strip())
    c["_hired"] = c["Outcome"] == "Hired / Offer"
    c["_rej"] = c["Outcome"] == "Rejected"
    c["_proc"] = c["Outcome"] == "In process"
    c["_unit"] = c["Company"] + " · " + c["JD"].fillna("")
    c["_job"] = [job_ids.get((co, j), "") for co, j in zip(c["Company"], c["JD"])]
    first = lambda s: next((v for v in s if isinstance(v, str) and v), "")  # noqa: E731
    join = lambda s: ", ".join(sorted(dict.fromkeys(v for v in s if isinstance(v, str) and v)))  # noqa: E731
    g = c.groupby("_sid", sort=False)
    out = pd.DataFrame({
        "Name": g["Name"].agg(first),
        "Mobile": g["Mobile"].agg(first),
        "Email": g["Email"].agg(first),
        "Product": g["Product"].agg(first),
        "Category": g["Category"].agg(lambda s: next((v for v in sorted(s.dropna(), key=lambda x: _CAT_RANK.get(x, 9))), "")),
        "Bucket": g["Bucket"].agg(first),
        "Cycle": g["Cycle"].agg(first),
        "In Bucket": g["In Bucket"].any(),
        "In Assessment Pool": g["In Assessment Pool"].any(),
        "TR Cleared": g["TR Cleared"].any(),
        "Total applications": g["Applications"].sum().astype(int),
        "Companies applied": c[c["Applied"]].groupby("_sid")["Company"].nunique(),
        "JDs applied": c[c["Applied"]].groupby("_sid")["_unit"].nunique(),
        "Companies in sheet": c[c["In Sheet"]].groupby("_sid")["Company"].nunique(),
        "Interviewed (JDs)": g["Interviewed"].sum().astype(int),
        "In process": g["_proc"].sum().astype(int),
        "Rejected": g["_rej"].sum().astype(int),
        "Hired / Offer": g["_hired"].sum().astype(int),
        "Companies applied to (list)": c[c["Applied"]].groupby("_sid")["Company"].agg(join),
        "Hired at (Job IDs)": c[c["_hired"]].groupby("_sid")["_job"].agg(join),
        "Rejected at (Job IDs)": c[c["_rej"]].groupby("_sid")["_job"].agg(join),
    })
    for col in ("Companies applied", "JDs applied", "Companies in sheet"):
        out[col] = out[col].fillna(0).astype(int)
    for col in ("Companies applied to (list)", "Hired at (Job IDs)", "Rejected at (Job IDs)"):
        out[col] = out[col].fillna("")
    return out.sort_values(["Total applications", "Interviewed (JDs)"], ascending=False).reset_index(drop=True)


JOB_POOL_COLS = ["Applied", "Sent to Company", "Interviewed", "Hired / Offer", "Rejected", "In process",
                 "No-show / Dropped", "Not interviewed yet"]


def job_pool_table(cand_jd: pd.DataFrame, job_ids: dict, groups: dict) -> pd.DataFrame:
    """Long table: Pool group x Company x Job ID with the applied -> sent -> outcome funnel.
    groups: {name: mask over cand_jd}. Rows without a Job ID (not in applications) are left out."""
    c = cand_jd.copy()
    raw_job = c["Job ID"] if "Job ID" in c else pd.Series(None, index=c.index)
    c["Job ID"] = [job_ids.get((co, j)) or (r if isinstance(r, str) else "") for co, j, r in
                   zip(c["Company"], c["JD"], raw_job)]
    c["_sent"] = c["Applied"] & c["In Sheet"]
    has_job = c["Job ID"] != ""
    out = []
    for name, mask in groups.items():
        d = c[mask & has_job]
        if d.empty:
            continue
        g = d.groupby(["Company", "Job ID"])
        t = pd.DataFrame({"Applied": g["Applied"].sum(), "Sent to Company": g["_sent"].sum(),
                          "Interviewed": (d["Interviewed"] & d["_sent"]).groupby([d["Company"], d["Job ID"]]).sum()})
        s = d[d["_sent"]]
        oc = pd.crosstab([s["Company"], s["Job ID"]], s["Outcome"]).reindex(columns=SHEET_OUTCOMES, fill_value=0)
        t = t.join(oc).fillna(0).astype(int)[JOB_POOL_COLS].sort_values(["Applied"], ascending=False)
        tot = t.sum()
        t = t.reset_index()
        t = pd.concat([t, pd.DataFrame([{"Company": "Total", "Job ID": "", **tot}])], ignore_index=True)
        t.insert(0, "Pool group", name)
        out.append(t)
    if not out:
        return pd.DataFrame(columns=["Pool group", "Company", "Job ID", *JOB_POOL_COLS])
    t = pd.concat(out, ignore_index=True)
    pct = lambda a, b: (100 * t[a] / t[b].where(t[b] > 0)).round(1)  # noqa: E731
    t["Applied → Sent to Company %"] = pct("Sent to Company", "Applied")
    t["Interviewed % of sheet"] = pct("Interviewed", "Sent to Company")
    t["Rejected % of interviewed"] = pct("Rejected", "Interviewed")
    return t


JOB_POOL_PCT = ["Applied → Sent to Company %", "Interviewed % of sheet", "Rejected % of interviewed"]


def job_pool_wide(long: pd.DataFrame, groups: list) -> pd.DataFrame:
    """One row per Company x Job ID. Two-level columns: main column = metric (Applied, Sent to Company…),
    sub-columns = pool groups (Bucket A, B, C, D, TR Cleared)."""
    d = long[long["Company"] != "Total"]
    cols = pd.MultiIndex.from_product([JOB_POOL_COLS + JOB_POOL_PCT, groups])
    if d.empty:
        return pd.DataFrame(columns=cols, index=pd.MultiIndex.from_tuples([], names=["Company", "Job ID"]))
    w = d.pivot_table(index=["Company", "Job ID"], columns="Pool group", values=JOB_POOL_COLS,
                      aggfunc="sum", fill_value=0)
    w = w.reindex(columns=pd.MultiIndex.from_product([JOB_POOL_COLS, groups]), fill_value=0).astype(int)
    w = w.sort_values((JOB_POOL_COLS[0], groups[0]), ascending=False)
    w.loc[("Total", ""), :] = w.sum()
    w = w.astype(int)
    pct = lambda a, b: (100 * w[a] / w[b].where(w[b] > 0)).round(1)  # noqa: E731
    for name, (a, b) in zip(JOB_POOL_PCT, [("Sent to Company", "Applied"), ("Interviewed", "Sent to Company"),
                                           ("Rejected", "Interviewed")]):
        p = pct(a, b)
        for g in groups:
            w[(name, g)] = p[g]
    return w[cols]


def grouped_html(df: pd.DataFrame) -> str:
    """HTML table with merged two-level headers (Streamlit's grid can't show them)."""
    fmt = {c: "{:.1f}" for c in df.columns if c[0] in JOB_POOL_PCT}
    sty = df.style.format(fmt, na_rep="–").format_index(lambda v: v, axis=0)
    css = ("<style>.jp-wrap{overflow:auto;max-height:600px;border:1px solid rgba(128,128,128,.35);border-radius:6px}"
           ".jp-wrap table{border-collapse:collapse;font-size:13px;white-space:nowrap}"
           ".jp-wrap th,.jp-wrap td{border:1px solid rgba(128,128,128,.3);padding:4px 8px;text-align:center}"
           ".jp-wrap thead th{position:sticky;top:0;background:var(--background-color,#0e1117);z-index:1}"
           ".jp-wrap thead tr:nth-child(2) th{top:27px}"
           ".jp-wrap tbody th{text-align:left}</style>")
    return css + '<div class="jp-wrap">' + sty.to_html() + "</div>"


def to_excel_bytes(df: pd.DataFrame) -> bytes | None:
    try:
        import io
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            df.to_excel(xw, sheet_name="By bucket & TR")
        return buf.getvalue()
    except ImportError:
        return None


def pool_masks(cand: pd.DataFrame, bucket_values) -> dict:
    in_pool = cand["In Bucket"] | cand["In Assessment Pool"]
    masks = {
        "Everyone": pd.Series(True, index=cand.index),
        "From placement pool (bucket or assessment)": in_pool,
        "From bucket pool": cand["In Bucket"],
    }
    for b in bucket_values:
        masks[b if "bucket" in b.lower() else f"Bucket {b}"] = cand["Bucket"] == b
    masks["From assessment pool"] = cand["In Assessment Pool"]
    masks["TR Cleared"] = cand["TR Cleared"]
    masks["Not in any pool"] = ~in_pool
    return masks


def company_table(cand: pd.DataFrame, mask: pd.Series, has_apps: bool, sheet_companies: set) -> pd.DataFrame:
    """Per company: applied -> in company sheet -> interviewed -> outcome."""
    c = cand[mask]
    cols = (["Applied", "Applied & in sheet"] if has_apps else []) + ["In company sheet", "Interviewed"] + SHEET_OUTCOMES
    if c.empty:
        return pd.DataFrame(columns=["Company", *cols])
    c = c.assign(_ais=c["Applied"] & c["In Sheet"])
    g = c.groupby("Company")
    t = pd.DataFrame({
        "Applied": g["Applied"].sum(),
        "Applied & in sheet": g["_ais"].sum(),
        "In company sheet": g["In Sheet"].sum(),
        "Interviewed": g["Interviewed"].sum(),
    })
    s = c[c["In Sheet"]]
    oc = pd.crosstab(s["Company"], s["Outcome"]).reindex(index=t.index, columns=SHEET_OUTCOMES, fill_value=0)
    t = pd.concat([t, oc], axis=1).fillna(0).astype(int)[cols]
    if has_apps:
        t["In sheet, not in applications"] = t["In company sheet"] - t["Applied & in sheet"]
    t = t.sort_values(cols[0], ascending=False)
    t.loc["Total"] = t.sum()
    if has_apps:
        t["Applied → sheet %"] = (100 * t["Applied & in sheet"] / t["Applied"].where(t["Applied"] > 0)).round(1)
    t["Interviewed % of sheet"] = (100 * t["Interviewed"] / t["In company sheet"].where(t["In company sheet"] > 0)).round(1)
    t["Rejected % of interviewed"] = (100 * t["Rejected"] / t["Interviewed"].where(t["Interviewed"] > 0)).round(1)
    t.insert(0, "Sheet read", ["" if i == "Total" else ("Yes" if i in sheet_companies else "No") for i in t.index])
    t.index.name = "Company"
    return t.reset_index()


def product_table(cand: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    c = cand[mask]
    cats = [x for x in CAT_ORDER if x in set(c["Category"])]
    rows = []
    for comp, d in c.groupby("Company"):
        row = {"Company": comp}
        for cat in cats:
            x = d[d["Category"] == cat]
            if cat != "Not in applications":
                row[f"{cat} · applied"] = int(x["Applied"].sum())
            row[f"{cat} · in sheet"] = int(x["In Sheet"].sum())
            row[f"{cat} · interviewed"] = int(x["Interviewed"].sum())
            row[f"{cat} · rejected"] = int((x["Outcome"] == "Rejected").sum())
            row[f"{cat} · hired"] = int((x["Outcome"] == "Hired / Offer").sum())
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows)
    t = t.sort_values(t.columns[1], ascending=False)
    return pd.concat([t, pd.DataFrame([{"Company": "Total", **t.drop(columns="Company").sum()}])], ignore_index=True)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def _pick(label, df, default, key):
    options = [NONE] + list(df.columns)
    v = st.selectbox(label, options, index=options.index(default) if default in options else 0, key=key)
    return None if v == NONE else v


def render(pools, tr_sum, tr_rule, academy_kw, intensive_kw):
    st.title("🏢 All Companies — Applications to Outcomes")
    st.caption("Reads every company sheet in the tracker (sheets must be shared as *Anyone with the link → Viewer*), "
               "adds the all-company applications export, matches everyone to the placement pools, and counts "
               "applied → in company sheet → interviewed → rejected / hired.")

    i1, i2 = st.columns(2)
    with i1:
        tracker = data_input("1 · Tracker (Company + Company Sheet Link + Job ID)", "mc_tracker")
    with i2:
        apps = data_input("2 · Optional · All applications (Company Name, Job ID, Phone, User Product…)", "mc_apps")
    if tracker is None:
        st.info("Add the tracker: upload it as CSV, or paste its cells / link.")
        st.stop()

    # ---------------- columns ----------------
    tcols = list(tracker.columns)
    comp_default = core.find_col(tracker, ["company"], exclude=("link", "sheet", "id")) or tcols[0]
    link_default = core.find_col(tracker, ["companysheetlink", "companysheet"])
    job_default = core.find_col(tracker, ["jobid", "jobiduuid"])
    jd_default = core.find_col(tracker, ["jdlabel", "jdname", "jd"], exclude=("link", "date", "id"))
    role_default = core.find_col(tracker, ["techstack", "jobrole", "role", "jobtitle", "designation", "position", "stack"],
                                 exclude=("link", "date", "id"))
    name_cols_default = [c for c in tcols if "company" in core.norm_col(c)
                         and not any(x in core.norm_col(c) for x in ("link", "sheet", "id"))]
    a_det = detect_app_columns(apps) if apps is not None else {}
    need_map = apps is not None and not (a_det["mobile"] and (a_det["job"] or a_det["company"]))
    with st.expander("Columns", expanded=need_map or not link_default):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Tracker**")
            comp_col = st.selectbox("Company (name shown in reports)", tcols, index=tcols.index(comp_default), key="mc_comp")
            link_col = _pick("Company Sheet Link", tracker, link_default, "mc_link")
            job_col = _pick("Job ID (links applications to the company)", tracker, job_default, "mc_job")
            jd_col = _pick("JD label (JD-1, JD-2…)", tracker, jd_default, "mc_jd")
            role_col = _pick("Tech stack / role (optional, used in the JD name)", tracker, role_default, "mc_role")
            name_cols = st.multiselect("Other company-name columns (to match applications by name)", tcols,
                                       default=name_cols_default, key="mc_names")
        a_cols = {}
        with c2:
            st.markdown("**Applications**")
            if apps is None:
                st.caption("Not added.")
            else:
                for k, lbl in (("company", "Company Name"), ("job", "Job ID"), ("name", "Name"), ("mobile", "Phone / Mobile"),
                               ("email", "Email"), ("uid", "User ID"), ("product", "Product"), ("resume", "Resume")):
                    a_cols[k] = _pick(lbl, apps, a_det.get(k), f"mc_a_{k}")
    all_tabs = st.toggle("Read every tab of each sheet", value=True, key="mc_alltabs",
                         help="Off = read only the tab the link points to (its gid).")
    if not link_col:
        st.warning("Pick the column that holds the company sheet links.")
        st.stop()
    if apps is not None and not (a_cols.get("mobile") or a_cols.get("email") or a_cols.get("uid")):
        st.error("Map a Phone / Email / User ID column for the applications file.")
        st.stop()

    tracker = tracker.copy()
    tracker[comp_col] = tracker[comp_col].fillna("").astype(str).str.strip()
    # One name per JD row: label, else tech stack, else short Job ID
    _s = lambda c: tracker[c].fillna("").astype(str).str.strip() if c else pd.Series("", index=tracker.index)  # noqa: E731
    jd_name = _s(jd_col)
    role = _s(role_col)
    jd_name = jd_name.where(jd_name != "", role)
    single = tracker.groupby(comp_col)[comp_col].transform("size") == 1
    jd_name = jd_name.where((jd_name != "") | ~single, "Only JD")
    jd_name = jd_name.where(jd_name != "", "Job " + _s(job_col).str[:8])
    tracker["_JD"] = jd_name.where(jd_name != "", "Main")
    if role_col:  # 'JD-1 · Java' when both exist
        both = (_s(jd_col) != "") & (role != "")
        tracker.loc[both, "_JD"] = _s(jd_col)[both] + " · " + role[both]
    # Every Job ID is its own JD: if two Job IDs of a company end up with the same name, add the ID
    jid = _s(job_col)
    dup = tracker.assign(_j=jid).groupby([comp_col, "_JD"])["_j"].transform("nunique") > 1
    tracker.loc[dup, "_JD"] = tracker.loc[dup, "_JD"] + " (" + jid[dup].str[:8] + ")"
    tracker["_jobid"] = jid
    tracker["_role"] = role.where(role != "")
    tr = tracker[[comp_col, link_col]].copy()
    tr[link_col] = tr[link_col].fillna("").astype(str).str.strip()
    no_link = tr[~tr[link_col].str.contains("docs.google.com/spreadsheets", regex=False)]
    tr = tr.drop(no_link.index)
    tr["_key"] = [sheet_id(l) if all_tabs else f"{sheet_id(l)}#{link_gid(l)}" for l in tr[link_col]]
    tr = tr.drop_duplicates([comp_col, "_key"])

    app_rows = None
    if apps is not None:
        app_rows = prepare_apps(apps, a_cols, tracker, comp_col, job_col, name_cols, pools, tr_sum,
                                academy_kw, intensive_kw)

    companies = sorted(set(tr[comp_col]) | (set(app_rows["Company"]) if app_rows is not None else set()))
    if st.checkbox(f"All companies ({len(companies)})", value=True, key="mc_all"):
        sel = companies
    else:
        sel = st.multiselect("Companies", companies, key="mc_sel")
    b1, b2 = st.columns([1, 5])
    if b1.button("▶ Fetch & analyse", type="primary"):
        st.session_state["mc_go"] = True
    if b2.button("↻ Refresh sheets", help="Sheets are cached for 10 minutes"):
        fetch_all.clear()
    if not st.session_state.get("mc_go"):
        st.caption(f"{len(tr)} sheet(s) to read · {len(companies)} companies"
                   + (f" · {len(app_rows):,} applications" if app_rows is not None else "")
                   + (f" · {len(no_link)} tracker row(s) have no sheet link" if len(no_link) else ""))
        st.stop()

    jobs = tuple((c, l, all_tabs) for c, l in zip(tr[comp_col], tr[link_col]) if c in sel)
    with st.spinner(f"Reading {len(jobs)} sheet(s)…"):
        tabs = fetch_all(jobs)

    # ---------------- analyse sheets ----------------
    frames, log, others = [], [], []
    for t in tabs:
        entry = {"Company": t["Company"], "Tab": t["Tab"], "Rows": 0, "Round columns": "", "Status": "",
                 "Rows without mobile/email": 0, "Link": t["Link"]}
        if t["df"] is None:
            entry["Status"] = "❌ " + t["Note"]
        else:
            res, info = analyse_tab(t["df"], pools, tr_sum, academy_kw, intensive_kw)
            entry["Rows"] = len(t["df"])
            if res is None:
                entry["Status"] = info
            else:
                entry.update({"Status": "✓ used" + (f" — {t['Note']}" if t["Note"] else ""),
                              "Round columns": info["round_cols"], "Rows without mobile/email": info["no_id"]})
                frames.append(res.assign(Company=t["Company"], Tab=t["Tab"], Applied=False, **{"In Sheet": True}))
                others += [{"Company": t["Company"], "Tab": t["Tab"], "Value": v} for v in info["other"]]
        log.append(entry)
    log = pd.DataFrame(log)
    sheet_companies = {f["Company"].iat[0] for f in frames if len(f)}

    parts = frames + ([app_rows[app_rows["Company"].isin(sel)]] if app_rows is not None else [])
    parts = [p for p in parts if len(p)]
    if not parts:
        st.error("Nothing to show — no candidate tabs could be read and no applications were added.")
        st.dataframe(log, hide_index=True, width="stretch")
        st.stop()
    has_apps = app_rows is not None
    apps_sel = app_rows[app_rows["Company"].isin(sel)] if has_apps else None

    # ---------------- JD: Job ID -> student -> company sheet ----------------
    if frames:
        sheet_rows = assign_sheet_jd(pd.concat(frames, ignore_index=True), apps_sel)
        parts = [sheet_rows] + ([apps_sel] if has_apps and len(apps_sel) else [])
    all_rows = pd.concat(parts, ignore_index=True)

    level = st.radio("Show stats per", ["JD (Job ID)", "Company (all JDs combined)"], horizontal=True, key="mc_level")
    per_jd = level.startswith("JD")
    cand_c = merge_people(all_rows, has_apps)  # company level, for the headline
    cand_jd = merge_people(all_rows, has_apps, per_jd=True)
    cand = cand_jd if per_jd else cand_c
    if per_jd:  # tables group by this label
        tv = cand.assign(Company=cand["Company"] + UNIT_SEP + cand["JD"])
        sheet_units = {u for u, c in zip(tv["Company"], cand["Company"]) if c in sheet_companies}
    else:
        tv, sheet_units = cand, sheet_companies
    job_ids = {(c, j): i for c, j, i in zip(tracker[comp_col], tracker["_JD"], tracker["_jobid"]) if i}
    masks = pool_masks(cand, pools["bucket_values"])
    mc = pool_masks(cand_c, pools["bucket_values"])
    in_pool = mc["From placement pool (bucket or assessment)"]

    # ---------------- headline ----------------
    n = lambda m: int(m.sum())  # noqa: E731
    st.subheader("Overall")
    if has_apps:
        r = st.columns(5)
        r[0].metric("Total applications", f"{int(cand_c['Applications'].sum()):,}")
        r[1].metric("Unique applicants (per company)", f"{n(cand_c['Applied']):,}")
        r[2].metric("Applicants from placement pool", f"{n(cand_c['Applied'] & in_pool):,}")
        r[3].metric("Applied & in company sheet", f"{n(cand_c['Applied'] & cand_c['In Sheet']):,}")
        r[4].metric("Companies", cand_c["Company"].nunique())
    r = st.columns(6)
    r[0].metric("Company sheets read", len(sheet_companies))
    r[1].metric("Candidates in sheets", f"{n(cand_c['In Sheet']):,}")
    r[2].metric("Sheet · from pool", n(cand_c["In Sheet"] & in_pool))
    r[3].metric("Pool · interviewed", n(in_pool & cand_c["Interviewed"]))
    r[4].metric("Pool · rejected", n(in_pool & (cand_c["Outcome"] == "Rejected")))
    r[5].metric("Pool · hired / offer", n(in_pool & (cand_c["Outcome"] == "Hired / Offer")))
    st.caption("Headline counts are per company: someone who applied twice to the same company, or appears in several "
               "tabs, is counted once (best outcome kept). In the per-JD tables a person is counted once per JD. "
               f"TR Cleared rule: {tr_rule}.")
    if per_jd:
        u = int((cand["In Sheet"] & (cand["JD"] == "Not in applications")).sum())
        if u:
            st.caption(f"{u} company-sheet candidate(s) have no application, so they have no Job ID and are shown "
                       "under 'Not in applications'.")

    # ---------------- phone search ----------------
    q = st.text_input("🔎 Search a student by phone number", key="mc_phone",
                      placeholder="e.g. 9876543210 (with or without +91 / spaces)")
    if q.strip():
        mob = core.norm_mobile(q)
        if not mob:
            st.warning("That doesn't look like a phone number.")
        else:
            hit = cand_jd[cand_jd["Mobile"].fillna("") == mob]
            if hit.empty:
                st.info(f"No application or company-sheet row found for {mob}.")
            else:
                h0 = hit.iloc[0]
                pool = (h0["Bucket"] if isinstance(h0["Bucket"], str) and h0["Bucket"] else "Not in bucket") + \
                    (" · Assessment pool" if hit["In Assessment Pool"].any() else "") + \
                    (" · TR Cleared" if hit["TR Cleared"].any() else "")
                st.markdown(f"**{h0['Name'] or '—'}** · {mob} · {h0['Category'] or '—'} · {pool} · "
                            f"applied to **{hit.loc[hit['Applied'], 'Company'].nunique()}** compan(ies), "
                            f"**{int(hit['Applied'].sum())}** Job ID(s)")
                res = pd.DataFrame({
                    "Company": hit["Company"],
                    "Job ID": [job_ids.get((c, j)) or (r if isinstance(r, str) else "") for c, j, r in
                               zip(hit["Company"], hit["JD"], hit.get("Job ID", pd.Series(None, index=hit.index)))],
                    "Applied": hit["Applied"], "In company sheet": hit["In Sheet"], "Outcome": hit["Outcome"],
                    "Rounds": hit["Rounds"],
                }).sort_values(["Applied", "Company"], ascending=[False, True])
                st.dataframe(res, hide_index=True, width="stretch")

    tab_c, tab_b, tab_p, tab_x, tab_s, tab_l, tab_log = st.tabs(
        ["By company", "By bucket & TR (per Job ID)", "By product", "Pool vs all", "Students (all companies)",
         "Candidates", "Sheets read & checks"])

    with tab_b:
        jm = pool_masks(cand_jd, pools["bucket_values"])
        groups = {k: v for k, v in jm.items()
                  if k in [b if "bucket" in b.lower() else f"Bucket {b}" for b in pools["bucket_values"]]}
        groups["TR Cleared"] = jm["TR Cleared"]
        jp = job_pool_wide(job_pool_table(cand_jd, job_ids, groups), list(groups))
        b1, b2 = st.columns(2)
        show_m = b1.multiselect("Main columns", JOB_POOL_COLS + JOB_POOL_PCT, default=JOB_POOL_COLS + JOB_POOL_PCT,
                                key="mc_b_metrics")
        show_g = b2.multiselect("Sub-columns", list(groups), default=list(groups), key="mc_b_groups")
        view = jp[[(m, g) for m in show_m for g in show_g]] if show_m and show_g else jp.iloc[:, :0]
        st.markdown(grouped_html(view), unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        xl = to_excel_bytes(jp)
        if xl:
            d1.download_button("⬇️ Download Excel (merged headers)", xl, file_name="job_id_by_bucket_and_tr.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="mc_b_xl")
        else:
            d1.caption("For the Excel download run: pip install openpyxl")
        d2.download_button("⬇️ Download CSV (2 header rows)", jp.to_csv().encode("utf-8"),
                           file_name="job_id_by_bucket_and_tr.csv", mime="text/csv", key="mc_b_dl")
        st.caption("Same funnel per Job ID, counted only for students in that bucket / TR Cleared. "
                   "Sent to Company = applied to that Job ID and present in the company sheet; outcomes are counted "
                   "among those. A TR Cleared student is also counted in their bucket.")

    with tab_s:
        stu = student_table(cand_jd, job_ids)
        sm = pool_masks(stu, pools["bucket_values"])
        g1, g2, g3 = st.columns(3)
        s_group = g1.selectbox("Pool group", list(sm), key="mc_s_group")
        s_cats = [x for x in CAT_ORDER if x in set(stu["Category"])]
        s_cat = g2.multiselect("Product", s_cats, default=s_cats, key="mc_s_cat")
        only_app = g3.checkbox("Only students who applied", value=has_apps, key="mc_s_only", disabled=not has_apps)
        sv = stu[sm[s_group] & stu["Category"].isin(s_cat)]
        if only_app:
            sv = sv[sv["Total applications"] > 0]
        k = st.columns(5)
        k[0].metric("Unique students", f"{len(sv):,}")
        k[1].metric("Total applications", f"{int(sv['Total applications'].sum()):,}")
        k[2].metric("Avg applications / student", f"{sv['Total applications'].mean():.1f}" if len(sv) else "0")
        k[3].metric("Interviewed at least once", f"{int((sv['Interviewed (JDs)'] > 0).sum()):,}")
        k[4].metric("Hired / offer at least once", f"{int((sv['Hired / Offer'] > 0).sum()):,}")
        st.dataframe(sv, hide_index=True, width="stretch")
        st.download_button("⬇️ Download students CSV", sv.to_csv(index=False).encode("utf-8"),
                           file_name="students_all_companies.csv", mime="text/csv", key="mc_s_dl")
        st.caption("One row per student (by mobile, else email) across all companies. Total applications = rows in "
                   "the applications file; JDs applied = distinct Job IDs; Interviewed / Rejected / Hired are counted "
                   "per JD.")

    with tab_c:
        f1, f2 = st.columns(2)
        group = f1.selectbox("Pool group", list(masks), index=0, key="mc_group")
        cats = [x for x in CAT_ORDER if x in set(cand["Category"])]
        sel_cat = f2.multiselect("Product", cats, default=cats, key="mc_cat")
        tbl = company_table(tv, masks[group] & cand["Category"].isin(sel_cat), has_apps, sheet_units)
        if per_jd:
            tbl = split_units(tbl, job_ids)
        st.dataframe(tbl, hide_index=True, width="stretch")
        st.download_button("⬇️ Download this table (CSV)", tbl.to_csv(index=False).encode("utf-8"),
                           file_name=f"companies_{re.sub(r'[^A-Za-z0-9]+', '_', group)}.csv", mime="text/csv")
        st.caption("'Sheet read = No' means the company has applications but its sheet wasn't read "
                   "(no link, not public, or no candidate tab), so its sheet columns are 0.")

    with tab_p:
        group_p = st.selectbox("Pool group", list(masks), index=0, key="mc_group_p")
        pt = product_table(tv, masks[group_p])
        if per_jd:
            pt = split_units(pt, job_ids)
        st.dataframe(pt, hide_index=True, width="stretch")
        if len(pt):
            st.download_button("⬇️ Download (CSV)", pt.to_csv(index=False).encode("utf-8"),
                               file_name="companies_by_product.csv", mime="text/csv")

    with tab_x:
        st.caption("Per company: everyone vs. students from the placement pool.")
        rows = []
        for comp, g in tv.groupby("Company"):
            p = g["In Bucket"] | g["In Assessment Pool"]
            row = {"Company": comp}
            if has_apps:
                row.update({"All · applied": n(g["Applied"]), "Pool · applied": n(p & g["Applied"])})
            row.update({
                "All · in sheet": n(g["In Sheet"]), "All · interviewed": n(g["Interviewed"]),
                "All · rejected": n(g["Outcome"] == "Rejected"), "All · hired": n(g["Outcome"] == "Hired / Offer"),
                "Pool · in sheet": n(p & g["In Sheet"]), "Pool · interviewed": n(p & g["Interviewed"]),
                "Pool · rejected": n(p & (g["Outcome"] == "Rejected")),
                "Pool · hired": n(p & (g["Outcome"] == "Hired / Offer")),
            })
            for b in pools["bucket_values"]:
                row[f"{b} · total"] = n(g["Bucket"] == b)
            row["Assessment pool · total"] = n(g["In Assessment Pool"])
            row["TR Cleared · total"] = n(g["TR Cleared"])
            rows.append(row)
        wide = pd.DataFrame(rows)
        wide = wide.sort_values(wide.columns[1], ascending=False)
        wide = pd.concat([wide, pd.DataFrame([{"Company": "Total", **wide.drop(columns="Company").sum()}])],
                         ignore_index=True)
        if per_jd:
            wide = split_units(wide, job_ids)
        st.dataframe(wide, hide_index=True, width="stretch")
        st.download_button("⬇️ Download (CSV)", wide.to_csv(index=False).encode("utf-8"),
                           file_name="companies_pool_vs_all.csv", mime="text/csv")

    with tab_l:
        f1, f2, f3, f4, f5 = st.columns(5)
        s_comp = f1.multiselect("Company", sorted(cand["Company"].unique()), key="mc_fcomp")
        s_group = f2.selectbox("Pool group", list(masks), key="mc_fgroup")
        cats = [x for x in CAT_ORDER if x in set(cand["Category"])]
        s_cat = f3.multiselect("Product", cats, default=cats, key="mc_fcat")
        s_out = f4.multiselect("Outcome", OUTCOMES, default=OUTCOMES, key="mc_fout")
        search = f5.text_input("Search name / mobile", key="mc_search")
        v = cand[masks[s_group] & cand["Outcome"].isin(s_out) & cand["Category"].isin(s_cat)]
        if s_comp:
            v = v[v["Company"].isin(s_comp)]
        if search:
            s = search.strip().lower()
            v = v[v["Name"].fillna("").str.lower().str.contains(s, regex=False)
                  | v["Mobile"].fillna("").str.contains(s, regex=False)]
        v = v.assign(**{"Job ID": [job_ids.get((c, j), "") for c, j in zip(v["Company"], v["JD"])]}) if per_jd \
            else v.assign(**{"Job ID": ""})
        cols = ["Company", "JD", "Job ID", "JD source", "Name", "Mobile", "Email", "Product", "Category", "Applied", "In Sheet", "Tab",
                "Bucket", "Cycle", "In Assessment Pool", "TR Cleared", "TR1 Status", "TR2 Status",
                "Interviewed", "Outcome", "Rounds", "Resume"]
        st.caption(f"{len(v):,} candidates")
        st.dataframe(v[cols], hide_index=True, width="stretch",
                     column_config={"Resume": st.column_config.LinkColumn("Resume", display_text="Open")})
        st.download_button("⬇️ Download these candidates (CSV)", v[cols].to_csv(index=False).encode("utf-8"),
                           file_name="company_candidates.csv", mime="text/csv")

    with tab_log:
        st.dataframe(log, hide_index=True, width="stretch",
                     column_config={"Link": st.column_config.LinkColumn("Link", display_text="Open")})
        if len(no_link):
            st.markdown("**Tracker rows without a company sheet link**")
            st.dataframe(tracker.loc[no_link.index], hide_index=True, width="stretch")
        if has_apps:
            unmatched = sorted(set(app_rows["Company"]) - set(tracker[comp_col]))
            if unmatched:
                st.markdown(f"**{len(unmatched)} application company name(s) not found in the tracker** "
                            "(no Job ID / name match — shown under their own name, no sheet)")
                st.write(", ".join(unmatched))
            dropped = len(apps) - len(app_rows)
            if dropped:
                st.caption(f"{dropped} application row(s) had no company and were left out.")
        if others:
            st.markdown("**Status values not recognised** (not counted as interviewed)")
            st.dataframe(pd.DataFrame(others), hide_index=True, width="stretch")
        st.caption(
            "Applied = in the applications file. In company sheet = listed in the company's sheet. "
            "Interviewed = any round marked Cleared / Selected / Rejected / On hold. Rejected = any round marked "
            "Rejected / Not cleared. Hired / Offer = a cleared status containing Hired / Offer / Joined. "
            "Applications and sheet rows are linked by mobile (last 10 digits). Pool matching uses User ID, then "
            "mobile, then email. Product: Intensive / Academy from the sidebar keywords, anything else External."
        )