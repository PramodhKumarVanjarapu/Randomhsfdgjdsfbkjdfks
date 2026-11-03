"""Matching and statistics logic for the Company Applicant Stats app.

Kept separate from the Streamlit UI so it can be tested on its own.
"""
from __future__ import annotations

import io
import re

import pandas as pd

JUNK_VALUES = {"", "nan", "none", "null", "-", "#ref!", "#n/a", "na", "n/a"}


# --------------------------------------------------------------------------- #
# Normalisers
# --------------------------------------------------------------------------- #
def norm_col(c) -> str:
    """'Student \\nMobile Number' -> 'studentmobilenumber'."""
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def _is_junk(x) -> bool:
    return pd.isna(x) or str(x).strip().lower() in JUNK_VALUES


def norm_mobile(x):
    """Keep the last 10 digits (drops +91 / 91 / leading 0 / '.0' from Excel)."""
    if _is_junk(x):
        return None
    s = str(x).strip()
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    digits = re.sub(r"\D", "", s)
    return digits[-10:] if len(digits) >= 10 else None


def norm_email(x):
    if _is_junk(x):
        return None
    s = str(x).strip().lower()
    return s if "@" in s else None


def norm_text(x):
    if _is_junk(x):
        return None
    return re.sub(r"\s+", " ", str(x).strip().lower())


def norm_status(x) -> str:
    """'Not cleared' / 'Not Cleared ' -> 'not cleared'; blanks/'-' -> ''."""
    if _is_junk(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip().lower())


def normaliser_for(col_name: str):
    n = norm_col(col_name)
    if any(k in n for k in ("mobile", "phone", "contact")):
        return norm_mobile
    if "mail" in n:
        return norm_email
    return norm_text


# --------------------------------------------------------------------------- #
# IO + column detection
# --------------------------------------------------------------------------- #
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), dtype=str, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df.columns = [str(c).strip() for c in df.columns]
    return df.dropna(how="all")


def find_col(df: pd.DataFrame, keywords, exclude=()):
    """First column whose normalised name equals, then contains, a keyword."""
    cols = {c: norm_col(c) for c in df.columns}
    cols = {c: n for c, n in cols.items() if not any(e in n for e in exclude)}
    for kw in keywords:
        for c, n in cols.items():
            if n == kw:
                return c
    for kw in keywords:
        for c, n in cols.items():
            if kw in n:
                return c
    return None


UID_KW = ["userid", "uid", "studentid", "candidateid", "applicantid"]
EMAIL_KW = ["email", "emailid", "mail"]
MOBILE_KW = ["mobilenumber", "mobile", "phonenumber", "phone", "contactnumber"]
NAME_KW = ["studentname", "candidatename", "fullname", "name"]
PRODUCT_KW = ["product", "productname", "producttype"]
RESUME_KW = ["resumelink", "resume", "cv"]


def detect_columns(df: pd.DataFrame) -> dict:
    return {
        "uid": find_col(df, UID_KW),
        "email": find_col(df, EMAIL_KW),
        "mobile": find_col(df, MOBILE_KW, exclude=("mail",)),
        "name": find_col(df, NAME_KW, exclude=("company", "college", "mail")),
        "product": find_col(df, PRODUCT_KW),
        "resume": find_col(df, RESUME_KW),
    }


# --------------------------------------------------------------------------- #
# Product -> category
# --------------------------------------------------------------------------- #
def categorize_product(p, academy_kw, intensive_kw) -> str:
    if _is_junk(p):
        return "External"
    up = str(p).upper()
    if any(k and k in up for k in intensive_kw):
        return "Intensive"
    if any(k and k in up for k in academy_kw):
        return "Academy"
    return "External"


# --------------------------------------------------------------------------- #
# Static pools
# --------------------------------------------------------------------------- #
def build_pools(bucket_df: pd.DataFrame, tr_df: pd.DataFrame) -> dict:
    """Index the two static files for fast lookup by user id / mobile / email."""
    # ---- Bucket (offline attended pool) ----
    b = bucket_df.copy()
    bc = {
        "uid": find_col(b, UID_KW),
        "mobile": find_col(b, MOBILE_KW, exclude=("mail",)),
        "email": find_col(b, EMAIL_KW),
        "bucket": find_col(b, ["bucket"]),
        "cycle": find_col(b, ["cycle", "offlinecycle"]),
    }
    if not bc["bucket"]:
        raise ValueError("Bucket file has no 'Bucket' column.")
    b["_uid"] = b[bc["uid"]].map(norm_text) if bc["uid"] else None
    b["_mob"] = b[bc["mobile"]].map(norm_mobile) if bc["mobile"] else None
    b["_email"] = b[bc["email"]].map(norm_email) if bc["email"] else None
    b["_bucket"] = b[bc["bucket"]].fillna("").str.strip()
    b["_cycle"] = b[bc["cycle"]].fillna("").str.strip() if bc["cycle"] else ""
    b = b.reset_index(drop=True)

    def index_by(df, key):
        s = df[key].dropna()
        return dict(zip(s.values, s.index))  # last wins; keys are unique in practice

    bucket_idx = {k: index_by(b, f"_{k}") for k in ("uid", "mob", "email")}

    # ---- TR / assessment pool (main) ----
    t = tr_df.copy()
    tc = {
        "uid": find_col(t, UID_KW),
        "mobile": find_col(t, MOBILE_KW, exclude=("mail",)),
        "tr1": find_col(t, ["tr1status"]),
        "tr2": find_col(t, ["tr2status"]),
        "role": find_col(t, ["eligiblerole"]),
    }
    if not tc["tr1"] and not tc["tr2"]:
        raise ValueError("Assessment pool file has no 'TR1 Status' / 'TR2 Status' columns.")
    t["_uid"] = t[tc["uid"]].map(norm_text) if tc["uid"] else None
    t["_mob"] = t[tc["mobile"]].map(norm_mobile) if tc["mobile"] else None
    t["_tr1"] = t[tc["tr1"]].map(norm_status) if tc["tr1"] else ""
    t["_tr2"] = t[tc["tr2"]].map(norm_status) if tc["tr2"] else ""
    t["_role"] = t[tc["role"]].fillna("").str.strip() if tc["role"] else ""
    # A student can have several rows (one per eligible role) -> group key
    t["_key"] = [
        u if isinstance(u, str) else (f"m:{m}" if isinstance(m, str) else f"row{i}")
        for i, u, m in zip(t.index, t["_uid"], t["_mob"])
    ]
    t = t.reset_index(drop=True)

    tr_by_uid = t.dropna(subset=["_uid"]).groupby("_uid")["_key"].first().to_dict()
    tr_by_mob = t.dropna(subset=["_mob"]).groupby("_mob")["_key"].first().to_dict()
    # Mobile can differ between the two static files; bridge through user id.
    for mob, uid in zip(b["_mob"], b["_uid"]):
        if mob and uid and uid in tr_by_uid and mob not in tr_by_mob:
            tr_by_mob[mob] = tr_by_uid[uid]

    # ...and the other way: a mobile only present in the assessment file still finds the bucket row.
    for mob, uid in zip(t["_mob"], t["_uid"]):
        if isinstance(mob, str) and isinstance(uid, str) and uid in bucket_idx["uid"] and mob not in bucket_idx["mob"]:
            bucket_idx["mob"][mob] = bucket_idx["uid"][uid]

    statuses = sorted(
        {s for s in pd.concat([t["_tr1"], t["_tr2"]]).unique() if s},
        key=lambda s: (s != "cleared", s),
    )

    return {
        "bucket": b,
        "bucket_idx": bucket_idx,
        "tr": t,
        "tr_by_uid": tr_by_uid,
        "tr_by_mob": tr_by_mob,
        "statuses": statuses,
        "bucket_values": sorted(v for v in b["_bucket"].unique() if v),
    }


TR_RULES = [
    "TR1 Cleared OR TR2 Cleared",
    "TR1 Cleared AND TR2 Cleared",
    "TR1 Cleared",
    "TR2 Cleared",
]


def tr_summary(pools: dict, rule: str, cleared_values) -> pd.DataFrame:
    """One row per assessment-pool student, with the TR Cleared flag for `rule`."""
    t = pools["tr"].copy()
    ok = set(cleared_values)
    c1, c2 = t["_tr1"].isin(ok), t["_tr2"].isin(ok)
    t["_cleared"] = {
        TR_RULES[0]: c1 | c2,
        TR_RULES[1]: c1 & c2,
        TR_RULES[2]: c1,
        TR_RULES[3]: c2,
    }[rule]

    def join(s):
        vals = [v.title() for v in s if v]
        return " | ".join(dict.fromkeys(vals))

    return t.groupby("_key").agg(
        tr_cleared=("_cleared", "any"),
        tr1=("_tr1", join),
        tr2=("_tr2", join),
        roles=("_role", lambda s: " | ".join(dict.fromkeys(v for v in s if v))),
    )


# --------------------------------------------------------------------------- #
# Company processing
# --------------------------------------------------------------------------- #
def prepare_company(
    apps: pd.DataFrame,
    app_cols: dict,
    contacts: pd.DataFrame | None,
    contact_cols: dict,
    join_app_col: str | None,
    join_contact_col: str | None,
    academy_kw,
    intensive_kw,
) -> pd.DataFrame:
    """One row per candidate in the MOBILE + RESUME file (the base for all stats).

    Product / name / user id / email are looked up from the applications file.
    Candidates with no matching application (or a blank Product) become External.
    Falls back to the applications file as the base if no mobile file is given.
    """
    a = apps.copy().reset_index(drop=True)
    linked = contacts is not None and join_app_col and join_contact_col
    base = contacts.copy().reset_index(drop=True) if linked else a

    out = pd.DataFrame(index=base.index)
    for k in ("Name", "Product", "Resume", "_uid", "_email", "_mob"):
        out[k] = pd.Series(None, index=base.index, dtype=object)

    if linked:
        fa, fc = normaliser_for(join_app_col), normaliser_for(join_contact_col)
        f = norm_mobile if norm_mobile in (fa, fc) else (norm_email if norm_email in (fa, fc) else norm_text)
        cname = find_col(base, NAME_KW, exclude=("company", "college", "mail"))

        # Applications indexed by the join key; if a candidate applied more than once,
        # keep the row with the strongest product (Intensive > Academy > External).
        a["_join"] = a[join_app_col].map(f)
        a["_cat"] = a[app_cols["product"]].map(lambda p: categorize_product(p, academy_kw, intensive_kw)) if app_cols.get("product") else "External"
        a["_rank"] = a["_cat"].map({"Intensive": 0, "Academy": 1}).fillna(2)
        a = a.dropna(subset=["_join"]).sort_values("_rank").drop_duplicates("_join", keep="first").set_index("_join")

        key = base[join_contact_col].map(f)
        out["_app_found"] = key.isin(a.index)
        pick = lambda col: key.map(a[col]).astype(object) if col else pd.Series(None, index=base.index, dtype=object)  # noqa: E731

        if contact_cols.get("mobile"):
            out["_mob"] = base[contact_cols["mobile"]].map(norm_mobile).astype(object)
        if app_cols.get("mobile"):
            out["_mob"] = out["_mob"].combine_first(pick(app_cols["mobile"]).map(norm_mobile))
        if contact_cols.get("resume"):
            out["Resume"] = base[contact_cols["resume"]].astype(object)
        out["Name"] = (base[cname].astype(object) if cname else out["Name"]).combine_first(pick(app_cols.get("name")))
        out["Product"] = pick(app_cols.get("product"))
        if app_cols.get("uid"):
            out["_uid"] = pick(app_cols["uid"]).map(norm_text)
        if app_cols.get("email"):
            out["_email"] = pick(app_cols["email"]).map(norm_email)
    else:
        out["_app_found"] = True
        out["Name"] = a[app_cols["name"]].astype(object) if app_cols.get("name") else out["Name"]
        out["Product"] = a[app_cols["product"]].astype(object) if app_cols.get("product") else out["Product"]
        if app_cols.get("uid"):
            out["_uid"] = a[app_cols["uid"]].map(norm_text).astype(object)
        if app_cols.get("email"):
            out["_email"] = a[app_cols["email"]].map(norm_email).astype(object)
        if app_cols.get("mobile"):
            out["_mob"] = a[app_cols["mobile"]].map(norm_mobile).astype(object)

    out["Mobile"] = out["_mob"]
    out["Category"] = out["Product"].map(lambda p: categorize_product(p, academy_kw, intensive_kw))
    rowkey = pd.Series("row" + out.index.astype(str), index=out.index)
    out["_person"] = (
        out["_uid"]
        .combine_first(out["_mob"].map(lambda v: f"m:{v}" if isinstance(v, str) else None))
        .combine_first(out["_email"].map(lambda v: f"e:{v}" if isinstance(v, str) else None))
        .combine_first(rowkey)
    )
    return out



def match_company(prepared: pd.DataFrame, pools: dict, tr_sum: pd.DataFrame) -> pd.DataFrame:
    """Attach bucket + TR info to each application row."""
    df = prepared.copy()
    b, idx = pools["bucket"], pools["bucket_idx"]

    def bucket_row(r):
        for key, col, label in (("uid", "_uid", "User ID"), ("mob", "_mob", "Mobile"), ("email", "_email", "Email")):
            v = r[col]
            if isinstance(v, str) and v in idx[key]:
                return idx[key][v], label
        return None, None

    hits = df.apply(bucket_row, axis=1, result_type="expand")
    df["_b_row"], df["Bucket Match By"] = hits[0], hits[1]
    has_b = df["_b_row"].notna()
    df["In Bucket"] = has_b
    df["Bucket"] = None
    df["Cycle"] = None
    df.loc[has_b, "Bucket"] = b.loc[df.loc[has_b, "_b_row"].astype(int), "_bucket"].values
    df.loc[has_b, "Cycle"] = b.loc[df.loc[has_b, "_b_row"].astype(int), "_cycle"].values
    # Fill a missing mobile from the pool when matched by user id / email
    pool_mob = pd.Series(None, index=df.index, dtype=object)
    pool_mob[has_b] = b.loc[df.loc[has_b, "_b_row"].astype(int), "_mob"].values
    df["Mobile"] = df["Mobile"].combine_first(pool_mob)
    pool_uid = pd.Series(None, index=df.index, dtype=object)
    pool_uid[has_b] = b.loc[df.loc[has_b, "_b_row"].astype(int), "_uid"].values

    def tr_key(i, r):
        for uid in (r["_uid"], pool_uid[i]):
            if isinstance(uid, str) and uid in pools["tr_by_uid"]:
                return pools["tr_by_uid"][uid]
        m = r["_mob"]
        if isinstance(m, str) and m in pools["tr_by_mob"]:
            return pools["tr_by_mob"][m]
        return None

    df["_tr_key"] = [tr_key(i, r) for i, r in df.iterrows()]
    keys = [k if isinstance(k, str) and k in tr_sum.index else None for k in df["_tr_key"]]
    df["In Assessment Pool"] = [k is not None for k in keys]
    df["TR Cleared"] = [bool(tr_sum.at[k, "tr_cleared"]) if k else False for k in keys]
    df["TR1 Status"] = [tr_sum.at[k, "tr1"] if k else "" for k in keys]
    df["TR2 Status"] = [tr_sum.at[k, "tr2"] if k else "" for k in keys]
    df["Eligible Role"] = [tr_sum.at[k, "roles"] if k else "" for k in keys]
    return df


def unique_candidates(matched: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeat applications to one row per person (flags are OR-ed)."""
    agg = {
        "Name": "first",
        "Mobile": "first",
        "Product": lambda s: " | ".join(dict.fromkeys(str(v) for v in s if not _is_junk(v))),
        "Category": lambda s: _pick_category(s),
        "In Bucket": "any",
        "Bucket": "first",
        "Cycle": "first",
        "In Assessment Pool": "any",
        "TR Cleared": "any",
        "TR1 Status": "first",
        "TR2 Status": "first",
        "Eligible Role": "first",
        "Resume": "first",
        "Bucket Match By": "first",
    }
    g = matched.groupby("_person", sort=False).agg(agg)
    g.insert(0, "Applications", matched.groupby("_person", sort=False).size())
    return g.reset_index(drop=True)


def _pick_category(s):
    order = ["Intensive", "Academy", "External", "Unknown"]
    vals = set(s)
    return next((c for c in order if c in vals), "Unknown")
