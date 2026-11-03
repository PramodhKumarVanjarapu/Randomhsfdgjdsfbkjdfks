"""Company Applicant Stats — Streamlit app.

Run:  streamlit run app.py
Static pools are read from ./data/ (can be replaced from the sidebar).
"""
from pathlib import Path

import pandas as pd
import streamlit as st

import v1.core as core
import v3.round_mode as round_mode

st.set_page_config(page_title="Company Applicant Stats", page_icon="📊", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
BUCKET_FILE = DATA_DIR / "offline_attended_pool.csv"
TR_FILE = DATA_DIR / "assessment_pool_main.csv"
NONE = "— none —"


@st.cache_data(show_spinner=False)
def load_csv(data: bytes) -> pd.DataFrame:
    return core.read_csv_bytes(data)


@st.cache_data(show_spinner=False)
def load_pools(bucket_bytes: bytes, tr_bytes: bytes):
    return core.build_pools(load_csv(bucket_bytes), load_csv(tr_bytes))


def file_bytes(uploaded, default_path: Path):
    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name
    if default_path.exists():
        return default_path.read_bytes(), default_path.name
    return None, None


def col_picker(label, df, default, key):
    options = [NONE] + list(df.columns)
    idx = options.index(default) if default in options else 0
    choice = st.selectbox(label, options, index=idx, key=key)
    return None if choice == NONE else choice


def kw_list(text):
    return [k.strip().upper() for k in text.split(",") if k.strip()]


# --------------------------------------------------------------------------- #
# Sidebar: static data + rules
# --------------------------------------------------------------------------- #
with st.sidebar:
    mode = st.radio("Mode", ["Applications + Mobile file", "Round funnel (company sheet)"], key="mode")
    st.header("Static data")
    up_bucket = st.file_uploader("Bucket data (offline attended pool)", type="csv", key="up_bucket")
    up_tr = st.file_uploader("Assessment pool (TR statuses)", type="csv", key="up_tr")
    bucket_bytes, bucket_name = file_bytes(up_bucket, BUCKET_FILE)
    tr_bytes, tr_name = file_bytes(up_tr, TR_FILE)

    if not bucket_bytes or not tr_bytes:
        st.error("Static files missing. Put them in ./data/ or upload them above.")
        st.stop()
    try:
        pools = load_pools(bucket_bytes, tr_bytes)
    except Exception as e:  # noqa: BLE001
        st.error(f"Couldn't read static files: {e}")
        st.stop()

    n_tr_students = pools["tr"]["_key"].nunique()
    st.caption(
        f"Bucket: **{len(pools['bucket']):,}** students · `{bucket_name}`  \n"
        f"Assessment pool: **{n_tr_students:,}** students "
        f"({len(pools['tr']):,} rows) · `{tr_name}`"
    )

    st.header("TR Cleared rule")
    tr_rule = st.radio("Count a student as TR Cleared when", core.TR_RULES, index=0)
    status_labels = {s: s.title() for s in pools["statuses"]}
    cleared_values = st.multiselect(
        "Statuses that count as 'Cleared'",
        options=list(status_labels),
        default=[s for s in status_labels if s == "cleared"],
        format_func=status_labels.get,
    )
    tr_sum = core.tr_summary(pools, tr_rule, cleared_values)
    st.caption(
        f"With this rule, **{int(tr_sum['tr_cleared'].sum())} of {len(tr_sum)}** "
        "assessment-pool students are TR Cleared."
    )

    st.header("Product → category")
    academy_kw = kw_list(st.text_input("Academy if Product contains", "ACADEMY"))
    intensive_kw = kw_list(st.text_input("Intensive if Product contains", "INTENSIVE"))
    st.caption("Anything else is **External**; blank Product is **Unknown**.")


if mode.startswith("Round funnel"):
    round_mode.render(pools, tr_sum, tr_rule, academy_kw, intensive_kw)
    st.stop()

# --------------------------------------------------------------------------- #
# Company uploads
# --------------------------------------------------------------------------- #
st.title("📊 Company Applicant Stats")
company = st.text_input("Company name", placeholder="e.g. Acme Technologies")

c1, c2 = st.columns(2)
with c1:
    up_apps = st.file_uploader("1 · Company applications CSV (with Product)", type="csv", key="apps")
with c2:
    up_contacts = st.file_uploader("2 · Mobile number + resume link CSV", type="csv", key="contacts")

if up_apps is None:
    st.info("Upload the company's applications CSV to see the stats. The mobile/resume CSV is optional but recommended.")
    st.stop()

apps = load_csv(up_apps.getvalue())
contacts = load_csv(up_contacts.getvalue()) if up_contacts is not None else None
det_a = core.detect_columns(apps)
det_c = core.detect_columns(contacts) if contacts is not None else {}

# --------------------------------------------------------------------------- #
# Column mapping
# --------------------------------------------------------------------------- #
needs_attention = not det_a["product"] or not (det_a["mobile"] or det_a["uid"] or det_a["email"])
with st.expander("Column mapping (auto-detected — check once per new file format)", expanded=needs_attention):
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Applications file**")
        app_cols = {
            "product": col_picker("Product", apps, det_a["product"], "a_product"),
            "name": col_picker("Name", apps, det_a["name"], "a_name"),
            "mobile": col_picker("Mobile", apps, det_a["mobile"], "a_mobile"),
            "uid": col_picker("User ID", apps, det_a["uid"], "a_uid"),
            "email": col_picker("Email", apps, det_a["email"], "a_email"),
        }
    contact_cols, join_a, join_c = {}, None, None
    with m2:
        st.markdown("**Mobile + resume file**")
        if contacts is None:
            st.caption("Not uploaded.")
        else:
            contact_cols = {
                "mobile": col_picker("Mobile", contacts, det_c["mobile"], "c_mobile"),
                "resume": col_picker("Resume link", contacts, det_c["resume"], "c_resume"),
            }
            st.markdown("**Link the two files on**")
            # Default join: the first identity column both files have
            default_a, default_c = None, None
            for k in ("uid", "email", "mobile"):
                if det_a.get(k) and det_c.get(k):
                    default_a, default_c = app_cols.get(k) or det_a[k], det_c[k]
                    break
            j1, j2 = st.columns(2)
            with j1:
                join_a = col_picker("Applications column", apps, default_a, "j_a")
            with j2:
                join_c = col_picker("Mobile/resume column", contacts, default_c, "j_c")

if not app_cols["product"]:
    st.warning("Pick the **Product** column above — Intensive / Academy / External depend on it.")
if not (app_cols["mobile"] or app_cols["uid"] or app_cols["email"] or (join_a and contact_cols.get("mobile"))):
    st.error("No way to identify candidates: map a Mobile, User ID or Email column (or link the mobile file).")
    st.stop()
if contacts is not None and not (join_a and join_c):
    st.warning("The mobile/resume file isn't linked to the applications yet — choose the columns to link on.")

# --------------------------------------------------------------------------- #
# Compute
# --------------------------------------------------------------------------- #
prepared = core.prepare_company(apps, app_cols, contacts, contact_cols, join_a, join_c, academy_kw, intensive_kw)
matched = core.match_company(prepared, pools, tr_sum)
cand = core.unique_candidates(matched)

title = f"{company} — " if company else ""
st.subheader(f"{title}Summary")

n = lambda mask: int(mask.sum())  # noqa: E731
cat = cand["Category"]
r1 = st.columns(5)
r1[0].metric("Total candidates (mobile file)", f"{len(matched):,}")
r1[1].metric("Unique candidates", f"{len(cand):,}")
r1[2].metric("Intensive", n(cat == "Intensive"))
r1[3].metric("Academy", n(cat == "Academy"))
r1[4].metric("External", n(cat == "External"))
r2 = st.columns(5)
r2[0].metric("Bucket students", n(cand["In Bucket"]))
r2[1].metric("TR Cleared", n(cand["TR Cleared"]), help=f"Rule: {tr_rule}")
r2[2].metric("In assessment pool", n(cand["In Assessment Pool"]))
r2[3].metric("With resume link", n(cand["Resume"].notna()))
if n(cat == "Unknown"):
    r2[4].metric("Unknown product", n(cat == "Unknown"))

if len(matched) != len(cand):
    st.caption(f"{len(matched) - len(cand)} repeat applications were merged, so category counts are per candidate.")

tab_break, tab_cand, tab_map, tab_quality = st.tabs(["Breakdown", "Candidates", "Product mapping", "Data checks"])

# ---- Breakdown --------------------------------------------------------------
with tab_break:
    order = [c for c in ["Intensive", "Academy", "External", "Unknown"] if c in set(cat)]
    by_cat = (
        cand.groupby("Category")
        .agg(
            Candidates=("Category", "size"),
            **{"Bucket": ("In Bucket", "sum"), "TR Cleared": ("TR Cleared", "sum"),
               "In assessment pool": ("In Assessment Pool", "sum")},
        )
        .reindex(order)
    )
    by_cat.loc["Total"] = by_cat.sum()
    b1, b2 = st.columns([3, 2])
    with b1:
        st.markdown("**By product category**")
        st.dataframe(by_cat.astype(int), width="stretch")
        st.markdown("**Category × bucket**")
        bucket_label = cand["Bucket"].fillna("Not in bucket")
        xt = pd.crosstab(cand["Category"], bucket_label, margins=True, margins_name="Total")
        cols = [c for c in pools["bucket_values"] if c in xt.columns] + [
            c for c in ["Not in bucket", "Total"] if c in xt.columns
        ]
        st.dataframe(xt[cols].reindex(order + ["Total"]), width="stretch")
    with b2:
        st.markdown("**Bucket students by bucket**")
        bc = cand.loc[cand["In Bucket"], "Bucket"].value_counts().reindex(pools["bucket_values"]).fillna(0).astype(int)
        st.bar_chart(bc)
        st.markdown("**TR Cleared × bucket**")
        st.dataframe(
            pd.crosstab(cand["TR Cleared"].map({True: "TR Cleared", False: "Not TR Cleared"}), bucket_label),
            width="stretch",
        )

    summary = pd.DataFrame(
        {
            "Company": [company or ""],
            "Total applications": [len(matched)],
            "Unique candidates": [len(cand)],
            "Intensive": [n(cat == "Intensive")],
            "Academy": [n(cat == "Academy")],
            "External": [n(cat == "External")],
            "Unknown product": [n(cat == "Unknown")],
            "Bucket students": [n(cand["In Bucket"])],
            **{b: [n(cand["Bucket"] == b)] for b in pools["bucket_values"]},
            "TR Cleared": [n(cand["TR Cleared"])],
            "In assessment pool": [n(cand["In Assessment Pool"])],
            "TR rule": [tr_rule],
        }
    )
    st.download_button(
        "⬇️ Download summary CSV",
        summary.to_csv(index=False).encode("utf-8"),
        file_name=f"{(company or 'company').replace(' ', '_')}_summary.csv",
        mime="text/csv",
    )

# ---- Candidates -------------------------------------------------------------
with tab_cand:
    f1, f2, f3, f4 = st.columns([2, 2, 1, 1])
    sel_cat = f1.multiselect("Category", order, default=order)
    bucket_opts = pools["bucket_values"] + ["Not in bucket"]
    sel_bucket = f2.multiselect("Bucket", bucket_opts, default=bucket_opts)
    only_tr = f3.checkbox("TR Cleared only")
    search = f4.text_input("Search name / mobile")

    view = cand[cand["Category"].isin(sel_cat) & cand["Bucket"].fillna("Not in bucket").isin(sel_bucket)]
    if only_tr:
        view = view[view["TR Cleared"]]
    if search:
        s = search.strip().lower()
        view = view[
            view["Name"].fillna("").str.lower().str.contains(s, regex=False)
            | view["Mobile"].fillna("").str.contains(s, regex=False)
        ]

    show_cols = ["Name", "Mobile", "Product", "Category", "Bucket", "Cycle", "TR Cleared",
                 "TR1 Status", "TR2 Status", "Eligible Role", "Resume", "Applications"]
    st.caption(f"{len(view):,} candidates")
    st.dataframe(
        view[show_cols],
        width="stretch",
        hide_index=True,
        column_config={"Resume": st.column_config.LinkColumn("Resume", display_text="Open")},
    )
    st.download_button(
        "⬇️ Download these candidates (CSV)",
        view[show_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"{(company or 'company').replace(' ', '_')}_candidates.csv",
        mime="text/csv",
    )

# ---- Product mapping --------------------------------------------------------
with tab_map:
    st.caption("Every Product value in this file and the category it was put in. Adjust the keywords in the sidebar if any look wrong.")
    pm = (
        matched.assign(Product=matched["Product"].fillna("(blank)"))
        .groupby(["Product", "Category"]).size().rename("Applications").reset_index()
        .sort_values(["Category", "Applications"], ascending=[True, False])
    )
    st.dataframe(pm, width="stretch", hide_index=True)

# ---- Data checks ------------------------------------------------------------
with tab_quality:
    no_mobile = n(cand["Mobile"].isna())
    checks = {
        "Candidates without a usable mobile number": no_mobile,
        "Candidates with no application (counted as External)": (
            n(~prepared["_app_found"]) if contacts is not None and join_a and join_c else "file not linked"
        ),
        "Bucket matches by mobile": n(cand["Bucket Match By"] == "Mobile"),
        "Bucket matches by user ID": n(cand["Bucket Match By"] == "User ID"),
        "Bucket matches by email": n(cand["Bucket Match By"] == "Email"),
    }
    st.table(pd.DataFrame({"Check": checks.keys(), "Count": [str(v) for v in checks.values()]}))
    st.caption(
        "Mobiles are compared on their last 10 digits, so +91 / 0 prefixes and spaces don't matter. "
        "A candidate is matched to the static files by User ID first, then mobile, then email."
    )
    if no_mobile:
        st.markdown("**Candidates without a mobile**")
        st.dataframe(cand.loc[cand["Mobile"].isna(), ["Name", "Product", "Category"]], hide_index=True)
