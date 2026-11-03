"""Placement-pool totals (bucket + assessment cleared) and how many applied to this company."""
import pandas as pd
import streamlit as st

CAT_ORDER = ["Intensive", "Academy", "External"]


def pool_coverage(pools: dict, tr_sum: pd.DataFrame, tr_rule: str, cand: pd.DataFrame) -> pd.DataFrame:
    """One row per pool group: size of the pool vs. how many applied to this company."""
    bucket_sizes = pools["bucket"]["_bucket"].replace("", pd.NA).dropna().value_counts()
    label = lambda b: b if "bucket" in b.lower() else f"Bucket {b}"  # noqa: E731
    groups = [(label(b), int(bucket_sizes.get(b, 0)), cand["Bucket"] == b) for b in pools["bucket_values"]]
    groups += [
        ("All buckets", len(pools["bucket"]), cand["In Bucket"]),
        ("Assessment cleared", len(tr_sum), cand["In Assessment Pool"]),
        (f"TR Cleared ({tr_rule})", int(tr_sum["tr_cleared"].sum()), cand["TR Cleared"]),
    ]
    rows = []
    for name, size, mask in groups:
        applied = int(mask.sum())
        row = {
            "Pool group": name,
            "In pool": size,
            "Applied": applied,
            "% of pool applied": round(100 * applied / size, 1) if size else 0.0,
            "Not applied": max(size - applied, 0),
        }
        for c in CAT_ORDER:
            row[f"Applied · {c}"] = int((mask & (cand["Category"] == c)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def render_pool_metrics(pools, tr_sum, cand):
    """Headline row: pool totals and how many of them applied."""
    st.markdown("**Placement pool**")
    in_b, in_a = int(cand["In Bucket"].sum()), int(cand["In Assessment Pool"].sum())
    nb, na = len(pools["bucket"]), len(tr_sum)
    r = st.columns(5)
    r[0].metric("Total candidates in buckets", f"{nb:,}", help="Everyone in the static bucket file")
    r[1].metric("Bucket students applied", f"{in_b:,}", help=f"{100 * in_b / nb:.1f}% of the bucket pool" if nb else None)
    r[2].metric("Total assessment cleared", f"{na:,}", help="Everyone in the static assessment pool file")
    r[3].metric("Assessment cleared applied", f"{in_a:,}", help=f"{100 * in_a / na:.1f}% of the assessment pool" if na else None)
    r[4].metric("Assessment cleared, not applied", f"{max(na - in_a, 0):,}")


def render_pool_tab(pools, tr_sum, tr_rule, cand, company=""):
    cov = pool_coverage(pools, tr_sum, tr_rule, cand)
    st.caption("How much of the placement pool applied to this company, and from which product.")
    st.dataframe(cov, hide_index=True, width="stretch")
    st.download_button(
        "⬇️ Download pool coverage CSV",
        cov.to_csv(index=False).encode("utf-8"),
        file_name=f"{(company or 'company').replace(' ', '_')}_pool_coverage.csv",
        mime="text/csv",
        key="pool_cov_dl",
    )
