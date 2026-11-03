"""Round-funnel logic: detect round columns, classify statuses, build funnels."""
from __future__ import annotations

import re
import urllib.request

import pandas as pd

# --------------------------------------------------------------------------- #
# Known round column names (exact matches after normalisation)
# --------------------------------------------------------------------------- #
ROUND_COLUMNS = [
    # --- given ---
    "Screening Round", "Assessment Round", "Assignment Round", "TR Round 1", "TR Round 2",
    "TR Round 3", "TR Round", "Technical Round", "Technical Discussion", "Technical + HR Round",
    "Technical Screening", "HR Round", "Hr Round", "MR Round", "MR", "Final Status",
    "Final round", "Final Round", "Final Call", "Founder Round", "CEO round\n(optional)",
    "CEO Round", "CTO Round", "Behavioral Round", "Personal Interview", "General Interview",
    "Interview", "Interview Round 1", "Interview Round 2", "AI Interview", "Telephonic round",
    "Telephonic Round", "Telephonic Interview", "Communication Round", "Introductory call",
    "Screening Call", "screening call", "Verbal Screening", "Screening", "Screening Status",
    "Online Test", "Online Coding", "Coding test", "Aptitude Test", "Written Test",
    "Group Discussion", "GD", "Assignment", "Assignment (Task)", "Assignment \nRound",
    "Assignment Submissions", "Hackathon", "Client interaction", "Interview status",
    "Invite status", "Shortlisted", "Status", "Attended", "Registration status",
    # --- added ---
    "Resume Screening", "Resume Shortlisting", "Profile Shortlisting", "Shortlist Status",
    "Pre Screening", "Pre-Screening", "Initial Screening", "HR Screening", "Phone Screening",
    "Assessment", "Assessment Status", "Test Status", "Test", "Coding Round", "Coding Assessment",
    "Coding Challenge", "Online Assessment", "OA", "Aptitude Round", "MCQ Round",
    "Technical Test", "Machine Test", "Practical Round", "Live Coding", "Pair Programming",
    "Assignment 1", "Assignment 2", "Assignment 3", "Assignment Status", "Take Home Assignment",
    "Case Study", "Case Study Round", "Presentation Round", "Project Round", "Task Round",
    "TR1", "TR2", "TR3", "TR 1", "TR 2", "TR 3", "TR1 Status", "TR2 Status",
    "Technical Round 1", "Technical Round 2", "Technical Round 3", "Technical Interview",
    "Technical Interview 1", "Technical Interview 2", "Tech Round", "Tech Round 1", "Tech Round 2",
    "System Design Round", "DSA Round", "L1", "L2", "L3", "L1 Round", "L2 Round", "L3 Round",
    "Round 1", "Round 2", "Round 3", "Round 4", "Interview Round 3",
    "Managerial Round", "Manager Round", "Hiring Manager Round", "Director Round", "VP Round",
    "Panel Interview", "Panel Round", "Face to Face", "F2F Round", "F2F", "Virtual Interview",
    "Video Interview", "Culture Fit", "Cultural Fit Round", "Culture Round", "Values Round",
    "HR Interview", "HR Discussion", "HR", "Salary Discussion", "Fitment Round",
    "Client Round", "Client Interview", "Customer Round",
    "Final Interview", "Final Result", "Final Selection", "Result", "Selection Status",
    "Offer Status", "Offer", "Offer Released", "Offer Accepted", "Joining Status", "Joined",
    "Onboarding Status", "Document Verification", "BGV", "Background Verification",
    "Attendance", "Registration", "Interested", "Confirmation", "Confirmation Status",
]

# Generalised name patterns (used when the exact list doesn't match)
_INCLUDE = re.compile(
    r"\bround\b|interview|screening|\bscreen\b|assessment|assignment|\btest\b|coding|aptitude|"
    r"written|hackathon|discussion|\bgd\b|\b(tr|mr|hr|l\d|f2f|oa)\s*\d*\b|"
    r"\b(ceo|cto|coo|founder|director|manager|managerial|panel|client|culture|cultural|behaviou?ral)\b|"
    r"\bfinal\b|\boffer\b|selection|shortlist|\bstatus\b|\bresult\b|\bcall\b|joining|\bjoined\b|"
    r"\btask\b|case study|machine test|attended|attendance|registration|\binvite\b|bgv|verification"
)
# Columns that look like rounds by name but hold links / dates / scores / free text
_EXCLUDE = re.compile(
    r"link|url|github|git-hub|git hub|repo|recording|\bloom\b|drive|published|deployed|"
    r"\bdate\b|\btime\b|\bslot\b|\bscore\b|marks|percent|cgpa|gpa|feedback|comment|remark|"
    r"reason|notes?\b|panelist|interviewer|evaluator|\bby\b|e-?mail|mobile|phone number|"
    r"resume|location|\bname\b|course|department|year|college|\bctc\b|expected"
)


def norm_name(c) -> str:
    s = str(c).lower().replace("\n", " ")
    s = re.sub(r"\(optional\)", "", s)
    return re.sub(r"\s+", " ", s).strip()


_KNOWN = {norm_name(c) for c in ROUND_COLUMNS}


# --------------------------------------------------------------------------- #
# Status classification
# --------------------------------------------------------------------------- #
STAGES = ["Cleared", "Rejected", "Pending", "No-show", "Dropped", "Other"]

_RULES = [  # checked in order; negatives before positives
    ("No-show", r"not attended|did ?n[o']?t attend|\babsent\b|no[ -]?show|not respond|no response|"
                r"unreachable|not reachable|\brnr\b|not appeared|did ?n[o']?t appear|not picking|"
                r"switched off|call not"),
    ("Dropped", r"not interested|declin|withdr[ae]w|backed out|\bdrop|opted out|offer reject|"
                r"not joined|did ?n[o']?t join|abscond|not submitted|did ?n[o']?t submit|not willing|"
                r"\bduplicate\b"),
    ("Rejected", r"reject|not selected|not cleared|not shortlisted|not qualified|disqualif|\bfail|"
                 r"not fit|\bunfit\b|not recommended|\bno hire\b|^no$|^n$|not moved|^x$"),
    ("Pending", r"schedul|pending|in progress|on hold|\bhold\b|awaiting|yet to|waiting|under review|"
                r"in review|tbd|ongoing|\binvited\b|\bsent\b|^submitted|follow ?up|in process|"
                r"to be |not started|next week|planned"),
    ("Cleared", r"select|cleared|\bclear\b|shortlist|qualified|\bpass|hired|\bhire\b|offer|accept|"
                r"\bjoined\b|moved to|next round|^yes$|^y$|\bdone\b|completed|^attended|registered|"
                r"recommended|go ahead|onboard|confirmed|proceed|\bok\b|\bgood\b|^✓|^✔|true"),
]
_RULES = [(stage, re.compile(p)) for stage, p in _RULES]


def classify(value, extra_pass=(), extra_fail=()) -> str:
    """Map a raw round value to one of STAGES, or '' for blank."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return ""
    v = re.sub(r"\s+", " ", str(value).strip().lower())
    if v in ("", "-", "--", "nan", "none", "null", "#ref!", "#n/a", "na", "n/a"):
        return ""
    if any(w and w in v for w in extra_fail):
        return "Rejected"
    if any(w and w in v for w in extra_pass):
        return "Cleared"
    for stage, rx in _RULES:
        if rx.search(v):
            return stage
    return "Other"


# --------------------------------------------------------------------------- #
# Column detection
# --------------------------------------------------------------------------- #
def detect_round_columns(df: pd.DataFrame, min_status_share: float = 0.5) -> list[str]:
    """Round columns in sheet order.

    A column qualifies if its name is a known round name, or matches the generic
    round patterns without matching an exclude pattern, AND its filled values
    mostly look like statuses (so free-text / link columns are skipped).
    """
    found = []
    for c in df.columns:
        n = norm_name(c)
        if not n or n.startswith("unnamed"):
            continue
        known = n in _KNOWN
        if not known and (not _INCLUDE.search(n) or _EXCLUDE.search(n)):
            continue
        vals = df[c].dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        if len(vals) == 0:
            if known:
                found.append(c)  # round not started yet
            continue
        if vals.str.contains(r"https?://", regex=True).mean() > 0.2:
            continue
        share = (vals.map(classify) != "Other").mean()
        if share >= min_status_share or (known and vals.nunique() <= 12):
            found.append(c)
    return found


# --------------------------------------------------------------------------- #
# Funnel
# --------------------------------------------------------------------------- #
def funnel_table(stages: pd.DataFrame, round_cols: list[str], total: int) -> pd.DataFrame:
    """stages: DataFrame of classified statuses (one column per round)."""
    rows = []
    for c in round_cols:
        s = stages[c]
        reached = int((s != "").sum())
        counts = {k: int((s == k).sum()) for k in STAGES}
        rows.append(
            {
                "Round": c.replace("\n", " "),
                "Reached": reached,
                **counts,
                "Cleared % of reached": round(100 * counts["Cleared"] / reached, 1) if reached else 0.0,
                "Cleared % of total": round(100 * counts["Cleared"] / total, 1) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def out_of_order(stages: pd.DataFrame, round_cols: list[str]) -> pd.Series:
    """True where a candidate has a later-round status but an earlier round isn't Cleared."""
    flag = pd.Series(False, index=stages.index)
    not_cleared_before = pd.Series(False, index=stages.index)
    for c in round_cols:
        flag |= not_cleared_before & (stages[c] != "")
        not_cleared_before |= stages[c] != "Cleared"
    return flag


def furthest_cleared(stages: pd.DataFrame, round_cols: list[str]) -> pd.Series:
    out = pd.Series("—", index=stages.index, dtype=object)
    for c in round_cols:
        out[stages[c] == "Cleared"] = c.replace("\n", " ")
    return out


def current_outcome(stages: pd.DataFrame, round_cols: list[str]) -> pd.Series:
    """Status in the last round the candidate has any value for."""
    out = pd.Series("Not started", index=stages.index, dtype=object)
    for c in round_cols:
        has = stages[c] != ""
        out[has] = stages.loc[has, c] + " @ " + c.replace("\n", " ")
    return out


# --------------------------------------------------------------------------- #
# Google Sheets
# --------------------------------------------------------------------------- #
def gsheet_csv_url(url: str) -> str:
    url = url.strip()
    gid = re.search(r"[#?&]gid=(\d+)", url)
    gid_q = f"&gid={gid.group(1)}" if gid else ""
    m = re.search(r"/spreadsheets/d/e/([\w-]+)", url)  # "Publish to web" link
    if m:
        return f"https://docs.google.com/spreadsheets/d/e/{m.group(1)}/pub?output=csv{gid_q}"
    m = re.search(r"/spreadsheets/d/([\w-]+)", url)
    if m:
        return f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv{gid_q}"
    raise ValueError("That doesn't look like a Google Sheets link.")


def fetch_gsheet(url: str) -> bytes:
    req = urllib.request.Request(gsheet_csv_url(url), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
        ctype = r.headers.get("Content-Type", "")
    if "text/html" in ctype or data.lstrip()[:15].lower().startswith((b"<!doctype", b"<html")):
        raise PermissionError(
            "Google returned a sign-in page. Set the sheet's sharing to "
            "'Anyone with the link → Viewer' (or upload a CSV export instead)."
        )
    return data
