# Google authentication:
# Put your Google service-account JSON in this folder as:
#   service_account.json
#
# If your Google Docs are set to "Anyone with the link -> Viewer",
# you do not need to share each document with the service-account email.
# token.json is NOT required.
#

# Multi-key setup:
#   pip install python-dotenv
#   Put MISTRAL_API_KEY_1, MISTRAL_API_KEY_2, ... in .env
#   OR put the raw keys (one per line, labels allowed) in keys.txt.

"""
Round-Level Interview Intelligence Generator

Pipeline:
CSV
  -> Company
  -> Hiring Intelligence Google Docs
  -> Read every Google Docs tab
  -> LLM extracts candidate / round / questions / topics
  -> Group candidates by round
  -> LLM creates round-level analysis
  -> Create ONE Google Doc per company

Setup:
1. pip install pandas mistralai python-dotenv google-api-python-client google-auth
2. Enable the Google Docs API and Google Drive API in Google Cloud.
3. Create a SERVICE ACCOUNT key (JSON) and save it as service_account.json.
4. Share each source Hiring Intelligence doc with the service-account email
   (or make sure "Anyone with the link" access works for your docs).
5. Set SHARE_WITH_EMAIL=you@example.com so the generated docs are shared
   with you (they are owned by the service account otherwise).
6. Put your CSV beside this script (default name: RSA_DATA.csv).
7. Put MISTRAL_API_KEY (or MISTRAL_API_KEY_1, _2, ...) in .env, or keys in keys.txt.
8. Run: python trail.py

Mistral is used through the official `mistralai` Python SDK.
The default model is `ministral-8b-latest` (override with MISTRAL_MODEL).
"""

import os
import re
from dotenv import load_dotenv
import json
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd

from googleapiclient.discovery import build
from google.oauth2 import service_account
try:
    from mistralai.client import Mistral   # mistralai >= 2.x
except ImportError:
    from mistralai import Mistral          # mistralai 1.x


# ============================================================
# CONFIG
# ============================================================

CSV_FILE = os.getenv("CSV_FILE", "RSA_DATA.csv")
SERVICE_ACCOUNT_FILE = os.getenv(
    "GOOGLE_SERVICE_ACCOUNT_FILE",
    "service_account.json",
)

MISTRAL_MODEL = os.getenv(
    "MISTRAL_MODEL",
    "ministral-8b-latest",
)

SHARE_WITH_EMAIL = os.getenv("SHARE_WITH_EMAIL", "").strip()

OUTPUT_DIR = Path("output")
RAW_DIR = OUTPUT_DIR / "raw_json"

MAX_DOCUMENT_CHARS = int(os.getenv("MAX_DOCUMENT_CHARS", "30000"))
MAX_ROUND_CHARS = int(os.getenv("MAX_ROUND_CHARS", "50000"))
SLEEP_BETWEEN_LLM_CALLS = float(
    os.getenv("SLEEP_BETWEEN_LLM_CALLS", "0.5")
)

# Full Docs scope is required because the script also CREATES the
# final company-level Google Docs.
SCOPES = [
    "https://www.googleapis.com/auth/documents",
    # Needed to share the generated docs (files this app created).
    "https://www.googleapis.com/auth/drive.file",
]

# ============================================================
# GOOGLE SERVICE ACCOUNT
# ============================================================

def authenticate_google():
    """Authenticate using a Google service-account JSON file.

    No OAuth browser login and no token.json are required.
    """
    credentials_path = Path(SERVICE_ACCOUNT_FILE)

    if not credentials_path.exists():
        raise RuntimeError(
            f"Google service-account file not found: "
            f"{credentials_path.resolve()}"
        )

    try:
        creds = service_account.Credentials.from_service_account_file(
            str(credentials_path),
            scopes=SCOPES,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Google service-account JSON: {exc}"
        ) from exc

    docs_service = build(
        "docs",
        "v1",
        credentials=creds,
    )

    # Drive service is useful later if you want to manage/share generated
    # documents. It is not required just to read public Docs or create a Doc.
    drive_service = build(
        "drive",
        "v3",
        credentials=creds,
    )

    return docs_service, drive_service


# ============================================================
# MISTRAL MULTI-KEY CLIENT
# ============================================================

def load_mistral_keys():
    """Load Mistral keys from keys.txt or environment variables.

    Recommended:
      keys.txt

    The file may contain account labels and blank lines. The script
    extracts 32-character alphanumeric Mistral key tokens and removes
    duplicates while preserving order.

    Alternatively, .env may contain:
      MISTRAL_API_KEY_1=...
      MISTRAL_API_KEY_2=...
      ...
    """
    load_dotenv()

    keys = []

    # First: numbered environment variables.
    numbered = []
    for env_name, value in os.environ.items():
        match = re.fullmatch(r"MISTRAL_API_KEY_(\d+)", env_name)
        if match and value and value.strip():
            numbered.append(
                (int(match.group(1)), value.strip().strip('"').strip("'"))
            )

    for _, value in sorted(numbered):
        keys.append(value)

    # Also allow a single MISTRAL_API_KEY.
    single = os.getenv("MISTRAL_API_KEY")
    if single and single.strip():
        keys.append(single.strip().strip('"').strip("'"))

    # If no environment keys exist, read keys.txt.
    keys_file = Path("keys.txt")
    if not keys and keys_file.exists():
        raw = keys_file.read_text(encoding="utf-8")

        # Your supplied Mistral keys are 32-character alphanumeric tokens.
        candidates = re.findall(
            r"(?<![A-Za-z0-9])[A-Za-z0-9]{32}(?![A-Za-z0-9])",
            raw,
        )
        keys.extend(candidates)

    # Remove duplicates while preserving order.
    unique_keys = []
    seen = set()

    for key in keys:
        if key and key not in seen:
            seen.add(key)
            unique_keys.append(key)

    if not unique_keys:
        raise RuntimeError(
            "No Mistral API keys found. Put your keys in keys.txt "
            "or set MISTRAL_API_KEY_1, MISTRAL_API_KEY_2, etc."
        )

    return unique_keys


MISTRAL_KEYS = load_mistral_keys()
_mistral_index = 0


def get_mistral_client():
    return Mistral(
        api_key=MISTRAL_KEYS[_mistral_index]
    )


def mistral_chat_complete(**kwargs):
    """Call Mistral and rotate through keys if a request fails."""
    global _mistral_index

    last_error = None

    for attempt in range(len(MISTRAL_KEYS)):
        client = get_mistral_client()

        try:
            return client.chat.complete(**kwargs)

        except Exception as exc:
            last_error = exc

            # Move to the next key.
            _mistral_index = (
                _mistral_index + 1
            ) % len(MISTRAL_KEYS)

            print(
                f"Mistral key failed "
                f"(attempt {attempt + 1}/{len(MISTRAL_KEYS)}). "
                f"Switching to next key..."
            )

    raise RuntimeError(
        f"All {len(MISTRAL_KEYS)} Mistral API keys failed. "
        f"Last error: {last_error}"
    )


def validate_config():
    """Validate files/config before processing."""
    if not Path(CSV_FILE).exists():
        raise RuntimeError(
            f"CSV file not found: {Path(CSV_FILE).resolve()}"
        )

    if not Path(SERVICE_ACCOUNT_FILE).exists():
        raise RuntimeError(
            f"Service-account JSON not found: "
            f"{Path(SERVICE_ACCOUNT_FILE).resolve()}"
        )

    if not MISTRAL_KEYS:
        raise RuntimeError("No Mistral API keys loaded.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not SHARE_WITH_EMAIL:
        print(
            "WARNING: SHARE_WITH_EMAIL is not set. Generated docs will be "
            "owned by the service account and you won't be able to open them."
        )


# ============================================================
# GOOGLE DOC HELPERS
# ============================================================

def extract_document_id(url):
    if not url or pd.isna(url):
        return None

    match = re.search(
        r"docs\.google\.com/document/d/([A-Za-z0-9_-]+)",
        str(url)
    )

    return match.group(1) if match else None


def extract_text_from_body(body):
    """Collect text from paragraphs, tables (incl. nested) and TOCs."""
    parts = []

    def read_elements(elements):
        for element in elements:
            paragraph = element.get("paragraph")
            if paragraph:
                for item in paragraph.get("elements", []):
                    text_run = item.get("textRun")
                    if text_run:
                        parts.append(text_run.get("content", ""))

            table = element.get("table")
            if table:
                for row in table.get("tableRows", []):
                    cells = []
                    for cell in row.get("tableCells", []):
                        start = len(parts)
                        read_elements(cell.get("content", []))
                        cell_text = "".join(parts[start:]).strip()
                        del parts[start:]
                        cells.append(cell_text)
                    parts.append(" | ".join(cells) + "\n")

            toc = element.get("tableOfContents")
            if toc:
                read_elements(toc.get("content", []))

    read_elements(body.get("content", []))
    return "".join(parts)


def get_document_tabs(docs_service, document_id):
    """
    Returns every tab, including nested child tabs.
    """

    document = (
        docs_service.documents()
        .get(
            documentId=document_id,
            includeTabsContent=True
        )
        .execute()
    )

    tabs = []

    def walk(tab_list):
        for tab in tab_list:
            props = tab.get("tabProperties", {})
            document_tab = tab.get("documentTab", {})

            tabs.append({
                "tab_id": props.get("tabId", ""),
                "tab_title": props.get("title", ""),
                "text": extract_text_from_body(
                    document_tab.get("body", {})
                ),
            })

            children = tab.get("childTabs", [])
            if children:
                walk(children)

    walk(document.get("tabs", []))

    return tabs


# ============================================================
# CSV HELPERS
# ============================================================

def extract_google_doc_links(cell):
    """
    Extract Google Docs document URLs from the Hiring Intelligence cell.
    Handles markdown links and escaped URLs.
    """

    if cell is None or pd.isna(cell):
        return []

    value = str(cell).replace(r"\https://", "https://")

    # Capture document URLs whether followed by /edit, query params,
    # markdown delimiters, HTML, etc.
    matches = re.findall(
        r"https?://docs\.google\.com/document/d/[A-Za-z0-9_-]+(?:[^)\s<]*)",
        value
    )

    clean = []

    for url in matches:
        url = url.rstrip("],.;")
        url = url.replace("\\", "")

        # Keep only the actual document URL portion.
        m = re.search(
            r"(https?://docs\.google\.com/document/d/[A-Za-z0-9_-]+)",
            url
        )

        if m:
            clean.append(m.group(1))

    return list(dict.fromkeys(clean))


def load_companies():
    """
    Reads a normal CSV.

    If the source was exported from the markdown-style table shown in the
    conversation, convert it to CSV first. The script expects:
      Company Name
      Hiring Intelligence
    """

    df = pd.read_csv(
        CSV_FILE,
        dtype=str,
        keep_default_na=False
    )

    required = [
        "Company Name",
        "Hiring Intelligence",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing CSV columns: " + ", ".join(missing)
        )

    return df


# ============================================================
# LLM
# ============================================================

def call_llm(system_prompt, user_prompt, temperature=0.1):
    """
    Calls the official Mistral Python SDK.

    Mistral chat completions support JSON mode through:
    response_format={"type": "json_object"}.
    """

    response = mistral_chat_complete(
        model=MISTRAL_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=temperature,
        response_format={
            "type": "json_object"
        },
    )

    content = response.choices[0].message.content

    # Some SDK versions return a list of content chunks instead of a str.
    if isinstance(content, list):
        content = "".join(
            getattr(chunk, "text", "") or
            (chunk.get("text", "") if isinstance(chunk, dict) else "")
            for chunk in content
        )

    result = parse_json_response(content)

    if not isinstance(result, dict):
        raise ValueError(
            f"LLM returned {type(result).__name__}, expected a JSON object"
        )

    return result


def parse_json_response(content):
    if isinstance(content, dict):
        return content

    text = str(content).strip()

    # Remove markdown code fences if present.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to locate the outer JSON object.
        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])

        raise


# ============================================================
# LLM: CANDIDATE / ROUND EXTRACTION
# ============================================================

EXTRACTION_SYSTEM_PROMPT = """
You are an interview-data extraction system.

Your job is to extract factual information from one Google Docs interview tab.

IMPORTANT:
- Do not invent candidate names, rounds, questions, technologies, or outcomes.
- The tab title is evidence.
- The document body is evidence.
- If the tab title and body disagree, use the most explicit information from
  the document body and mention the conflict in notes.
- If round cannot be determined, return "UNKNOWN".
- If candidate cannot be determined, return "UNKNOWN".
- Preserve interview questions as close to their original wording as possible.
- Do not create questions that are not present.
- Ignore navigation text, blank lines, generic document UI text, and URLs.

Return ONLY valid JSON with this structure:

{
  "candidate": "string",
  "round": "string",
  "interviewer": "string or UNKNOWN",
  "outcome": "string or UNKNOWN",
  "questions": [
    {
      "question": "exact or near-exact question",
      "topic": "best-supported topic or UNKNOWN"
    }
  ],
  "notes": ["short factual notes"]
}
"""


def extract_candidate_interview(tab_title, text):
    text = text[:MAX_DOCUMENT_CHARS]

    user_prompt = f"""
TAB TITLE:
{tab_title}

DOCUMENT CONTENT:
{text}
"""

    return call_llm(
        EXTRACTION_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.0
    )


# ============================================================
# ROUND LEVEL LLM ANALYSIS
# ============================================================

ROUND_ANALYSIS_SYSTEM_PROMPT = """
You are an interview intelligence analyst.

You are given interview records from multiple candidates who went through
the SAME company and SAME interview round.

Create a factual round-level summary.

IMPORTANT:
- Do not invent information.
- Base every conclusion only on supplied interview records.
- Preserve actual question wording where possible.
- Group semantically identical or nearly identical questions together.
- Distinguish common/repeated questions from questions asked only once.
- Count occurrences based on the supplied candidate records.
- Do not treat a question as common unless it occurs for at least two candidates.
- Difficulty should be "Easy", "Medium", "Hard", or "Mixed" only when supported
  by the questions. Otherwise use "Unknown".
- Topic names should be concise.

Return ONLY valid JSON:

{
  "round": "TR1",
  "candidate_count": 0,
  "topics": [
    {
      "topic": "string",
      "question_count": 0
    }
  ],
  "common_questions": [
    {
      "question": "string",
      "candidate_count": 0
    }
  ],
  "other_questions": [
    {
      "question": "string",
      "candidate_count": 0
    }
  ],
  "difficulty": "Easy|Medium|Hard|Mixed|Unknown",
  "round_summary": "short factual summary"
}
"""


def analyze_round(round_name, candidate_records):
    compact_records = []

    for record in candidate_records:
        compact_records.append({
            "candidate": record.get(
                "candidate",
                "UNKNOWN"
            ),
            "outcome": record.get(
                "outcome",
                "UNKNOWN"
            ),
            "questions": record.get(
                "questions",
                []
            ),
        })

    serialized = json.dumps(
        compact_records,
        ensure_ascii=False,
        indent=2
    )

    if len(serialized) > MAX_ROUND_CHARS:
        serialized = serialized[:MAX_ROUND_CHARS] + (
            "\n[TRUNCATED: remaining interview text omitted]"
        )

    user_prompt = f"""
COMPANY ROUND:
{round_name}

INTERVIEW RECORDS:
{serialized}
"""

    return call_llm(
        ROUND_ANALYSIS_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.1
    )


# ============================================================
# COMPANY LEVEL ANALYSIS
# ============================================================

COMPANY_ANALYSIS_SYSTEM_PROMPT = """
You are an interview intelligence analyst.

You are given round-level summaries for ONE company.

Create a concise company-level analysis.

Only use information present in the summaries.
Do not invent rounds, technologies, questions, or hiring criteria.

Return ONLY valid JSON:

{
  "rounds_conducted": ["TR1", "TR2"],
  "most_tested_topics": [
    {
      "topic": "string",
      "evidence": "short factual explanation"
    }
  ],
  "frequently_asked_questions": [
    "question"
  ],
  "round_progression": "short factual observation",
  "overall_summary": "short factual summary"
}
"""


def analyze_company(company_name, round_summaries):
    user_prompt = f"""
COMPANY:
{company_name}

ROUND SUMMARIES:
{json.dumps(
    round_summaries,
    ensure_ascii=False,
    indent=2
)}
"""

    return call_llm(
        COMPANY_ANALYSIS_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.1
    )


# ============================================================
# GOOGLE DOC OUTPUT
# ============================================================

def share_google_doc(drive_service, document_id, email):
    drive_service.permissions().create(
        fileId=document_id,
        body={"type": "user", "role": "writer", "emailAddress": email},
        sendNotificationEmail=False,
    ).execute()


def create_google_doc(docs_service, title, content):
    document = (
        docs_service.documents()
        .create(
            body={"title": title}
        )
        .execute()
    )

    document_id = document["documentId"]

    docs_service.documents().batchUpdate(
        documentId=document_id,
        body={
            "requests": [
                {
                    "insertText": {
                        "location": {"index": 1},
                        "text": content,
                    }
                }
            ]
        }
    ).execute()

    return document_id


def format_round(round_name, summary):
    lines = []

    lines.append(f"{round_name}")
    lines.append("=" * 70)
    lines.append(
        f"Candidates Assessed: "
        f"{summary.get('candidate_count', 'Unknown')}"
    )
    lines.append(
        f"Difficulty: "
        f"{summary.get('difficulty', 'Unknown')}"
    )
    lines.append("")

    lines.append("Topics Covered")
    lines.append("-" * 30)

    for item in summary.get("topics", []):
        topic = item.get("topic", "UNKNOWN")
        count = item.get("question_count", 0)
        lines.append(f"• {topic} ({count} questions)")

    if not summary.get("topics"):
        lines.append("• UNKNOWN")

    lines.append("")
    lines.append("Common Questions")
    lines.append("-" * 30)

    common = summary.get("common_questions", [])

    if common:
        for item in common:
            q = item.get("question", "")
            count = item.get("candidate_count", 0)
            lines.append(
                f"• {q} "
                f"[asked to {count} candidates]"
            )
    else:
        lines.append("• No repeated question identified.")

    lines.append("")
    lines.append("Other Questions")
    lines.append("-" * 30)

    others = summary.get("other_questions", [])

    if others:
        for item in others:
            q = item.get("question", "")
            count = item.get("candidate_count", 0)
            lines.append(
                f"• {q} "
                f"[asked to {count} candidate(s)]"
            )
    else:
        lines.append("• None identified.")

    lines.append("")
    lines.append("Round Summary")
    lines.append("-" * 30)
    lines.append(
        summary.get(
            "round_summary",
            "No summary available."
        )
    )

    lines.append("")

    return "\n".join(lines)


def format_company_document(
    company_name,
    round_summaries,
    company_analysis
):
    lines = []

    lines.append(company_name)
    lines.append("INTERVIEW INTELLIGENCE")
    lines.append("=" * 80)
    lines.append("")
    lines.append(
        "Generated from candidate interview documents and grouped by round."
    )
    lines.append("")

    # --------------------------------------------------------
    # Overall analysis
    # --------------------------------------------------------

    lines.append("OVERALL COMPANY ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    lines.append("Rounds Conducted")
    lines.append("-" * 30)

    for round_name in company_analysis.get(
        "rounds_conducted",
        []
    ):
        lines.append(f"• {round_name}")

    if not company_analysis.get("rounds_conducted"):
        lines.append("• UNKNOWN")

    lines.append("")
    lines.append("Most Tested Topics")
    lines.append("-" * 30)

    for item in company_analysis.get(
        "most_tested_topics",
        []
    ):
        lines.append(
            f"• {item.get('topic', 'UNKNOWN')}: "
            f"{item.get('evidence', '')}"
        )

    lines.append("")
    lines.append("Frequently Asked Questions")
    lines.append("-" * 30)

    for question in company_analysis.get(
        "frequently_asked_questions",
        []
    ):
        lines.append(f"• {question}")

    if not company_analysis.get(
        "frequently_asked_questions"
    ):
        lines.append("• None identified.")

    lines.append("")
    lines.append("Round Progression")
    lines.append("-" * 30)
    lines.append(
        company_analysis.get(
            "round_progression",
            "No progression analysis available."
        )
    )

    lines.append("")
    lines.append("Overall Summary")
    lines.append("-" * 30)
    lines.append(
        company_analysis.get(
            "overall_summary",
            "No overall summary available."
        )
    )

    lines.append("")
    lines.append("")
    lines.append("ROUND-LEVEL ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    for round_name, summary in round_summaries.items():
        lines.append(
            format_round(
                round_name,
                summary
            )
        )
        lines.append("\n")

    return "\n".join(lines)


# ============================================================
# ROUND NAME NORMALISATION
# ============================================================

def normalize_round(value):
    if value is None:
        return "UNKNOWN"

    name = re.sub(r"\s+", " ", str(value)).strip()

    if not name or name.lower() in {"none", "null", "unknown", "n/a"}:
        return "UNKNOWN"

    m = re.fullmatch(
        r"(?:tr|technical(?: interview)?(?: round)?)\s*-?\s*(\d+)",
        name,
        flags=re.IGNORECASE,
    )
    if m:
        return f"TR{m.group(1)}"

    if re.fullmatch(r"hr(?: round| interview)?", name, flags=re.IGNORECASE):
        return "HR"

    return name


# ============================================================
# MAIN COMPANY PROCESSING
# ============================================================

def process_company(
    docs_service,
    drive_service,
    company_name,
    hiring_intelligence
):
    print(f"\n{'=' * 80}")
    print(f"COMPANY: {company_name}")
    print(f"{'=' * 80}")

    links = extract_google_doc_links(
        hiring_intelligence
    )

    if not links:
        print("No Google Docs links found.")
        return None

    print(
        f"Found {len(links)} Hiring Intelligence document(s)."
    )

    candidate_records = []

    # --------------------------------------------------------
    # Read every source document and every tab
    # --------------------------------------------------------

    for doc_index, link in enumerate(links, start=1):
        document_id = extract_document_id(link)

        if not document_id:
            print(
                f"[{doc_index}] Invalid Google Docs URL."
            )
            continue

        print(
            f"[{doc_index}/{len(links)}] Reading document..."
        )

        try:
            tabs = get_document_tabs(
                docs_service,
                document_id
            )
        except Exception as exc:
            print(
                f"Could not read document: {exc}"
            )
            continue

        for tab_index, tab in enumerate(
            tabs,
            start=1
        ):
            title = tab["tab_title"]
            text = tab["text"]

            if not text.strip():
                print(
                    f"  Tab {tab_index}: {title} "
                    f"(empty - skipped)"
                )
                continue

            print(
                f"  Tab {tab_index}: {title} -> LLM extraction"
            )

            try:
                extracted = extract_candidate_interview(
                    title,
                    text
                )
            except Exception as exc:
                print(
                    f"  LLM extraction failed: {exc}"
                )
                continue

            extracted["_source_document_id"] = document_id
            extracted["_source_tab"] = title
            extracted["_source_url"] = link

            candidate_records.append(
                extracted
            )

            print(
                f"    Candidate: "
                f"{extracted.get('candidate', 'UNKNOWN')}"
            )
            print(
                f"    Round: "
                f"{extracted.get('round', 'UNKNOWN')}"
            )
            print(
                f"    Questions: "
                f"{len(extracted.get('questions', []))}"
            )

            time.sleep(
                SLEEP_BETWEEN_LLM_CALLS
            )

    if not candidate_records:
        print("No interview records extracted.")
        return None

    # --------------------------------------------------------
    # Save raw extraction for debugging/audit
    # --------------------------------------------------------

    safe_company = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        company_name
    )

    raw_file = RAW_DIR / f"{safe_company}.json"

    raw_file.write_text(
        json.dumps(
            candidate_records,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Group candidate records by round
    # --------------------------------------------------------

    grouped = defaultdict(list)

    for record in candidate_records:
        round_name = normalize_round(record.get("round"))

        grouped[round_name].append(
            record
        )

    print(
        f"\nRounds found: "
        f"{', '.join(grouped.keys())}"
    )

    # --------------------------------------------------------
    # Analyze every round
    # --------------------------------------------------------

    round_summaries = {}

    for round_name, records in grouped.items():
        print(
            f"\nAnalyzing {round_name} "
            f"({len(records)} candidates)..."
        )

        try:
            summary = analyze_round(
                round_name,
                records
            )
        except Exception as exc:
            print(
                f"Round analysis failed: {exc}"
            )

            summary = {
                "round": round_name,
                "candidate_count": len(records),
                "topics": [],
                "common_questions": [],
                "other_questions": [],
                "difficulty": "Unknown",
                "round_summary": (
                    "LLM analysis failed."
                ),
            }

        summary["candidate_count"] = len({
            str(r.get("candidate", "")).strip().lower() or f"_{i}"
            for i, r in enumerate(records)
        })

        round_summaries[round_name] = summary

        time.sleep(
            SLEEP_BETWEEN_LLM_CALLS
        )

    # --------------------------------------------------------
    # Company-level analysis
    # --------------------------------------------------------

    print("\nCreating company-level analysis...")

    try:
        company_analysis = analyze_company(
            company_name,
            list(round_summaries.values())
        )
    except Exception as exc:
        print(
            f"Company analysis failed: {exc}"
        )

        company_analysis = {
            "rounds_conducted": list(
                round_summaries.keys()
            ),
            "most_tested_topics": [],
            "frequently_asked_questions": [],
            "round_progression": (
                "LLM analysis failed."
            ),
            "overall_summary": (
                "LLM analysis failed."
            ),
        }

    # --------------------------------------------------------
    # Generate final Google Doc
    # --------------------------------------------------------

    final_content = format_company_document(
        company_name,
        round_summaries,
        company_analysis
    )

    document_title = (
        f"{company_name} - Interview Intelligence"
    )

    document_id = create_google_doc(
        docs_service,
        document_title,
        final_content
    )

    document_url = (
        f"https://docs.google.com/document/d/"
        f"{document_id}/edit"
    )

    if SHARE_WITH_EMAIL:
        try:
            share_google_doc(drive_service, document_id, SHARE_WITH_EMAIL)
        except Exception as exc:
            print(f"Could not share doc with {SHARE_WITH_EMAIL}: {exc}")

    print(
        f"\nCREATED: {document_url}"
    )

    return {
        "company": company_name,
        "document_id": document_id,
        "document_url": document_url,
        "candidate_records": len(candidate_records),
        "rounds": list(grouped.keys()),
    }


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    validate_config()

    print("Authenticating with Google...")
    docs_service, drive_service = authenticate_google()

    print("Loading CSV...")
    df = load_companies()

    print(
        f"Loaded {len(df)} company row(s)."
    )

    results = []

    for index, row in df.iterrows():
        company_name = str(
            row["Company Name"]
        ).strip()

        hiring_intelligence = row[
            "Hiring Intelligence"
        ]

        if not company_name:
            continue

        if not str(hiring_intelligence).strip():
            print(
                f"\nSkipping {company_name}: "
                "Hiring Intelligence is empty."
            )
            continue

        try:
            result = process_company(
                docs_service,
                drive_service,
                company_name,
                hiring_intelligence
            )

            if result:
                results.append(result)

        except Exception as exc:
            print(
                f"\nFAILED: {company_name}"
            )
            print(exc)

    # --------------------------------------------------------
    # Save final mapping
    # --------------------------------------------------------

    if results:
        output_df = pd.DataFrame(results)

        output_df.to_csv(
            OUTPUT_DIR / "generated_company_documents.csv",
            index=False
        )

    print("\n" + "=" * 80)
    print("PROCESS COMPLETE")
    print("=" * 80)
    print(
        f"Companies successfully processed: "
        f"{len(results)}"
    )

    if results:
        print(
            f"Mapping saved to: "
            f"{OUTPUT_DIR / 'generated_company_documents.csv'}"
        )


if __name__ == "__main__":
    main()