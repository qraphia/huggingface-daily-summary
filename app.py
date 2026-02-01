import datetime as dt
import html
import os
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests


# =========================
# Config
# =========================

APP_TITLE = "Hugging Face Daily Summary"
APP_SUBTITLE = "A clean, readable viewer for Hugging Face Daily Papers. Shows full abstracts (from arXiv when possible)."

HF_DAILY_PAPERS_API = "https://huggingface.co/api/daily_papers"
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_ABS = "https://arxiv.org/abs/{paper_id}"
ARXIV_PDF = "https://arxiv.org/pdf/{paper_id}.pdf"

HTTP_TIMEOUT_S = int(os.getenv("HTTP_TIMEOUT_S", "25"))
CACHE_TTL_S = int(os.getenv("CACHE_TTL_S", "300"))  # 5 min
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
BACKOFF_BASE_S = float(os.getenv("BACKOFF_BASE_S", "1.0"))

# Optional (helps with rate limits). Set as Space secret if you want.
HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "qraphia/huggingface-daily-summary (clean; gradio)",
        "Accept": "application/json",
    }
)
if HF_TOKEN:
    SESSION.headers["Authorization"] = f"Bearer {HF_TOKEN}"

# Cache: (date, limit) -> (ts, data, headers)
HF_CACHE: Dict[Tuple[str, int], Tuple[float, List[Dict[str, Any]], Dict[str, str]]] = {}
# Cache: arxiv_id -> (ts, arxiv_meta)
ARXIV_CACHE: Dict[str, Tuple[float, "ArxivMeta"]] = {}

CSS = """
:root{
  --bg:#f7f8fb;
  --card:#ffffff;
  --bd:#e6e8ef;
  --text:#111827;
  --muted:#6b7280;
  --chip:#f3f4f6;
  --link:#2563eb;
  --shadow: 0 1px 1px rgba(17,24,39,0.04);
}

body{ background: var(--bg); }
.gradio-container{ max-width: 1040px !important; }

.header{ margin: 4px 0 10px; }
.header h1{ margin: 0 0 0.25rem; font-size: 1.65rem; letter-spacing: -0.01em; }
.subtle{ color: var(--muted); font-size: 0.96rem; line-height: 1.45; }

.controls{
  background: var(--card);
  border: 1px solid var(--bd);
  border-radius: 16px;
  padding: 12px 12px;
  box-shadow: var(--shadow);
}

.paper-list{
  display:flex;
  flex-direction:column;
  gap: 12px;
  margin-top: 10px;
}

.card{
  background: var(--card);
  border: 1px solid var(--bd);
  border-radius: 16px;
  padding: 14px 14px;
  box-shadow: var(--shadow);
}

.card:hover{ border-color: #d9ddea; }

.title{
  font-size: 1.08rem;
  font-weight: 800;
  color: var(--text);
  line-height: 1.35;
  margin: 0;
}

.meta{
  margin-top: 8px;
  color: var(--muted);
  font-size: 0.92rem;
  display:flex;
  flex-wrap:wrap;
  gap: 8px 10px;
  align-items:center;
}

.pill{
  display:inline-flex;
  align-items:center;
  gap: 6px;
  padding: 2px 10px;
  background: var(--chip);
  border: 1px solid var(--bd);
  border-radius: 999px;
  color: #374151;
  font-size: 0.86rem;
}

.links{
  margin-top: 10px;
  display:flex;
  flex-wrap:wrap;
  gap: 10px;
}

.links a{
  color: var(--link);
  text-decoration:none;
  font-size: 0.93rem;
  font-weight: 650;
}

.links a:hover{ text-decoration: underline; }

.abstract{
  margin-top: 12px;
  border-top: 1px dashed var(--bd);
  padding-top: 12px;
}

.abstract h4{
  margin: 0 0 8px;
  font-size: 0.98rem;
  letter-spacing: -0.01em;
}

.abstract-text{
  color: var(--text);
  font-size: 0.97rem;
  line-height: 1.62;
  white-space: pre-wrap;
}

.cite{
  margin-top: 10px;
}

details.citebox summary{
  cursor: pointer;
  color: #374151;
  font-weight: 700;
  font-size: 0.93rem;
}

pre.bibtex{
  margin-top: 8px;
  padding: 10px 10px;
  background: #0b1220;
  color: #e5e7eb;
  border-radius: 12px;
  border: 1px solid #111827;
  overflow-x: auto;
  font-size: 0.86rem;
  line-height: 1.45;
}

.small-muted{ color: var(--muted); font-size: 0.88rem; }

.error-box{
  background: #fff1f2;
  border: 1px solid #fecdd3;
  border-radius: 16px;
  padding: 12px 12px;
  color: #7f1d1d;
}

.diag{
  margin-top: 10px;
  background: #f9fafb;
  border: 1px solid var(--bd);
  border-radius: 16px;
  padding: 10px 10px;
  color: #374151;
  font-size: 0.88rem;
  white-space: pre-wrap;
}
"""


# =========================
# Models / Helpers
# =========================

@dataclass
class ApiError:
    status_code: Optional[int]
    reason: str
    body_snippet: str
    headers: Dict[str, str]


@dataclass
class ArxivMeta:
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    year: str


def esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def utc_today_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def normalize_text(s: str) -> str:
    if not s:
        return ""
    out = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    # remove possible "L1:" style markers found in some HF summaries
    for i in range(1, 120):
        out = out.replace(f"L{i}:", "")
    while "\n\n\n" in out:
        out = out.replace("\n\n\n", "\n\n")
    return out.strip()


def is_arxiv_id(paper_id: str) -> bool:
    # Basic acceptance: arXiv new-style (####.#####) or old-style (cs/######)
    pid = (paper_id or "").strip()
    if not pid:
        return False
    if pid.count("/") == 1 and any(ch.isdigit() for ch in pid):
        return True
    if "." in pid and pid.split(".")[0].isdigit():
        return True
    return True  # HF papers IDs are typically arXiv-like; keep permissive but safe with error handling


def extract_rate_limit_headers(h: Dict[str, str]) -> Dict[str, str]:
    keep: Dict[str, str] = {}
    for k, v in (h or {}).items():
        lk = k.lower()
        if lk.startswith("ratelimit-") or lk in ("retry-after",):
            keep[k] = v
    return keep


def safe_get_json(url: str, params: Dict[str, Any]) -> Tuple[Optional[Any], Optional[ApiError], Dict[str, str]]:
    last_err: Optional[ApiError] = None
    last_headers: Dict[str, str] = {}

    for attempt in range(MAX_RETRIES + 1):
        try:
            r = SESSION.get(url, params=params, timeout=HTTP_TIMEOUT_S)
            last_headers = dict(r.headers or {})
        except requests.RequestException as e:
            last_err = ApiError(None, type(e).__name__, str(e)[:400], {})
            time.sleep(min(20.0, BACKOFF_BASE_S * (2 ** attempt)))
            continue

        if r.status_code == 200:
            try:
                return r.json(), None, extract_rate_limit_headers(last_headers)
            except Exception:
                return None, ApiError(200, "Invalid JSON", (r.text or "")[:400], extract_rate_limit_headers(last_headers)), extract_rate_limit_headers(last_headers)

        # Retryable common cases (rate limit / transient server errors)
        if r.status_code in (429, 500, 502, 503, 504):
            retry_after = r.headers.get("Retry-After")
            wait_s = min(20.0, BACKOFF_BASE_S * (2 ** attempt))
            if retry_after:
                try:
                    wait_s = max(wait_s, float(retry_after))
                except Exception:
                    pass

            last_err = ApiError(r.status_code, r.reason or "HTTP error", (r.text or "")[:400], extract_rate_limit_headers(last_headers))
            time.sleep(wait_s)
            continue

        # Non-retryable: fail fast with details
        return None, ApiError(r.status_code, r.reason or "HTTP error", (r.text or "")[:400], extract_rate_limit_headers(last_headers)), extract_rate_limit_headers(last_headers)

    return None, last_err, extract_rate_limit_headers(last_headers)


def fetch_hf_daily(date_iso: str, limit: int) -> Tuple[Optional[List[Dict[str, Any]]], Optional[ApiError], Dict[str, str], bool]:
    key = (date_iso, int(limit))
    now = time.time()
    hit = HF_CACHE.get(key)
    if hit and (now - hit[0] < CACHE_TTL_S):
        return hit[1], None, hit[2], True

    data, err, rate_headers = safe_get_json(HF_DAILY_PAPERS_API, {"date": date_iso, "limit": int(limit), "sort": "trending"})
    if err is not None:
        return None, err, rate_headers, False

    if not isinstance(data, list):
        return None, ApiError(200, "Unexpected payload", str(type(data))[:200], rate_headers), rate_headers, False

    HF_CACHE[key] = (now, data, rate_headers)
    return data, None, rate_headers, False


def fetch_arxiv_meta(arxiv_id: str) -> Optional[ArxivMeta]:
    aid = (arxiv_id or "").strip()
    if not aid:
        return None

    now = time.time()
    hit = ARXIV_CACHE.get(aid)
    if hit and (now - hit[0] < 24 * 3600):  # 24h
        return hit[1]

    # arXiv API returns Atom XML
    params = {"id_list": aid}
    try:
        r = SESSION.get(ARXIV_API, params=params, timeout=HTTP_TIMEOUT_S)
        if r.status_code != 200:
            return None
        xml_text = r.text or ""
        root = ET.fromstring(xml_text)
    except Exception:
        return None

    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = root.find("a:entry", ns)
    if entry is None:
        return None

    title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
    abstract = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
    published = (entry.findtext("a:published", default="", namespaces=ns) or "").strip()
    year = published[:4] if len(published) >= 4 else ""

    authors: List[str] = []
    for a in entry.findall("a:author", ns):
        name = (a.findtext("a:name", default="", namespaces=ns) or "").strip()
        if name:
            authors.append(name)

    meta = ArxivMeta(
        arxiv_id=aid,
        title=normalize_text(title),
        abstract=normalize_text(abstract),
        authors=authors,
        year=year,
    )
    ARXIV_CACHE[aid] = (now, meta)
    return meta


def best_effort_github_link(paper_obj: Dict[str, Any], title: str) -> str:
    # Try common candidate keys. If none, fallback to GitHub search.
    candidates = [
        "github_url", "githubUrl", "code_url", "codeUrl", "repo_url", "repoUrl",
        "repository_url", "repositoryUrl", "source_code_url", "sourceCodeUrl",
        "implementation_url", "implementationUrl",
    ]
    for k in candidates:
        v = paper_obj.get(k)
        if isinstance(v, str) and v.strip().startswith("http"):
            return v.strip()

    # Some APIs embed urls inside a list/dict
    for k in ("links", "resources", "artifacts"):
        v = paper_obj.get(k)
        if isinstance(v, list):
            for it in v:
                if isinstance(it, dict):
                    u = it.get("url") or it.get("href")
                    if isinstance(u, str) and "github.com" in u:
                        return u.strip()
        if isinstance(v, dict):
            for _, u in v.items():
                if isinstance(u, str) and "github.com" in u:
                    return u.strip()

    q = urllib.parse.quote_plus((title or "").strip()[:180])
    return f"https://github.com/search?q={q}&type=repositories"


def build_bibtex(meta: ArxivMeta) -> str:
    # Minimal, robust BibTeX without guessing DOI/journal.
    key = meta.arxiv_id.replace("/", "_")
    authors = " and ".join(meta.authors) if meta.authors else "Unknown"
    title = meta.title.replace("{", "\\{").replace("}", "\\}")
    year = meta.year or "????"
    return (
        f"@misc{{arxiv:{key},\n"
        f"  title={{ {title} }},\n"
        f"  author={{ {authors} }},\n"
        f"  year={{ {year} }},\n"
        f"  eprint={{ {meta.arxiv_id} }},\n"
        f"  archivePrefix={{arXiv}}\n"
        f"}}"
    )


# =========================
# Rendering
# =========================

def render_cards(date_iso: str, limit: int, diagnostics: bool) -> Tuple[str, str]:
    date_iso = (date_iso or "").strip()
    try:
        dt.date.fromisoformat(date_iso)
    except Exception:
        return "", "❌ Invalid date. Use YYYY-MM-DD."

    limit = int(limit)
    limit = max(1, min(50, limit))

    items, err, rate_headers, cached = fetch_hf_daily(date_iso, limit)
    if err is not None:
        diag = ""
        if diagnostics:
            diag_txt = "\n".join(
                [
                    f"status_code: {err.status_code}",
                    f"reason: {err.reason}",
                    f"body_snippet: {err.body_snippet}",
                    f"rate_limit_headers: {err.headers}",
                    f"hint: set HF_TOKEN to reduce rate-limit errors (429).",
                ]
            )
            diag = f"<div class='diag'>{esc(diag_txt)}</div>"

        box = (
            "<div class='error-box'>"
            "<b>API request failed</b><br/>"
            f"Status: {esc(str(err.status_code))}<br/>"
            f"Reason: {esc(err.reason)}<br/>"
            f"Body (snippet): {esc(err.body_snippet)}"
            "</div>"
            f"{diag}"
        )
        return box, "❌ Failed to fetch daily papers. Enable diagnostics for details."

    assert items is not None
    if not items:
        return "<div class='paper-list'></div>", "No results."

    blocks: List[str] = ["<div class='paper-list'>"]

    for idx, it in enumerate(items, start=1):
        paper = (it or {}).get("paper") or {}

        paper_id = str(paper.get("id") or it.get("id") or "").strip()
        title = str(it.get("title") or paper.get("title") or "").strip() or "(no title)"
        upvotes = paper.get("upvotes")
        upvotes_text = str(upvotes) if upvotes is not None else "—"

        # Authors: prefer arXiv API if available, else HF authors
        hf_authors = []
        authors_obj = paper.get("authors") or []
        if isinstance(authors_obj, list):
            for a in authors_obj:
                if isinstance(a, dict) and a.get("name"):
                    hf_authors.append(str(a["name"]).strip())
        authors_display = ", ".join(hf_authors[:10]) + (f", et al. (+{len(hf_authors) - 10})" if len(hf_authors) > 10 else "")
        if not authors_display:
            authors_display = "—"

        # Abstract: always try arXiv for correctness; fallback to HF summary if arXiv not available
        abstract = ""
        bibtex = ""
        arxiv_ok = bool(paper_id)
        arxiv_meta: Optional[ArxivMeta] = None
        if arxiv_ok and is_arxiv_id(paper_id):
            arxiv_meta = fetch_arxiv_meta(paper_id)

        if arxiv_meta and arxiv_meta.abstract:
            abstract = arxiv_meta.abstract
            if arxiv_meta.authors:
                authors_display = ", ".join(arxiv_meta.authors[:10]) + (f", et al. (+{len(arxiv_meta.authors) - 10})" if len(arxiv_meta.authors) > 10 else "")
            bibtex = build_bibtex(arxiv_meta)
        else:
            # HF sometimes provides summary; use it as fallback to avoid empty UI.
            abstract = normalize_text(str(it.get("summary") or paper.get("summary") or ""))
            bibtex = ""

        abstract_html = (
            f"<div class='abstract-text'>{esc(abstract)}</div>"
            if abstract
            else "<div class='abstract-text'><span class='small-muted'>Abstract not available.</span></div>"
        )

        # Links
        links: List[str] = []
        if paper_id:
            links.append(f"<a href='{esc(ARXIV_ABS.format(paper_id=paper_id))}' target='_blank' rel='noopener'>arXiv</a>")
            links.append(f"<a href='{esc(ARXIV_PDF.format(paper_id=paper_id))}' target='_blank' rel='noopener'>PDF</a>")

        gh = best_effort_github_link(paper, title)
        if gh:
            links.append(f"<a href='{esc(gh)}' target='_blank' rel='noopener'>GitHub</a>")

        # Cite: keep a simple outbound link + optional BibTeX details
        cite_target = ARXIV_ABS.format(paper_id=paper_id) if paper_id else ""
        if cite_target:
            links.append(f"<a href='{esc(cite_target)}' target='_blank' rel='noopener'>Cite</a>")

        link_row = f"<div class='links'>{''.join(links)}</div>" if links else ""

        cite_box = ""
        if bibtex:
            cite_box = (
                "<div class='cite'>"
                "<details class='citebox'>"
                "<summary>Show BibTeX</summary>"
                f"<pre class='bibtex'>{esc(bibtex)}</pre>"
                "</details>"
                "</div>"
            )

        blocks.append(
            "<article class='card'>"
            f"<div class='title'>{idx}. {esc(title)}</div>"
            "<div class='meta'>"
            f"<span class='pill'>Upvotes: {esc(upvotes_text)}</span>"
            f"<span class='pill'>Authors: {esc(authors_display)}</span>"
            "</div>"
            "<div class='abstract'>"
            "<h4>Abstract</h4>"
            f"{abstract_html}"
            "</div>"
            f"{link_row}"
            f"{cite_box}"
            "</article>"
        )

    blocks.append("</div>")

    note = ""
    if diagnostics and rate_headers:
        note = f" RateLimit: {rate_headers}"

    status = f"✅ Loaded {len(items)} paper(s) for {date_iso} (UTC)."
    if cached:
        status += " (cached)"
    if note:
        status += note

    return "\n".join(blocks), status


# =========================
# UI (Gradio 6+ safe)
# =========================

def gradio_major_version() -> int:
    try:
        return int(getattr(gr, "__version__", "0").split(".")[0])
    except Exception:
        return 0


def build_app() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.HTML(
            "<div class='header'>"
            f"<h1>{esc(APP_TITLE)}</h1>"
            f"<div class='subtle'>{esc(APP_SUBTITLE)}</div>"
            "</div>"
        )

        with gr.Group(elem_classes=["controls"]):
            with gr.Row():
                date_in = gr.Textbox(label="Date (YYYY-MM-DD)", value=utc_today_iso(), max_lines=1)
                limit_in = gr.Slider(label="Limit", minimum=1, maximum=50, step=1, value=10)
                fetch_btn = gr.Button("Fetch", variant="primary")
            with gr.Row():
                diag_in = gr.Checkbox(label="Diagnostics (show rate-limit headers / error details)", value=False)

        status_out = gr.Markdown()
        html_out = gr.HTML()

        fetch_btn.click(
            fn=render_cards,
            inputs=[date_in, limit_in, diag_in],
            outputs=[html_out, status_out],
        )

        demo.load(
            fn=render_cards,
            inputs=[date_in, limit_in, diag_in],
            outputs=[html_out, status_out],
        )

    return demo


def launch(demo: gr.Blocks) -> None:
    major = gradio_major_version()
    if major >= 6:
        demo.queue().launch(css=CSS, theme=gr.themes.Soft(), ssr_mode=False)
    else:
        # If you want to support Gradio < 6, pin an older gradio and move css/theme into Blocks.
        demo.queue().launch(ssr_mode=False)


demo = build_app()

if __name__ == "__main__":
    launch(demo)
