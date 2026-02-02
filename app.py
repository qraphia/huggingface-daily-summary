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
# App config
# =========================

APP_TITLE = "Hugging Face Daily Summary"
APP_SUBTITLE = "A clean viewer for Hugging Face Daily Papers. No translation. No keypoints. Abstract-first."

HF_DAILY_PAPERS_API = "https://huggingface.co/api/daily_papers"
ARXIV_API = "https://export.arxiv.org/api/query"

ARXIV_ABS = "https://arxiv.org/abs/{paper_id}"
ARXIV_PDF = "https://arxiv.org/pdf/{paper_id}.pdf"

HTTP_TIMEOUT_S = int(os.getenv("HTTP_TIMEOUT_S", "25"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
BACKOFF_BASE_S = float(os.getenv("BACKOFF_BASE_S", "1.0"))

HF_CACHE_TTL_S = int(os.getenv("HF_CACHE_TTL_S", "300"))      # 5 min
ARXIV_CACHE_TTL_S = int(os.getenv("ARXIV_CACHE_TTL_S", "86400"))  # 24 h

# Optional: set as Space secret to reduce HF rate-limit issues.
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

# Caches
HF_CACHE: Dict[Tuple[str, int], Tuple[float, List[Dict[str, Any]], Dict[str, str]]] = {}
ARXIV_CACHE: Dict[str, Tuple[float, "ArxivMeta"]] = {}

# =========================
# UI styles
# =========================

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

.header{ margin: 6px 0 10px; }
.header h1{ margin: 0 0 0.25rem; font-size: 1.65rem; letter-spacing: -0.01em; }
.subtle{ color: var(--muted); font-size: 0.96rem; line-height: 1.45; }

.panel{
  background: var(--card);
  border: 1px solid var(--bd);
  border-radius: 16px;
  padding: 12px 12px;
  box-shadow: var(--shadow);
  margin-bottom: 10px;
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
  font-weight: 700;
}

.links a:hover{ text-decoration: underline; }

.chips{
  margin-top: 10px;
  display:flex;
  flex-wrap:wrap;
  gap: 6px;
}

.chip{
  background: var(--chip);
  border: 1px solid var(--bd);
  border-radius:999px;
  padding:2px 10px;
  font-size:0.82rem;
  color:#374151;
}

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

.abstract-preview{
  color: var(--text);
  font-size: 0.97rem;
  line-height: 1.62;
  white-space: pre-wrap;
}

details.abstract-full summary{
  cursor: pointer;
  color: #374151;
  font-weight: 800;
  font-size: 0.92rem;
  margin-top: 8px;
}

.abstract-full-text{
  margin-top: 8px;
  color: var(--text);
  font-size: 0.97rem;
  line-height: 1.62;
  white-space: pre-wrap;
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
# Data models
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

# =========================
# Utilities
# =========================

def esc(s: str) -> str:
    return html.escape(s or "", quote=True)

def utc_today_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()

def normalize_text(s: str) -> str:
    if not s:
        return ""
    out = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    # remove occasional "L1:" markers seen in some HF fields
    for i in range(1, 120):
        out = out.replace(f"L{i}:", "")
    while "\n\n\n" in out:
        out = out.replace("\n\n\n", "\n\n")
    return out.strip()

def extract_rate_limit_headers(headers: Dict[str, str]) -> Dict[str, str]:
    keep: Dict[str, str] = {}
    for k, v in (headers or {}).items():
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
            last_err = ApiError(None, type(e).__name__, str(e)[:500], {})
            time.sleep(min(20.0, BACKOFF_BASE_S * (2 ** attempt)))
            continue

        if r.status_code == 200:
            try:
                return r.json(), None, extract_rate_limit_headers(last_headers)
            except Exception:
                return None, ApiError(200, "Invalid JSON", (r.text or "")[:500], extract_rate_limit_headers(last_headers)), extract_rate_limit_headers(last_headers)

        # retryable
        if r.status_code in (429, 500, 502, 503, 504):
            retry_after = r.headers.get("Retry-After")
            wait_s = min(20.0, BACKOFF_BASE_S * (2 ** attempt))
            if retry_after:
                try:
                    wait_s = max(wait_s, float(retry_after))
                except Exception:
                    pass
            last_err = ApiError(r.status_code, r.reason or "HTTP error", (r.text or "")[:500], extract_rate_limit_headers(last_headers))
            time.sleep(wait_s)
            continue

        # non-retryable
        return None, ApiError(r.status_code, r.reason or "HTTP error", (r.text or "")[:500], extract_rate_limit_headers(last_headers)), extract_rate_limit_headers(last_headers)

    return None, last_err, extract_rate_limit_headers(last_headers)

# =========================
# HF Daily Papers
# =========================

def fetch_hf_daily(date_iso: str, limit: int) -> Tuple[Optional[List[Dict[str, Any]]], Optional[ApiError], Dict[str, str], bool]:
    key = (date_iso, int(limit))
    now = time.time()

    hit = HF_CACHE.get(key)
    if hit and (now - hit[0] < HF_CACHE_TTL_S):
        return hit[1], None, hit[2], True

    data, err, rate_headers = safe_get_json(
        HF_DAILY_PAPERS_API,
        {"date": date_iso, "limit": int(limit), "sort": "trending"},
    )
    if err is not None:
        return None, err, rate_headers, False

    if not isinstance(data, list):
        return None, ApiError(200, "Unexpected payload", str(type(data))[:200], rate_headers), rate_headers, False

    HF_CACHE[key] = (now, data, rate_headers)
    return data, None, rate_headers, False

# =========================
# arXiv batch fetch
# =========================

def _cache_get_arxiv(aid: str) -> Optional[ArxivMeta]:
    now = time.time()
    hit = ARXIV_CACHE.get(aid)
    if hit and (now - hit[0] < ARXIV_CACHE_TTL_S):
        return hit[1]
    return None

def _cache_put_arxiv(meta: ArxivMeta) -> None:
    ARXIV_CACHE[meta.arxiv_id] = (time.time(), meta)

def _chunk(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]

def _parse_arxiv_atom(xml_text: str) -> Dict[str, ArxivMeta]:
    out: Dict[str, ArxivMeta] = {}
    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return out

    for entry in root.findall("a:entry", ns):
        raw_id = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
        # raw_id: http://arxiv.org/abs/XXXX.XXXXXvN
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1].strip() if "/abs/" in raw_id else ""
        title = normalize_text(entry.findtext("a:title", default="", namespaces=ns) or "")
        abstract = normalize_text(entry.findtext("a:summary", default="", namespaces=ns) or "")
        published = (entry.findtext("a:published", default="", namespaces=ns) or "").strip()
        year = published[:4] if len(published) >= 4 else ""

        authors: List[str] = []
        for a in entry.findall("a:author", ns):
            name = (a.findtext("a:name", default="", namespaces=ns) or "").strip()
            if name:
                authors.append(name)

        if arxiv_id:
            out[arxiv_id] = ArxivMeta(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
            )
    return out

def fetch_arxiv_batch(arxiv_ids: List[str]) -> Dict[str, ArxivMeta]:
    # Return cached + newly fetched meta in a map.
    wanted = []
    result: Dict[str, ArxivMeta] = {}

    for aid in arxiv_ids:
        aid = (aid or "").strip()
        if not aid:
            continue
        cached = _cache_get_arxiv(aid)
        if cached:
            result[aid] = cached
        else:
            wanted.append(aid)

    if not wanted:
        return result

    # arXiv allows id_list with comma-separated ids. Keep chunks conservative.
    for group in _chunk(wanted, 25):
        params = {"id_list": ",".join(group)}
        last_err: Optional[str] = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                r = SESSION.get(ARXIV_API, params=params, timeout=HTTP_TIMEOUT_S)
                if r.status_code == 200:
                    parsed = _parse_arxiv_atom(r.text or "")
                    for k, meta in parsed.items():
                        result[k] = meta
                        _cache_put_arxiv(meta)
                    break
                last_err = f"HTTP {r.status_code}"
            except requests.RequestException as e:
                last_err = type(e).__name__
            time.sleep(min(10.0, BACKOFF_BASE_S * (2 ** attempt)))

        # If a group fails, we simply skip; UI will fallback to HF summary.
        _ = last_err

    return result

# =========================
# Presentation helpers
# =========================

def format_authors(names: List[str], max_names: int = 8) -> str:
    names = [n.strip() for n in (names or []) if n and n.strip()]
    if not names:
        return "—"
    if len(names) <= max_names:
        return ", ".join(names)
    return ", ".join(names[:max_names]) + f", et al. (+{len(names) - max_names})"

def get_hf_authors(paper: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    authors = paper.get("authors") or []
    if isinstance(authors, list):
        for a in authors:
            if isinstance(a, dict) and a.get("name"):
                out.append(str(a["name"]).strip())
    return out

def get_keywords(paper: Dict[str, Any]) -> List[str]:
    kws = paper.get("ai_keywords") or []
    if isinstance(kws, list):
        return [str(k).strip() for k in kws if str(k).strip()]
    return []

def best_github_url(paper: Dict[str, Any], title: str) -> Tuple[str, bool]:
    # Returns (url, is_fallback_search)
    candidates = [
        "github_url", "githubUrl", "code_url", "codeUrl", "repo_url", "repoUrl",
        "repository_url", "repositoryUrl", "source_code_url", "sourceCodeUrl",
        "implementation_url", "implementationUrl",
    ]
    for k in candidates:
        v = paper.get(k)
        if isinstance(v, str) and v.strip().startswith("http") and "github.com" in v:
            return v.strip(), False

    for k in ("links", "resources", "artifacts"):
        v = paper.get(k)
        if isinstance(v, list):
            for it in v:
                if isinstance(it, dict):
                    u = it.get("url") or it.get("href")
                    if isinstance(u, str) and "github.com" in u:
                        return u.strip(), False
        if isinstance(v, dict):
            for _, u in v.items():
                if isinstance(u, str) and "github.com" in u:
                    return u.strip(), False

    q = urllib.parse.quote_plus((title or "").strip()[:180])
    return f"https://github.com/search?q={q}&type=repositories", True

def abstract_preview(full: str, max_chars: int = 520) -> Tuple[str, bool]:
    s = (full or "").strip()
    if not s:
        return "", False
    if len(s) <= max_chars:
        return s, False
    return s[: max_chars - 1].rstrip() + "…", True

# =========================
# Render
# =========================

def apply_filter(items: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    q = (query or "").strip().lower()
    if not q:
        return items

    out: List[Dict[str, Any]] = []
    for it in items:
        paper = (it or {}).get("paper") or {}
        title = str(it.get("title") or paper.get("title") or "")
        authors = " ".join(get_hf_authors(paper))
        keywords = " ".join(get_keywords(paper))
        hay = f"{title}\n{authors}\n{keywords}".lower()
        if q in hay:
            out.append(it)
    return out

def apply_sort(items: List[Dict[str, Any]], sort_mode: str) -> List[Dict[str, Any]]:
    if sort_mode == "API order":
        return items

    def upvotes(it: Dict[str, Any]) -> int:
        paper = (it or {}).get("paper") or {}
        v = paper.get("upvotes")
        try:
            return int(v)
        except Exception:
            return -1

    def title_key(it: Dict[str, Any]) -> str:
        paper = (it or {}).get("paper") or {}
        t = str(it.get("title") or paper.get("title") or "")
        return t.lower()

    if sort_mode == "Upvotes (desc)":
        return sorted(items, key=upvotes, reverse=True)
    if sort_mode == "Title (A→Z)":
        return sorted(items, key=title_key)
    return items

def render(
    date_iso: str,
    limit: int,
    query: str,
    sort_mode: str,
    show_keywords: bool,
    expand_abstracts: bool,
    diagnostics: bool,
) -> Tuple[str, str]:
    date_iso = (date_iso or "").strip()
    try:
        dt.date.fromisoformat(date_iso)
    except Exception:
        return "", "❌ Invalid date. Use YYYY-MM-DD."

    limit = max(1, min(50, int(limit)))

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
                    "hint: set HF_TOKEN in Space secrets to reduce rate-limit errors (429).",
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
        return box, "❌ Failed to fetch daily papers."

    assert items is not None
    items = apply_filter(items, query)
    items = apply_sort(items, sort_mode)

    if not items:
        return "<div class='paper-list'></div>", "No results."

    # Batch fetch arXiv abstracts (best-effort). Fallback remains HF summary.
    arxiv_ids: List[str] = []
    for it in items:
        paper = (it or {}).get("paper") or {}
        pid = str(paper.get("id") or it.get("id") or "").strip()
        if pid:
            arxiv_ids.append(pid)
    arxiv_meta_map = fetch_arxiv_batch(arxiv_ids) if arxiv_ids else {}

    blocks: List[str] = ["<div class='paper-list'>"]
    for idx, it in enumerate(items, start=1):
        paper = (it or {}).get("paper") or {}

        pid = str(paper.get("id") or it.get("id") or "").strip()
        title = str(it.get("title") or paper.get("title") or "").strip() or "(no title)"
        upvotes = paper.get("upvotes")
        upvotes_text = str(upvotes) if upvotes is not None else "—"

        meta = arxiv_meta_map.get(pid)
        authors = meta.authors if (meta and meta.authors) else get_hf_authors(paper)
        authors_text = format_authors(authors)

        # Abstract: prefer arXiv; fallback to HF summary
        abstract_full = ""
        if meta and meta.abstract:
            abstract_full = meta.abstract
        else:
            abstract_full = normalize_text(str(it.get("summary") or paper.get("summary") or ""))

        preview, truncated = abstract_preview(abstract_full, max_chars=520)
        preview_html = (
            f"<div class='abstract-preview'>{esc(preview)}</div>"
            if preview
            else "<div class='abstract-preview'><span class='small-muted'>Abstract not available.</span></div>"
        )

        full_html = ""
        if truncated:
            open_attr = " open" if expand_abstracts else ""
            full_html = (
                f"<details class='abstract-full'{open_attr}>"
                "<summary>Show full abstract</summary>"
                f"<div class='abstract-full-text'>{esc(abstract_full)}</div>"
                "</details>"
            )

        # Links: arXiv, PDF, GitHub (no Cite)
        links: List[str] = []
        if pid:
            links.append(f"<a href='{esc(ARXIV_ABS.format(paper_id=pid))}' target='_blank' rel='noopener'>arXiv</a>")
            links.append(f"<a href='{esc(ARXIV_PDF.format(paper_id=pid))}' target='_blank' rel='noopener'>PDF</a>")

        gh_url, gh_is_search = best_github_url(paper, title)
        gh_label = "GitHub" if not gh_is_search else "GitHub (search)"
        links.append(f"<a href='{esc(gh_url)}' target='_blank' rel='noopener'>{gh_label}</a>")
        link_row = f"<div class='links'>{''.join(links)}</div>"

        # Keywords
        kw_html = ""
        if show_keywords:
            kws = get_keywords(paper)
            if kws:
                chips = "".join([f"<span class='chip'>{esc(k)}</span>" for k in kws[:18]])
                kw_html = f"<div class='chips'>{chips}</div>"

        blocks.append(
            "<article class='card'>"
            f"<div class='title'>{idx}. {esc(title)}</div>"
            "<div class='meta'>"
            f"<span class='pill'>Upvotes: {esc(upvotes_text)}</span>"
            f"<span class='pill'>Authors: {esc(authors_text)}</span>"
            "</div>"
            "<div class='abstract'>"
            "<h4>Abstract</h4>"
            f"{preview_html}"
            f"{full_html}"
            "</div>"
            f"{kw_html}"
            f"{link_row}"
            "</article>"
        )

    blocks.append("</div>")

    status = f"✅ Loaded {len(items)} paper(s) for {date_iso} (UTC)."
    if cached:
        status += " (cached)"
    if diagnostics and rate_headers:
        status += f" RateLimit: {rate_headers}"

    return "\n".join(blocks), status

# =========================
# Gradio UI
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

        with gr.Group(elem_classes=["panel"]):
            with gr.Row():
                date_in = gr.Textbox(label="Date (YYYY-MM-DD)", value=utc_today_iso(), max_lines=1)
                limit_in = gr.Slider(label="Limit", minimum=1, maximum=50, step=1, value=10)
                fetch_btn = gr.Button("Fetch", variant="primary")

            with gr.Row():
                query_in = gr.Textbox(label="Search (title / authors / keywords)", value="", max_lines=1)
                sort_in = gr.Dropdown(
                    label="Sort",
                    choices=["API order", "Upvotes (desc)", "Title (A→Z)"],
                    value="API order",
                )

            with gr.Row():
                show_kw_in = gr.Checkbox(label="Show keywords", value=False)
                expand_abs_in = gr.Checkbox(label="Auto-expand full abstracts", value=False)
                diag_in = gr.Checkbox(label="Diagnostics (show rate-limit headers / error details)", value=False)

        status_out = gr.Markdown()
        html_out = gr.HTML()

        fetch_btn.click(
            fn=render,
            inputs=[date_in, limit_in, query_in, sort_in, show_kw_in, expand_abs_in, diag_in],
            outputs=[html_out, status_out],
        )

        demo.load(
            fn=render,
            inputs=[date_in, limit_in, query_in, sort_in, show_kw_in, expand_abs_in, diag_in],
            outputs=[html_out, status_out],
        )

    return demo

def launch(demo: gr.Blocks) -> None:
    major = gradio_major_version()
    if major >= 6:
        demo.queue().launch(css=CSS, theme=gr.themes.Soft(), ssr_mode=False)
    else:
        # If you truly need Gradio < 6 support, pin it and move css/theme into Blocks().
        demo.queue().launch(ssr_mode=False)

demo = build_app()

if __name__ == "__main__":
    launch(demo)
