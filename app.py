import datetime as dt
import html
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import gradio as gr
import requests


APP_TITLE = "Hugging Face Daily Papers — Clean Viewer"

DAILY_PAPERS_API = "https://huggingface.co/api/daily_papers"
HF_PAPER_URL = "https://huggingface.co/papers/{paper_id}"
ARXIV_ABS_URL = "https://arxiv.org/abs/{paper_id}"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{paper_id}.pdf"

TOKYO_TZ = ZoneInfo("Asia/Tokyo")

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))  # 5 minutes
HTTP_TIMEOUT_SECONDS = int(os.getenv("HTTP_TIMEOUT_SECONDS", "25"))

_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": "qraphia/huggingface-daily-summary (clean; gradio)",
        "Accept": "application/json",
    }
)

_CACHE: Dict[Tuple[str, int], Tuple[float, List[Dict[str, Any]]]] = {}

CSS = """
:root{
  --bg: #f7f8fb;
  --card: #ffffff;
  --bd: #e6e8ef;
  --text: #111827;
  --muted: #6b7280;
  --chip: #f3f4f6;
  --link: #2563eb;
}

.gradio-container{
  max-width: 1040px !important;
}

body{
  background: var(--bg);
}

.header h1{
  margin-bottom: 0.2rem;
}

.subtle{
  color: var(--muted);
  font-size: 0.95rem;
  line-height: 1.4;
}

.paper-list{
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 8px;
}

.card{
  background: var(--card);
  border: 1px solid var(--bd);
  border-radius: 16px;
  padding: 14px 14px;
}

.card:hover{
  border-color: #d9ddea;
}

.title{
  font-size: 1.04rem;
  font-weight: 700;
  color: var(--text);
  text-decoration: none;
}

.title:hover{
  text-decoration: underline;
}

.meta{
  margin-top: 6px;
  color: var(--muted);
  font-size: 0.92rem;
  display: flex;
  flex-wrap: wrap;
  gap: 8px 10px;
  align-items: center;
}

.pill{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 2px 10px;
  background: var(--chip);
  border: 1px solid var(--bd);
  border-radius: 999px;
  color: #374151;
  font-size: 0.85rem;
}

.links{
  margin-top: 8px;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.links a{
  color: var(--link);
  text-decoration: none;
  font-size: 0.92rem;
}

.links a:hover{
  text-decoration: underline;
}

.chips{
  margin-top: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.chip{
  background: var(--chip);
  border: 1px solid var(--bd);
  border-radius: 999px;
  padding: 2px 10px;
  font-size: 0.82rem;
  color: #374151;
}

details.abstract{
  margin-top: 10px;
  border-top: 1px dashed var(--bd);
  padding-top: 10px;
}

details.abstract summary{
  cursor: pointer;
  font-weight: 650;
  color: #111827;
  font-size: 0.95rem;
}

.abstract-text{
  margin-top: 8px;
  color: #111827;
  font-size: 0.95rem;
  line-height: 1.55;
  white-space: pre-wrap;
}

.small-muted{
  color: var(--muted);
  font-size: 0.86rem;
}
"""


def today_in_tokyo_iso() -> str:
    return dt.datetime.now(TOKYO_TZ).date().isoformat()


def escape(s: str) -> str:
    return html.escape(s or "", quote=True)


def normalize_abstract(text: str) -> str:
    if not text:
        return ""
    # HF daily_papers sometimes includes "L1:" line markers
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = out.strip()
    # remove "L1:" / "L2:" markers at word boundaries
    # (avoid importing re to keep this file compact)
    # simple loop is fine because these markers are short
    for i in range(1, 50):
        out = out.replace(f"L{i}:", "")
    # normalize extra blank lines
    while "\n\n\n" in out:
        out = out.replace("\n\n\n", "\n\n")
    return out.strip()


def format_authors(authors: List[Dict[str, Any]], max_names: int = 8) -> Tuple[str, str]:
    names = [str(a.get("name", "")).strip() for a in (authors or []) if a.get("name")]
    if not names:
        return "", ""
    short = ", ".join(names[:max_names])
    if len(names) > max_names:
        short = f"{short}, et al. (+{len(names) - max_names})"
    full = ", ".join(names)
    return short, full


def fetch_daily_papers(date_str: str, limit: int) -> List[Dict[str, Any]]:
    key = (date_str, int(limit))
    now = time.time()

    hit = _CACHE.get(key)
    if hit and (now - hit[0] < CACHE_TTL_SECONDS):
        return hit[1]

    params = {"date": date_str, "limit": int(limit), "sort": "trending"}
    r = _SESSION.get(DAILY_PAPERS_API, params=params, timeout=HTTP_TIMEOUT_SECONDS)
    r.raise_for_status()

    data = r.json()
    if not isinstance(data, list):
        data = []

    _CACHE[key] = (now, data)
    return data


def apply_query_filter(items: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    q = (query or "").strip().lower()
    if not q:
        return items

    out: List[Dict[str, Any]] = []
    for it in items:
        paper = (it or {}).get("paper") or {}
        title = str(it.get("title") or paper.get("title") or "")
        abstract = str(it.get("summary") or paper.get("summary") or "")
        authors = paper.get("authors") or []
        author_text = " ".join([str(a.get("name", "")).lower() for a in authors if a.get("name")])
        keywords = paper.get("ai_keywords") or []
        kw_text = " ".join([str(k).lower() for k in keywords])

        hay = f"{title}\n{abstract}\n{author_text}\n{kw_text}".lower()
        if q in hay:
            out.append(it)
    return out


def render_cards(
    date_str: str,
    limit: int,
    query: str,
    show_keywords: bool,
    show_full_authors: bool,
) -> Tuple[str, str]:
    try:
        dt.date.fromisoformat(date_str)
    except Exception:
        return "", f"❌ Invalid date: `{date_str}`. Use YYYY-MM-DD."

    try:
        items = fetch_daily_papers(date_str, limit)
    except requests.HTTPError as e:
        return "", f"❌ API error: {type(e).__name__}"
    except Exception as e:
        return "", f"❌ Unexpected error: {type(e).__name__}"

    items = apply_query_filter(items, query)
    if not items:
        return "<div class='paper-list'></div>", "No results."

    blocks: List[str] = ["<div class='paper-list'>"]

    for idx, it in enumerate(items, start=1):
        paper = (it or {}).get("paper") or {}

        paper_id = str(paper.get("id") or it.get("id") or "").strip()
        title = str(it.get("title") or paper.get("title") or "").strip() or "(no title)"
        upvotes = paper.get("upvotes")
        upvotes_text = str(upvotes) if upvotes is not None else "—"

        authors_short, authors_full = format_authors(paper.get("authors") or [])
        abstract = normalize_abstract(str(it.get("summary") or paper.get("summary") or ""))

        hf_url = HF_PAPER_URL.format(paper_id=paper_id) if paper_id else ""
        arxiv_abs = ARXIV_ABS_URL.format(paper_id=paper_id) if paper_id else ""
        arxiv_pdf = ARXIV_PDF_URL.format(paper_id=paper_id) if paper_id else ""

        title_link = hf_url if hf_url else (arxiv_abs if arxiv_abs else "")
        title_html = escape(title)

        links: List[str] = []
        if hf_url:
            links.append(f"<a href='{escape(hf_url)}' target='_blank' rel='noopener'>HF Paper</a>")
        if arxiv_abs:
            links.append(f"<a href='{escape(arxiv_abs)}' target='_blank' rel='noopener'>arXiv</a>")
        if arxiv_pdf:
            links.append(f"<a href='{escape(arxiv_pdf)}' target='_blank' rel='noopener'>PDF</a>")

        link_row = f"<div class='links'>{''.join(links)}</div>" if links else ""

        kw_html = ""
        if show_keywords:
            keywords = paper.get("ai_keywords") or []
            if isinstance(keywords, list) and keywords:
                chips = "".join([f"<span class='chip'>{escape(str(k))}</span>" for k in keywords[:18]])
                kw_html = f"<div class='chips'>{chips}</div>"

        author_html = ""
        if authors_short:
            author_html = f"<span class='pill'>Authors: {escape(authors_short)}</span>"

        full_authors_html = ""
        if show_full_authors and authors_full and authors_full != authors_short:
            full_authors_html = (
                "<div style='margin-top:8px;'>"
                f"<div class='small-muted'>Full authors</div>"
                f"<div class='small-muted'>{escape(authors_full)}</div>"
                "</div>"
            )

        abstract_html = (
            f"<div class='abstract-text'>{escape(abstract)}</div>"
            if abstract
            else "<div class='abstract-text'><span class='small-muted'>No abstract provided.</span></div>"
        )

        if title_link:
            title_node = f"<a class='title' href='{escape(title_link)}' target='_blank' rel='noopener'>{idx}. {title_html}</a>"
        else:
            title_node = f"<span class='title'>{idx}. {title_html}</span>"

        blocks.append(
            "<article class='card'>"
            f"{title_node}"
            "<div class='meta'>"
            f"<span class='pill'>Upvotes: {escape(upvotes_text)}</span>"
            f"{author_html}"
            "</div>"
            f"{link_row}"
            f"{kw_html}"
            "<details class='abstract' open>"
            "<summary>Abstract</summary>"
            f"{abstract_html}"
            f"{full_authors_html}"
            "</details>"
            "</article>"
        )

    blocks.append("</div>")
    status = f"Showing {len(items)} paper(s) for {date_str}."
    if query.strip():
        status += f" Filter: “{query.strip()}”."
    return "\n".join(blocks), status


with gr.Blocks(title=APP_TITLE, css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML(
        "<div class='header'>"
        f"<h1>{escape(APP_TITLE)}</h1>"
        "<div class='subtle'>"
        "A slim viewer for Hugging Face Daily Papers. No translation, no heuristics: show the full abstract."
        "</div>"
        "</div>"
    )

    with gr.Row():
        date_in = gr.Textbox(label="Date (YYYY-MM-DD)", value=today_in_tokyo_iso(), max_lines=1)
        limit_in = gr.Slider(label="Limit", minimum=1, maximum=30, step=1, value=8)

    with gr.Row():
        query_in = gr.Textbox(label="Search (title / abstract / authors / keywords)", value="", max_lines=1)
    with gr.Row():
        show_kw_in = gr.Checkbox(label="Show keywords", value=True)
        show_full_authors_in = gr.Checkbox(label="Show full author list", value=False)

    with gr.Row():
        fetch_btn = gr.Button("Fetch", variant="primary")
        clear_btn = gr.Button("Clear search")

    status_out = gr.Markdown()
    html_out = gr.HTML()

    fetch_btn.click(
        fn=render_cards,
        inputs=[date_in, limit_in, query_in, show_kw_in, show_full_authors_in],
        outputs=[html_out, status_out],
    )

    clear_btn.click(
        fn=lambda: "",
        inputs=[],
        outputs=[query_in],
    ).then(
        fn=render_cards,
        inputs=[date_in, limit_in, query_in, show_kw_in, show_full_authors_in],
        outputs=[html_out, status_out],
    )

    demo.load(
        fn=render_cards,
        inputs=[date_in, limit_in, query_in, show_kw_in, show_full_authors_in],
        outputs=[html_out, status_out],
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch()
