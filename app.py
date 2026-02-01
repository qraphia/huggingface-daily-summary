import datetime as dt
import html
import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import gradio as gr
import requests

APP_TITLE = "HuggingFace Daily Summary"
APP_DESCRIPTION = (
    "Fetch Hugging Face Daily Papers for a given UTC date and show heuristic key points."
)

DAILY_PAPERS_API = "https://huggingface.co/api/daily_papers"
HF_PAPER_URL = "https://huggingface.co/papers/{paper_id}"
ARXIV_ABS_URL = "https://arxiv.org/abs/{paper_id}"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{paper_id}.pdf"

REQUEST_TIMEOUT_S = 30
DEFAULT_LIMIT = 5
MAX_LIMIT = 20
DEFAULT_POINTS = 4
MAX_POINTS = 6
MAX_KEYWORDS = 12

CSS = """
:root { --card-bd:#e5e7eb; --muted:#6b7280; }
.container { max-width: 980px; margin: 0 auto; }
.card {
  border: 1px solid var(--card-bd);
  border-radius: 14px;
  padding: 14px 14px;
  margin: 12px 0;
  background: white;
}
.title { font-size: 1.05rem; font-weight: 700; margin: 0 0 6px; line-height: 1.35; }
.meta { color: var(--muted); font-size: 0.92rem; margin: 0 0 8px; }
.links a { margin-right: 10px; font-size: 0.92rem; text-decoration: none; }
.badges { margin: 8px 0 10px; }
.badge {
  display: inline-block;
  padding: 2px 8px;
  margin: 0 6px 6px 0;
  border-radius: 999px;
  background: #f3f4f6;
  border: 1px solid var(--card-bd);
  font-size: 0.82rem;
}
.section { margin-top: 10px; }
.section h4 { margin: 8px 0 6px; font-size: 0.95rem; }
ul { margin: 6px 0 0 18px; }
small.note { color: var(--muted); }
"""

_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": "huggingface-daily-summary/2.0 (+https://huggingface.co/spaces/Qraphia/huggingface-daily-summary)"
    }
)

_RE_LINE_MARKER = re.compile(r"\bL\d+:\s*")
_RE_BULLET_LINE = re.compile(r"^\s*(?:[-*•–]|(\d+)[\.\)]|\((\d+)\))\s+")
_RE_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def utc_today_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def esc(text: str) -> str:
    return html.escape(text or "", quote=True)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = _RE_LINE_MARKER.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def extract_key_points(text: str, max_points: int) -> List[str]:
    """
    Heuristic extraction:
      1) If bullet-like lines exist, take the first max_points of them.
      2) Else, split into sentences and take the first max_points sentences.
    """
    text = normalize_text(text)
    if not text:
        return []

    # Prefer bullet-like lines if present
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bullet_lines: List[str] = []
    for ln in lines:
        if _RE_BULLET_LINE.search(ln):
            ln = _RE_BULLET_LINE.sub("", ln).strip()
            if ln:
                bullet_lines.append(ln)

    if bullet_lines:
        return [trim_point(p) for p in bullet_lines[:max_points]]

    # Fallback: sentence-ish split
    sents = [s.strip() for s in _RE_SENT_SPLIT.split(text) if s.strip()]
    if not sents:
        return []
    return [trim_point(s) for s in sents[:max_points]]


def trim_point(text: str, max_len: int = 260) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def format_authors(authors_any: Any, max_names: int = 8) -> Tuple[str, str]:
    """
    Returns (short, full). Accepts the API's typical list-of-dicts authors format.
    """
    if not isinstance(authors_any, list):
        return ("", "")
    names = []
    for a in authors_any:
        if isinstance(a, dict):
            n = (a.get("name") or "").strip()
            if n:
                names.append(n)

    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for n in names:
        if n not in seen:
            uniq.append(n)
            seen.add(n)

    if not uniq:
        return ("", "")

    full = ", ".join(uniq)
    if len(uniq) <= max_names:
        return (full, full)

    short = ", ".join(uniq[:max_names]) + f", et al. (+{len(uniq) - max_names})"
    return (short, full)


@lru_cache(maxsize=64)
def fetch_daily_papers(date_iso: str, limit: int) -> List[Dict[str, Any]]:
    params = {"date": date_iso, "limit": int(limit), "sort": "trending"}
    r = _SESSION.get(DAILY_PAPERS_API, params=params, timeout=REQUEST_TIMEOUT_S)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def render_card(
    idx: int,
    paper_id: str,
    title: str,
    upvotes: Any,
    authors_short: str,
    authors_full: str,
    keywords: Any,
    points: List[str],
) -> str:
    paper_id = (paper_id or "").strip()
    hf_url = HF_PAPER_URL.format(paper_id=paper_id) if paper_id else ""
    arxiv_abs = ARXIV_ABS_URL.format(paper_id=paper_id) if paper_id else ""
    arxiv_pdf = ARXIV_PDF_URL.format(paper_id=paper_id) if paper_id else ""

    links = []
    if hf_url:
        links.append(f"<a href='{esc(hf_url)}' target='_blank' rel='noopener'>HF Paper</a>")
    if arxiv_abs:
        links.append(f"<a href='{esc(arxiv_abs)}' target='_blank' rel='noopener'>arXiv</a>")
    if arxiv_pdf:
        links.append(f"<a href='{esc(arxiv_pdf)}' target='_blank' rel='noopener'>PDF</a>")

    meta_bits = []
    if upvotes is not None:
        meta_bits.append(f"▲ {esc(str(upvotes))}")
    if authors_short:
        meta_bits.append(esc(authors_short))

    badges = ""
    if isinstance(keywords, list) and keywords:
        badges = "".join(
            [f"<span class='badge'>{esc(str(k))}</span>" for k in keywords[:MAX_KEYWORDS]]
        )

    points_html = ""
    if points:
        li = "".join([f"<li>{esc(p)}</li>" for p in points])
        points_html = f"<div class='section'><h4>Key points</h4><ul>{li}</ul></div>"

    authors_details = ""
    if authors_full and (len(authors_full) > len(authors_short) + 10):
        authors_details = (
            "<details style='margin-top:10px;'>"
            "<summary><small>Full authors</small></summary>"
            f"<div class='meta' style='margin-top:6px;'>{esc(authors_full)}</div>"
            "</details>"
        )

    return (
        "<div class='card'>"
        f"<div class='title'>{idx}. {esc(title) if title else '(no title)'}</div>"
        f"<div class='meta'>{' · '.join(meta_bits)}</div>" if meta_bits else ""
    ) + (
        f"<div class='links'>{' '.join(links)}</div>" if links else ""
    ) + (
        f"<div class='badges'>{badges}</div>" if badges else ""
    ) + (
        points_html
    ) + (
        authors_details
    ) + "</div>"


def build_view(date_iso: str, limit: int, points_per_paper: int) -> Tuple[str, str]:
    date_iso = (date_iso or "").strip()
    try:
        dt.date.fromisoformat(date_iso)
    except Exception:
        return ("", f"❌ Invalid date: `{date_iso}`. Use `YYYY-MM-DD` (UTC).")

    limit = int(limit)
    limit = max(1, min(MAX_LIMIT, limit))

    points_per_paper = int(points_per_paper)
    points_per_paper = max(1, min(MAX_POINTS, points_per_paper))

    try:
        items = fetch_daily_papers(date_iso, limit)
    except requests.HTTPError as e:
        return ("", f"❌ Request failed (HTTP): `{str(e)}`")
    except requests.RequestException as e:
        return ("", f"❌ Request failed: `{type(e).__name__}`")

    if not items:
        empty = (
            "<div class='container'>"
            "<p class='meta'>No papers returned for that date. "
            "Try another UTC date.</p>"
            "</div>"
        )
        return (empty, f"0 papers for `{date_iso}` (UTC).")

    cards: List[str] = ["<div class='container'>"]
    for idx, it in enumerate(items, start=1):
        paper = (it or {}).get("paper") or {}

        paper_id = str(paper.get("id") or it.get("id") or "").strip()
        title = (it.get("title") or paper.get("title") or "").strip()

        summary = it.get("summary") or paper.get("summary") or ""
        ai_summary = paper.get("ai_summary") or ""
        text_for_points = summary or ai_summary or ""

        points = extract_key_points(text_for_points, max_points=points_per_paper)

        upvotes = paper.get("upvotes")
        keywords = paper.get("ai_keywords") or []
        authors_short, authors_full = format_authors(paper.get("authors") or [])

        cards.append(
            render_card(
                idx=idx,
                paper_id=paper_id,
                title=title,
                upvotes=upvotes,
                authors_short=authors_short,
                authors_full=authors_full,
                keywords=keywords,
                points=points,
            )
        )

    cards.append("</div>")
    status = f"✅ Loaded {len(items)} papers for `{date_iso}` (UTC)."
    return ("\n".join(cards), status)


with gr.Blocks() as demo:
    gr.Markdown(f"# {APP_TITLE}\n{APP_DESCRIPTION}")

    with gr.Row():
        date_in = gr.Textbox(
            label="Date (YYYY-MM-DD, UTC)",
            value=utc_today_iso(),
            max_lines=1,
        )
        limit_in = gr.Slider(
            label="Limit",
            minimum=1,
            maximum=MAX_LIMIT,
            step=1,
            value=DEFAULT_LIMIT,
        )
        points_in = gr.Slider(
            label="Key points per paper",
            minimum=1,
            maximum=MAX_POINTS,
            step=1,
            value=DEFAULT_POINTS,
        )
        run_btn = gr.Button("Run", variant="primary")

    status_out = gr.Markdown()
    html_out = gr.HTML()

    run_btn.click(
        fn=build_view,
        inputs=[date_in, limit_in, points_in],
        outputs=[html_out, status_out],
    )

demo.launch(theme=gr.themes.Soft(), css=CSS, ssr_mode=False)
