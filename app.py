import os
import re
import json
import time
import uuid
import logging
import datetime as dt
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from html import escape as html_escape

import numpy as np
import pandas as pd
import gradio as gr

from huggingface_hub import HfApi, InferenceClient
from sklearn.feature_extraction.text import TfidfVectorizer

# =============================
# Basic config
# =============================
APP_TITLE = "Hugging Face Daily Summary"
TAGLINE = "Daily Papers → extractive key points → optional Japanese (HF Inference translation)"
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)

ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(v\d+)?\b")

DEBUG = os.getenv("DEBUG", "0") == "1"  # set in Space Variables only while debugging
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()  # Space Secret (recommended)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s pid=%(process)d %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)

api = HfApi()


# =============================
# Utilities
# =============================
def rid() -> str:
    return uuid.uuid4().hex[:8]


def to_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if hasattr(x, "__dict__"):
        return dict(x.__dict__)
    return {"value": str(x)}


def first_nonempty(d: Dict[str, Any], keys: List[str], default: Any = "") -> Any:
    for k in keys:
        v = d.get(k, None)
        if v is None:
            continue
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and len(v) > 0:
            return v
    return default


def normalize_authors(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        out = []
        for a in v:
            if isinstance(a, str):
                s = a.strip()
                if s:
                    out.append(s)
            elif isinstance(a, dict):
                s = (a.get("name") or a.get("full_name") or a.get("author") or "").strip()
                if s:
                    out.append(s)
            else:
                s = str(a).strip()
                if s:
                    out.append(s)
        return ", ".join(out)
    return str(v).strip()


def guess_arxiv_id(p: Dict[str, Any]) -> Optional[str]:
    for k in ["id", "paper_id", "arxiv_id", "arxivId"]:
        v = p.get(k)
        if isinstance(v, str):
            m = ARXIV_ID_RE.search(v)
            if m:
                return m.group(0)

    for k in ["url", "paper_url", "hf_url", "link"]:
        v = p.get(k)
        if isinstance(v, str):
            m = ARXIV_ID_RE.search(v)
            if m:
                return m.group(0)
            m2 = re.search(r"/papers/([^/?#]+)", v)
            if m2 and ARXIV_ID_RE.match(m2.group(1)):
                return m2.group(1)

    m = ARXIV_ID_RE.search(json.dumps(p, ensure_ascii=False))
    return m.group(0) if m else None


def hf_paper_url(arxiv_id: Optional[str]) -> str:
    return f"https://huggingface.co/papers/{arxiv_id}" if arxiv_id else ""


def arxiv_abs_url(arxiv_id: Optional[str]) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""


def arxiv_pdf_url(arxiv_id: Optional[str]) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""


def best_text_for_summary(p: Dict[str, Any]) -> str:
    return first_nonempty(
        p,
        ["summary", "abstract", "paperAbstract", "description", "content", "text"],
        default="",
    ) or ""


# =============================
# Extractive summarization (robust)
# =============================
def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def extractive_bullets(text: str, k: int = 4) -> List[str]:
    sents = split_sentences(text)
    if len(sents) <= k:
        return sents
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(sents)  # (n_sents, n_terms) sparse
    centroid = np.asarray(X.mean(axis=0)).ravel()  # 1D dense
    scores = (X @ centroid).ravel()  # 1D dense; avoids .A pitfalls
    top = np.argsort(-scores)[:k]
    top_sorted = sorted(top.tolist())  # keep reading order
    return [sents[i] for i in top_sorted]


# =============================
# HF API (cached)
# =============================
@lru_cache(maxsize=4096)
def paper_info(arxiv_id: str) -> Dict[str, Any]:
    return to_dict(api.paper_info(id=arxiv_id))


@lru_cache(maxsize=512)
def list_daily(date_str: str, limit: int, sort: str) -> List[Dict[str, Any]]:
    papers = list(api.list_daily_papers(date=date_str, limit=limit, sort=sort))
    return [to_dict(p) for p in papers]


def find_latest_date(lookback_days: int = 30) -> Optional[str]:
    today = dt.date.today()
    for i in range(lookback_days + 1):
        d = (today - dt.timedelta(days=i)).isoformat()
        try:
            items = list(api.list_daily_papers(date=d, limit=1, sort="publishedAt"))
            if len(items) > 0:
                return d
        except Exception:
            continue
    return None


# =============================
# Translation via HF Inference API (no torch/transformers in Space)
# =============================
_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-jap"  # light, standard
_infer_client: Optional[InferenceClient] = None


def get_infer_client() -> InferenceClient:
    global _infer_client
    if _infer_client is None:
        # token is optional but recommended; provider defaults are ok
        _infer_client = InferenceClient(token=HF_TOKEN or None, provider="hf-inference", timeout=60)
    return _infer_client


def translate_en_to_ja(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if not HF_TOKEN:
        # Be explicit: translation needs a token for reliability
        raise gr.Error("日本語翻訳には Space Secret `HF_TOKEN` が必要です。")
    client = get_infer_client()
    out = client.translation(text, model=_TRANSLATION_MODEL)  # 공식メソッド :contentReference[oaicite:4]{index=4}
    # can be str or TranslationOutput
    if isinstance(out, str):
        return out.strip()
    return (getattr(out, "translation_text", None) or str(out)).strip()


# =============================
# Rendering
# =============================
CSS = """
:root { --card-radius: 14px; }
.hfds-wrap { display: grid; gap: 12px; }
.hfds-card {
  border: 1px solid rgba(0,0,0,.10);
  border-radius: var(--card-radius);
  padding: 14px 14px 10px 14px;
  background: rgba(255,255,255,.75);
}
.hfds-title { font-size: 16px; font-weight: 700; margin: 0 0 6px 0; }
.hfds-meta { font-size: 12px; opacity: .75; margin: 0 0 8px 0; }
.hfds-links a { font-size: 12px; margin-right: 10px; }
.hfds-sec { margin: 10px 0 4px 0; font-weight: 650; }
.hfds-ul { margin: 0 0 10px 18px; }
.hfds-status { font-size: 12px; opacity: .8; }
"""

def render_cards(records: List[Dict[str, Any]]) -> str:
    if not records:
        return "<div class='hfds-status'>No results.</div>"

    parts = ["<div class='hfds-wrap'>"]
    for r in records:
        title = html_escape(r.get("title", ""))
        authors = html_escape(r.get("authors", ""))
        published = html_escape(r.get("published", ""))
        hf_url = r.get("hf_url", "")
        ax_url = r.get("arxiv_abs_url", "")
        pdf_url = r.get("pdf_url", "")
        bullets_en = r.get("bullets_en", []) or []
        bullets_ja = r.get("bullets_ja", []) or []

        parts.append("<div class='hfds-card'>")
        parts.append(f"<div class='hfds-title'>{r.get('rank','')}. {title}</div>")

        meta_bits = []
        if authors:
            meta_bits.append(f"Authors: {authors}")
        if published:
            meta_bits.append(f"Published: {published}")
        if meta_bits:
            parts.append(f"<div class='hfds-meta'>{' • '.join(meta_bits)}</div>")

        links = []
        if hf_url:
            links.append(f"<a href='{hf_url}' target='_blank' rel='noopener'>HF</a>")
        if ax_url:
            links.append(f"<a href='{ax_url}' target='_blank' rel='noopener'>arXiv</a>")
        if pdf_url:
            links.append(f"<a href='{pdf_url}' target='_blank' rel='noopener'>PDF</a>")
        if links:
            parts.append(f"<div class='hfds-links'>{' '.join(links)}</div>")

        if bullets_en:
            parts.append("<div class='hfds-sec'>Key points</div><ul class='hfds-ul'>")
            for b in bullets_en:
                parts.append(f"<li>{html_escape(b)}</li>")
            parts.append("</ul>")

        if bullets_ja:
            parts.append("<div class='hfds-sec'>要点（日本語）</div><ul class='hfds-ul'>")
            for b in bullets_ja:
                parts.append(f"<li>{html_escape(b)}</li>")
            parts.append("</ul>")

        parts.append("</div>")  # card
    parts.append("</div>")  # wrap
    return "\n".join(parts)


def render_markdown(all_records: List[Dict[str, Any]], date_str: str) -> str:
    lines = [f"# {APP_TITLE} — {date_str}", ""]
    for r in all_records:
        lines += [f"## {r.get('rank')}. {r.get('title','')}", ""]
        lines += [
            f"- HF: {r.get('hf_url','')}",
            f"- arXiv: {r.get('arxiv_abs_url','')}",
            f"- PDF: {r.get('pdf_url','')}",
        ]
        if r.get("authors"):
            lines.append(f"- Authors: {r['authors']}")
        if r.get("published"):
            lines.append(f"- Published: {r['published']}")
        lines += ["", "**Key points**", ""]
        for b in (r.get("bullets_en") or []):
            lines.append(f"- {b}")
        if r.get("bullets_ja"):
            lines += ["", "**要点（日本語）**", ""]
            for b in (r.get("bullets_ja") or []):
                lines.append(f"- {b}")
        lines += [""]
    return "\n".join(lines)


# =============================
# Export
# =============================
def export_files(df: pd.DataFrame, md: str, date_str: str) -> Tuple[str, str, str]:
    date_str = (date_str or "unknown").strip()
    csv_path = EXPORT_DIR / f"hfdailysummary_{date_str}.csv"
    jsonl_path = EXPORT_DIR / f"hfdailysummary_{date_str}.jsonl"
    md_path = EXPORT_DIR / f"hfdailysummary_{date_str}.md"

    df2 = df.copy()
    df2["bullets_en"] = df2["bullets_en"].apply(lambda x: "\n".join(x) if isinstance(x, list) else str(x))
    df2["bullets_ja"] = df2["bullets_ja"].apply(lambda x: "\n".join(x) if isinstance(x, list) else str(x))
    df2.to_csv(csv_path, index=False)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    return str(csv_path), str(jsonl_path), str(md_path)


# =============================
# Main pipeline (with visible errors)
# =============================
def run_pipeline_safe(
    date_str: str,
    limit: int,
    sort: str,
    enrich: bool,
    k: int,
    query: str,
    page: int,
    per_page: int,
    output_lang: str,
    translate_top_n: int,
    polite_sleep: float,
    progress=gr.Progress(track_tqdm=False),
) -> Tuple[pd.DataFrame, str, str, str]:
    _rid = rid()
    try:
        date_str = (date_str or "").strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            raise gr.Error("date は YYYY-MM-DD 形式で指定してください。")

        query = (query or "").strip().lower()
        limit = int(limit)
        k = int(k)
        page = max(1, int(page))
        per_page = max(1, int(per_page))
        translate_top_n = int(translate_top_n)

        progress(0, desc="Fetching Daily Papers…")
        base = list_daily(date_str, limit=limit, sort=sort)

        rows: List[Dict[str, Any]] = []
        total = max(1, len(base))

        for i, p in enumerate(base, start=1):
            arxiv_id = guess_arxiv_id(p)
            merged = dict(p)

            if enrich and arxiv_id:
                try:
                    merged.update(paper_info(arxiv_id))
                except Exception:
                    pass

            title = first_nonempty(merged, ["title"], default="")
            authors = normalize_authors(first_nonempty(merged, ["authors", "authorNames"], default=[]))
            published = first_nonempty(merged, ["publishedAt", "published_at", "published"], default="")

            if query:
                if (query not in (title or "").lower()) and (query not in (authors or "").lower()):
                    progress(i / total)
                    continue

            text = best_text_for_summary(merged)
            bullets_en = extractive_bullets(text, k=k) if text else []

            rows.append(
                {
                    "date": date_str,
                    "rank": len(rows) + 1,
                    "arxiv_id": arxiv_id or "",
                    "title": title,
                    "authors": authors,
                    "published": str(published),
                    "hf_url": first_nonempty(merged, ["url", "paper_url", "hf_url"], default="") or hf_paper_url(arxiv_id),
                    "arxiv_abs_url": arxiv_abs_url(arxiv_id),
                    "pdf_url": arxiv_pdf_url(arxiv_id),
                    "bullets_en": bullets_en,
                    "bullets_ja": [],
                }
            )

            if polite_sleep and float(polite_sleep) > 0:
                time.sleep(float(polite_sleep))

            progress(i / total)

        df = pd.DataFrame(rows)
        if df.empty:
            return df, "<div class='hfds-status'>No results.</div>", "No results.", f"rid={_rid} no results"

        # Optional Japanese translation (top N only)
        if output_lang == "日本語":
            top_n = min(max(0, translate_top_n), len(df))
            if top_n > 0:
                progress(0, desc=f"Translating top {top_n}…")
                for idx in range(top_n):
                    bullets_en = df.at[idx, "bullets_en"] or []
                    bullets_ja = [translate_en_to_ja(b) for b in bullets_en]
                    df.at[idx, "bullets_ja"] = bullets_ja
                    progress((idx + 1) / top_n)

        # Page cards
        start = (page - 1) * per_page
        end = start + per_page
        cards_html = render_cards(df.iloc[start:end].to_dict(orient="records"))
        md = render_markdown(df.to_dict(orient="records"), date_str=date_str)
        status = f"rid={_rid} loaded={len(df)} page={page} per_page={per_page} lang={output_lang}"
        return df, cards_html, md, status

    except gr.Error:
        logging.exception("rid=%s gr.Error", _rid)
        raise
    except Exception as e:
        logging.exception("rid=%s unhandled", _rid)
        # user-safe error + request id
        raise gr.Error(f"Error (rid={_rid}): {type(e).__name__}: {e}")


def set_latest_date() -> str:
    d = find_latest_date(lookback_days=30)
    if not d:
        raise gr.Error("直近30日で daily papers が見つかりませんでした。")
    return d


def do_export(df: pd.DataFrame, md: str, date_str: str) -> Tuple[str, str, str]:
    if df is None or len(df) == 0:
        raise gr.Error("Export するデータがありません（先に Run してください）。")
    return export_files(df, md or "", date_str)


# =============================
# UI
# =============================
with gr.Blocks(title=APP_TITLE, css=CSS) as demo:
    gr.Markdown(f"# {APP_TITLE}\n\n{TAGLINE}")

    with gr.Row():
        date = gr.Textbox(label="Date (YYYY-MM-DD)", value=dt.date.today().isoformat(), scale=2)
        btn_latest = gr.Button("Latest (API)", scale=1)
        sort = gr.Dropdown(label="Sort", choices=["trending", "publishedAt"], value="trending", scale=1)

    with gr.Row():
        limit = gr.Slider(label="Fetch limit", minimum=5, maximum=200, step=5, value=50)
        k = gr.Slider(label="Bullets per paper", minimum=2, maximum=8, step=1, value=4)
        enrich = gr.Checkbox(label="Enrich via paper_info (slower)", value=True)

    with gr.Row():
        query = gr.Textbox(label="Filter (title/authors contains)", value="", placeholder="e.g., diffusion, RLHF, protein...")
        polite_sleep = gr.Slider(label="Polite sleep (sec) per paper", minimum=0.0, maximum=1.0, step=0.05, value=0.05)

    with gr.Row():
        output_lang = gr.Dropdown(label="Output language", choices=["English", "日本語"], value="English")
        translate_top_n = gr.Slider(label="Translate top N papers", minimum=0, maximum=30, step=1, value=10)

    with gr.Row():
        page = gr.Number(label="Page", value=1, precision=0)
        per_page = gr.Number(label="Per page", value=5, precision=0)
        run_btn = gr.Button("Run", variant="primary")

    status = gr.Markdown("")

    with gr.Tab("Cards"):
        cards = gr.HTML()

    with gr.Tab("Table"):
        table = gr.Dataframe(
            headers=["rank", "title", "authors", "published", "hf_url", "arxiv_abs_url", "pdf_url", "bullets_en", "bullets_ja"],
            datatype=["number", "str", "str", "str", "str", "str", "str", "str", "str"],
            wrap=True,
            interactive=False,
        )

    with gr.Tab("Export"):
        md_out = gr.Markdown()
        export_btn = gr.Button("Export (CSV/JSONL/MD)")
        out_csv = gr.File(label="CSV")
        out_jsonl = gr.File(label="JSONL")
        out_md = gr.File(label="Markdown")

    # Wire events
    btn_latest.click(fn=set_latest_date, inputs=None, outputs=[date])

    run_btn.click(
        fn=run_pipeline_safe,
        inputs=[date, limit, sort, enrich, k, query, page, per_page, output_lang, translate_top_n, polite_sleep],
        outputs=[table, cards, md_out, status],
    )

    export_btn.click(fn=do_export, inputs=[table, md_out, date], outputs=[out_csv, out_jsonl, out_md])

# queue is needed for modals (Warning/Info/Error) behavior
demo.queue()

if __name__ == "__main__":
    # show_error is useful only while debugging. Keep DEBUG=0 in production.
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=DEBUG)
