import os
import re
import json
import time
import datetime as dt
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gradio as gr

from huggingface_hub import HfApi
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# App: Hugging Face Daily Summary
# ============================================================

APP_TITLE = "Hugging Face Daily Summary"
TAGLINE = "Daily Papers → extractive key points → optional Japanese (NLLB)"
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)

ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(v\d+)?\b")

api = HfApi()


# ----------------------------
# Utilities
# ----------------------------
def to_dict(x: Any) -> Dict[str, Any]:
    if is_dataclass(x):
        return asdict(x)
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
                # common keys
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
    # common keys
    for k in ["id", "paper_id", "arxiv_id", "arxivId"]:
        v = p.get(k)
        if isinstance(v, str):
            m = ARXIV_ID_RE.search(v)
            if m:
                return m.group(0)

    # URLs
    for k in ["url", "paper_url", "hf_url", "link"]:
        v = p.get(k)
        if isinstance(v, str):
            m = ARXIV_ID_RE.search(v)
            if m:
                return m.group(0)
            m2 = re.search(r"/papers/([^/?#]+)", v)
            if m2 and ARXIV_ID_RE.match(m2.group(1)):
                return m2.group(1)

    # fallback: scan JSON dump
    m = ARXIV_ID_RE.search(json.dumps(p, ensure_ascii=False))
    return m.group(0) if m else None


def hf_paper_url(arxiv_id: Optional[str]) -> str:
    return f"https://huggingface.co/papers/{arxiv_id}" if arxiv_id else ""


def arxiv_abs_url(arxiv_id: Optional[str]) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""


def arxiv_pdf_url(arxiv_id: Optional[str]) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""


def best_text_for_summary(p: Dict[str, Any]) -> str:
    # field names can vary; keep a wide net
    return first_nonempty(
        p,
        ["summary", "abstract", "paperAbstract", "description", "content", "text"],
        default="",
    ) or ""


# ----------------------------
# Extractive summarization (non-generative)
# ----------------------------
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
    X = vec.fit_transform(sents)
    centroid = X.mean(axis=0)
    scores = (X @ centroid.T).A.ravel()

    top = np.argsort(-scores)[:k]
    top_sorted = sorted(top.tolist())  # keep original reading order
    return [sents[i] for i in top_sorted]


# ----------------------------
# HF API (cached)
# ----------------------------
@lru_cache(maxsize=4096)
def paper_info(arxiv_id: str) -> Dict[str, Any]:
    # id must be arXiv ID
    return to_dict(api.paper_info(id=arxiv_id))


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


# ----------------------------
# Translation (EN -> JA) using NLLB (no sacremoses)
# ----------------------------
_NLLB = None  # (tokenizer, model, device)


def load_nllb():
    global _NLLB
    if _NLLB is not None:
        return _NLLB

    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device)
    mdl.eval()

    _NLLB = (tok, mdl, device)
    return _NLLB


def translate_en_to_ja(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    tok, mdl, device = load_nllb()
    src = "eng_Latn"
    tgt = "jpn_Jpan"
    tok.src_lang = src

    # chunk input to keep stable
    chunks = [text[i : i + 900] for i in range(0, len(text), 900)]

    import torch

    outs = []
    with torch.no_grad():
        for c in chunks:
            inputs = tok(c, return_tensors="pt", truncation=True, max_length=512).to(device)
            gen = mdl.generate(
                **inputs,
                forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
                max_new_tokens=256,
                do_sample=False,
            )
            outs.append(tok.batch_decode(gen, skip_special_tokens=True)[0])
    return "\n".join(outs).strip()


# ----------------------------
# Rendering
# ----------------------------
def html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_cards(records: List[Dict[str, Any]]) -> str:
    if not records:
        return "<div class='empty'>No results.</div>"

    html = ["<div class='cards'>"]
    for r in records:
        title = html_escape(r.get("title", ""))
        authors = html_escape(r.get("authors", ""))
        published = html_escape(r.get("published", ""))
        hf_url = r.get("hf_url", "")
        ax_url = r.get("arxiv_abs_url", "")
        pdf_url = r.get("pdf_url", "")

        bullets_en = r.get("bullets_en", []) or []
        bullets_ja = r.get("bullets_ja", []) or []

        html.append("<div class='card'>")
        html.append(f"<div class='card-title'>{title}</div>")

        meta = []
        if authors:
            meta.append(f"<span><b>Authors:</b> {authors}</span>")
        if published:
            meta.append(f"<span><b>Published:</b> {published}</span>")
        if meta:
            html.append("<div class='card-meta'>" + " • ".join(meta) + "</div>")

        links = []
        if hf_url:
            links.append(f"<a href='{hf_url}' target='_blank' rel='noopener'>HF</a>")
        if ax_url:
            links.append(f"<a href='{ax_url}' target='_blank' rel='noopener'>arXiv</a>")
        if pdf_url:
            links.append(f"<a href='{pdf_url}' target='_blank' rel='noopener'>PDF</a>")
        html.append("<div class='card-links'>" + " · ".join(links) + "</div>")

        if bullets_en:
            html.append("<div class='section-h'>Key points</div><ul class='bullets'>")
            for b in bullets_en:
                html.append(f"<li>{html_escape(b)}</li>")
            html.append("</ul>")

        if bullets_ja:
            html.append("<div class='section-h'>要点（日本語）</div><ul class='bullets'>")
            for b in bullets_ja:
                html.append(f"<li>{html_escape(b)}</li>")
            html.append("</ul>")

        html.append("</div>")
    html.append("</div>")
    return "\n".join(html)


def render_markdown(all_records: List[Dict[str, Any]], date_str: str) -> str:
    lines = [f"# {APP_TITLE} — {date_str}", ""]
    for r in all_records:
        lines += [f"## {r.get('rank')}. {r.get('title','')}", ""]
        lines += [
            f"- HF: {r.get('hf_url','')}",
            f"- arXiv: {r.get('arxiv_abs_url','')}",
            f"- PDF: {r.get('pdf_url','')}",
            "",
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


# ----------------------------
# Export
# ----------------------------
def export_files(df: pd.DataFrame, md: str, date_str: str) -> Tuple[str, str, str]:
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


# ----------------------------
# Pipeline
# ----------------------------
def run_pipeline(
    date_str: str,
    limit: int,
    sort: str,
    enrich: bool,
    k: int,
    query: str,
    page: int,
    per_page: int,
    translate: bool,
    translate_top_n: int,
    polite_sleep: float,
    progress=gr.Progress(track_tqdm=False),
) -> Tuple[pd.DataFrame, str, str, str, str, str, str]:
    date_str = (date_str or "").strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        raise ValueError("date must be YYYY-MM-DD")

    query = (query or "").strip().lower()

    base = list_daily(date_str, limit=int(limit), sort=sort)

    rows: List[Dict[str, Any]] = []
    total = max(1, len(base))
    progress(0, desc="Fetching & processing…")

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
        bullets_en = extractive_bullets(text, k=int(k)) if text else []

        rows.append(
            {
                "date": date_str,
                "rank": len(rows) + 1,  # rank AFTER filtering
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

        if polite_sleep and polite_sleep > 0:
            time.sleep(float(polite_sleep))

        progress(i / total)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, "<div class='empty'>No results.</div>", "No results.", "", "", "", ""

    # Optional translation: only top N (after filtering) to keep compute bounded
    if translate:
        top_n = max(0, int(translate_top_n))
        top_n = min(top_n, len(df))
        if top_n > 0:
            progress(0, desc=f"Translating top {top_n}… (NLLB download on first run)")
            # load model once (slow first time)
            _ = load_nllb()
            for idx in range(top_n):
                bullets_en = df.at[idx, "bullets_en"] or []
                bullets_ja = [translate_en_to_ja(b) for b in bullets_en]
                df.at[idx, "bullets_ja"] = bullets_ja
                progress((idx + 1) / top_n)

    # Paging for cards
    page = max(1, int(page))
    per_page = max(1, int(per_page))
    start = (page - 1) * per_page
    end = start + per_page

    page_records = df.iloc[start:end].to_dict(orient="records")
    cards_html = render_cards(page_records)

    status = f"Loaded: {len(df)} papers. Showing {start+1}-{min(end, len(df))}."
    if translate:
        status += f" JA translated top {min(int(translate_top_n), len(df))}."

    # Exports
    all_records = df.to_dict(orient="records")
    md = render_markdown(all_records, date_str=date_str)
    csv_f, jsonl_f, md_f = export_files(df, md, date_str=date_str)

    # Table (for quick scan)
    table = df[
        ["rank", "title", "arxiv_id", "hf_url", "arxiv_abs_url", "pdf_url", "published", "authors"]
    ].copy()

    return table, cards_html, status, md, csv_f, jsonl_f, md_f


# ============================================================
# UI
# ============================================================
CSS = """
#app { max-width: 1200px; margin: 0 auto; }
.header { padding: 14px 16px; border-radius: 16px; border: 1px solid rgba(0,0,0,0.08); }
.title { font-size: 22px; font-weight: 800; margin: 0; }
.tagline { color: rgba(0,0,0,0.65); margin-top: 6px; }
.kbd { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; padding: 2px 6px; border-radius: 8px; background: rgba(0,0,0,0.06); }

.cards { display: grid; grid-template-columns: 1fr; gap: 12px; margin-top: 8px; }
.card { border: 1px solid rgba(0,0,0,0.10); border-radius: 16px; padding: 14px 14px; background: white; }
.card-title { font-weight: 800; font-size: 16px; line-height: 1.35; }
.card-meta { margin-top: 8px; font-size: 12px; color: rgba(0,0,0,0.70); }
.card-links { margin-top: 8px; font-size: 13px; }
.card-links a { text-decoration: none; }
.section-h { margin-top: 10px; font-weight: 750; font-size: 13px; }
.bullets { margin: 8px 0 0 18px; }
.empty { padding: 18px; border: 1px dashed rgba(0,0,0,0.25); border-radius: 16px; color: rgba(0,0,0,0.65); }
"""

def ui_latest():
    d = find_latest_date(lookback_days=30)
    return d or ""

def ui_yesterday():
    return (dt.date.today() - dt.timedelta(days=1)).isoformat()

with gr.Blocks(theme=gr.themes.Soft(), css=CSS) as demo:
    gr.HTML(f"""
    <div id="app" class="header">
      <div class="title">{APP_TITLE}</div>
      <div class="tagline">{TAGLINE}. Export: <span class="kbd">CSV</span> / <span class="kbd">JSONL</span> / <span class="kbd">MD</span></div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            date_str = gr.Textbox(label="Date (YYYY-MM-DD)", value=dt.date.today().isoformat())
            with gr.Row():
                btn_latest = gr.Button("Latest (API)", size="sm")
                btn_yday = gr.Button("Yesterday", size="sm")
            limit = gr.Slider(1, 100, value=30, step=1, label="Limit")
            sort = gr.Dropdown(["publishedAt", "trending"], value="publishedAt", label="Sort")
            enrich = gr.Checkbox(value=True, label="Enrich via paper_info (better text, slower)")
            k = gr.Slider(1, 8, value=4, step=1, label="Bullets (k)")
            query = gr.Textbox(label="Filter (title/authors substring)", value="")

            with gr.Accordion("Advanced", open=False):
                page = gr.Number(value=1, precision=0, label="Page")
                per_page = gr.Number(value=10, precision=0, label="Per page")
                polite_sleep = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Sleep per paper (rate-limit)")
                translate = gr.Checkbox(value=False, label="Translate EN→JA (NLLB; heavy first run)")
                translate_top_n = gr.Slider(0, 50, value=5, step=1, label="Translate only top N papers")

            run_btn = gr.Button("Run", variant="primary")
            status = gr.Markdown()

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Cards"):
                    cards = gr.HTML()
                with gr.Tab("Table"):
                    table = gr.Dataframe()
                with gr.Tab("Markdown"):
                    md_out = gr.Markdown()
                with gr.Tab("Downloads"):
                    csv_f = gr.File(label="CSV")
                    jsonl_f = gr.File(label="JSONL")
                    md_f = gr.File(label="Markdown")

    btn_latest.click(fn=ui_latest, inputs=[], outputs=[date_str])
    btn_yday.click(fn=ui_yesterday, inputs=[], outputs=[date_str])

    run_btn.click(
        fn=run_pipeline,
        inputs=[date_str, limit, sort, enrich, k, query, page, per_page, translate, translate_top_n, polite_sleep],
        outputs=[table, cards, status, md_out, csv_f, jsonl_f, md_f],
    )

if __name__ == "__main__":
    # Spaces sets PORT; local defaults to 7860
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
