import os
import re
import json
import time
import uuid
import math
import logging
import datetime as dt
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from html import escape as html_escape

import gradio as gr
from huggingface_hub import HfApi, InferenceClient

# ---------------------------
# Config
# ---------------------------
APP_TITLE = "Hugging Face Daily Summary"
TAGLINE = "Hugging Face Daily Papers を日付指定で取得 → 抽出要約 → （任意）日本語翻訳"

DEBUG = os.getenv("DEBUG", "0") == "1"
HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()  # Space Secrets 推奨
EXPORT_DIR = Path("/tmp/hfds_exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(v\d+)?\b")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")

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
.hfds-status { font-size: 12px; opacity: .85; }
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s pid=%(process)d %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)

api = HfApi()

# ---------------------------
# Helpers
# ---------------------------
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

def best_text(p: Dict[str, Any]) -> str:
    return first_nonempty(p, ["summary", "abstract", "paperAbstract", "description", "content", "text"], default="") or ""

# ---------------------------
# Extractive summarization (pure python TF-IDF)
# ---------------------------
def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]

def tokenize(s: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(s or "")]

def extractive_bullets(text: str, k: int = 4) -> List[str]:
    sents = split_sentences(text)
    if len(sents) <= k:
        return sents

    toks = [tokenize(s) for s in sents]
    N = len(sents)

    # document frequency
    df: Dict[str, int] = {}
    for ts in toks:
        for t in set(ts):
            df[t] = df.get(t, 0) + 1

    def idf(t: str) -> float:
        # smooth idf
        return math.log((N + 1) / (df.get(t, 0) + 1)) + 1.0

    # sentence tf-idf vectors + doc (centroid-like) vector
    sent_vecs: List[Dict[str, float]] = []
    doc_vec: Dict[str, float] = {}

    for ts in toks:
        tf: Dict[str, int] = {}
        for t in ts:
            tf[t] = tf.get(t, 0) + 1
        denom = max(1, len(ts))
        vec: Dict[str, float] = {}
        for t, c in tf.items():
            w = (c / denom) * idf(t)
            vec[t] = w
            doc_vec[t] = doc_vec.get(t, 0.0) + w
        sent_vecs.append(vec)

    # score by dot(vec, doc_vec)
    scores: List[float] = []
    for vec in sent_vecs:
        s = 0.0
        for t, w in vec.items():
            s += w * doc_vec.get(t, 0.0)
        scores.append(s)

    top_idx = sorted(range(N), key=lambda i: scores[i], reverse=True)[:k]
    top_idx = sorted(top_idx)  # preserve reading order
    return [sents[i] for i in top_idx]

# ---------------------------
# HF API (cached)
# ---------------------------
@lru_cache(maxsize=4096)
def paper_info(arxiv_id: str) -> Dict[str, Any]:
    return to_dict(api.paper_info(id=arxiv_id))

@lru_cache(maxsize=512)
def list_daily(date_str: str, limit: int, sort: str) -> List[Dict[str, Any]]:
    items = list(api.list_daily_papers(date=date_str, limit=limit, sort=sort))
    return [to_dict(x) for x in items]

def find_latest_date(lookback_days: int = 30) -> Optional[str]:
    today = dt.date.today()
    for i in range(lookback_days + 1):
        d = (today - dt.timedelta(days=i)).isoformat()
        try:
            if len(list(api.list_daily_papers(date=d, limit=1, sort="publishedAt"))) > 0:
                return d
        except Exception:
            pass
    return None

# ---------------------------
# Translation (serverless Inference API)
# ---------------------------
_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-jap"

def translate_en_to_ja(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if not HF_TOKEN:
        raise gr.Error("日本語翻訳には Space Secrets に `HF_TOKEN` が必要です。")
    client = InferenceClient(provider="hf-inference", token=HF_TOKEN, timeout=60)
    out = client.translation(text, model=_TRANSLATION_MODEL)
    if isinstance(out, str):
        return out.strip()
    return (getattr(out, "translation_text", None) or str(out)).strip()

# ---------------------------
# Rendering + Export
# ---------------------------
def render_cards(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "<div class='hfds-status'>No results.</div>"
    parts = ["<div class='hfds-wrap'>"]
    for r in rows:
        title = html_escape(r.get("title", ""))
        authors = html_escape(r.get("authors", ""))
        published = html_escape(r.get("published", ""))
        links = []
        if r.get("hf_url"):
            links.append(f"<a href='{r['hf_url']}' target='_blank' rel='noopener'>HF</a>")
        if r.get("arxiv_abs_url"):
            links.append(f"<a href='{r['arxiv_abs_url']}' target='_blank' rel='noopener'>arXiv</a>")
        if r.get("pdf_url"):
            links.append(f"<a href='{r['pdf_url']}' target='_blank' rel='noopener'>PDF</a>")

        parts.append("<div class='hfds-card'>")
        parts.append(f"<div class='hfds-title'>{r.get('rank','')}. {title}</div>")
        meta = []
        if authors:
            meta.append(f"Authors: {authors}")
        if published:
            meta.append(f"Published: {published}")
        if meta:
            parts.append(f"<div class='hfds-meta'>{' • '.join(meta)}</div>")
        if links:
            parts.append(f"<div class='hfds-links'>{' '.join(links)}</div>")

        bullets_en = r.get("bullets_en") or []
        bullets_ja = r.get("bullets_ja") or []

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

        parts.append("</div>")
    parts.append("</div>")
    return "\n".join(parts)

def rows_to_table(rows: List[Dict[str, Any]]) -> List[List[Any]]:
    cols = ["rank","title","authors","published","hf_url","arxiv_abs_url","pdf_url","bullets_en","bullets_ja"]
    table = []
    for r in rows:
        row = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, list):
                v = "\n".join(v)
            row.append(v)
        table.append(row)
    return table

def render_markdown(rows: List[Dict[str, Any]], date_str: str) -> str:
    lines = [f"# {APP_TITLE} — {date_str}", ""]
    for r in rows:
        lines += [f"## {r.get('rank')}. {r.get('title','')}", ""]
        lines += [f"- HF: {r.get('hf_url','')}",
                  f"- arXiv: {r.get('arxiv_abs_url','')}",
                  f"- PDF: {r.get('pdf_url','')}",]
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
        lines.append("")
    return "\n".join(lines)

def export_files(rows: List[Dict[str, Any]], md: str, date_str: str) -> Tuple[str, str, str]:
    stamp = f"{date_str}_{rid()}"
    csv_path = EXPORT_DIR / f"hfdailysummary_{stamp}.csv"
    jsonl_path = EXPORT_DIR / f"hfdailysummary_{stamp}.jsonl"
    md_path = EXPORT_DIR / f"hfdailysummary_{stamp}.md"

    # CSV (minimal)
    import csv
    cols = ["date","rank","arxiv_id","title","authors","published","hf_url","arxiv_abs_url","pdf_url","bullets_en","bullets_ja"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["bullets_en"] = "\n".join(rr.get("bullets_en") or [])
            rr["bullets_ja"] = "\n".join(rr.get("bullets_ja") or [])
            w.writerow({c: rr.get(c, "") for c in cols})

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md or "")

    # Return absolute paths (and ensure they are files)
    for p in [csv_path, jsonl_path, md_path]:
        if not p.is_file():
            raise gr.Error(f"Export failed: not a file: {p}")
    return str(csv_path.resolve()), str(jsonl_path.resolve()), str(md_path.resolve())

# ---------------------------
# Pipeline
# ---------------------------
def run_pipeline(
    date_str: str,
    limit: int,
    sort: str,
    enrich: bool,
    k: int,
    query: str,
    output_lang: str,
    translate_top_n: int,
    polite_sleep: float,
    progress=gr.Progress(track_tqdm=False),
) -> Tuple[List[Dict[str, Any]], List[List[Any]], str, str]:
    _rid = rid()
    date_str = (date_str or "").strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        raise gr.Error("date は YYYY-MM-DD 形式で指定してください。")

    query = (query or "").strip().lower()
    limit = int(limit)
    k = int(k)
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

        text = best_text(merged)
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

    if not rows:
        return [], [], "<div class='hfds-status'>No results.</div>", f"rid={_rid} no results"

    if output_lang == "日本語":
        top_n = min(max(0, translate_top_n), len(rows))
        if top_n > 0:
            progress(0, desc=f"Translating top {top_n}…")
            for idx in range(top_n):
                rows[idx]["bullets_ja"] = [translate_en_to_ja(b) for b in (rows[idx].get("bullets_en") or [])]
                progress((idx + 1) / top_n)

    cards_html = render_cards(rows)
    status = f"rid={_rid} rows={len(rows)} lang={output_lang}"
    table = rows_to_table(rows)
    return rows, table, cards_html, status

def set_latest_date() -> str:
    d = find_latest_date(lookback_days=30)
    if not d:
        raise gr.Error("直近30日で daily papers が見つかりませんでした。")
    return d

def do_export(rows: List[Dict[str, Any]], date_str: str) -> Tuple[str, str, str]:
    if not rows:
        raise gr.Error("Export するデータがありません（先に Run してください）。")
    md = render_markdown(rows, date_str)
    return export_files(rows, md, date_str)

# ---------------------------
# UI
# ---------------------------
with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}\n\n{TAGLINE}")

    state_rows = gr.State([])

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
        polite_sleep = gr.Slider(label="Polite sleep (sec) per paper", minimum=0.0, maximum=0.5, step=0.05, value=0.05)

    with gr.Row():
        output_lang = gr.Dropdown(label="Output language", choices=["English", "日本語"], value="English")
        translate_top_n = gr.Slider(label="Translate top N papers", minimum=0, maximum=30, step=1, value=10)

    run_btn = gr.Button("Run", variant="primary")
    status = gr.Markdown("")

    with gr.Tab("Cards"):
        cards = gr.HTML()

    with gr.Tab("Table"):
        table = gr.Dataframe(
            headers=["rank","title","authors","published","hf_url","arxiv_abs_url","pdf_url","bullets_en","bullets_ja"],
            datatype=["number","str","str","str","str","str","str","str","str"],
            wrap=True,
            interactive=False,
        )

    with gr.Tab("Export"):
        gr.Markdown("Run 後に Export を押すと、CSV / JSONL / Markdown を生成します。")
        export_btn = gr.Button("Export (CSV/JSONL/MD)")
        out_csv = gr.File(label="CSV")
        out_jsonl = gr.File(label="JSONL")
        out_md = gr.File(label="Markdown")

    btn_latest.click(fn=set_latest_date, inputs=None, outputs=[date])

    run_btn.click(
        fn=run_pipeline,
        inputs=[date, limit, sort, enrich, k, query, output_lang, translate_top_n, polite_sleep],
        outputs=[state_rows, table, cards, status],
    )

    export_btn.click(
        fn=do_export,
        inputs=[state_rows, date],
        outputs=[out_csv, out_jsonl, out_md],
    )

demo.queue()

if __name__ == "__main__":
    # Gradio 6: css/theme は launch 側へ寄せる
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=DEBUG,
        css=CSS,
        theme=gr.themes.Soft(),
        ssr_mode=False,
    )
