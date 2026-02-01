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
import pandas as pd
from huggingface_hub import HfApi, InferenceClient


# =========================
# Config
# =========================
APP_TITLE = "Hugging Face Daily Summary"
TAGLINE = "Daily Papers → key points → optional Japanese translation (NLLB)"
DEBUG = os.getenv("DEBUG", "0") == "1"

# Space Secrets に設定する（Settings → Variables and secrets）
HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()

EXPORT_DIR = Path("/tmp/hfds_exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(v\d+)?\b")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")

# 翻訳（OPUSは捨てる）
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "jpn_Jpan"

# 日本語空白の崩れを除去するための文字クラス
JP_CHARS = r"\u3040-\u30FF\u3400-\u9FFF\uF900-\uFAFF"

CSS = """
.hfds-wrap { display: grid; gap: 12px; }
.hfds-card { border: 1px solid rgba(0,0,0,.12); border-radius: 14px; padding: 14px; background: rgba(255,255,255,.78); }
.hfds-title { font-size: 16px; font-weight: 750; margin: 0 0 6px 0; line-height: 1.35; }
.hfds-meta { font-size: 12px; opacity: .8; margin: 0 0 8px 0; white-space: normal; word-break: break-word; }
.hfds-links { margin: 4px 0 0 0; }
.hfds-links a { font-size: 12px; margin-right: 10px; text-decoration: none; }
.hfds-sec { margin: 10px 0 4px 0; font-weight: 700; font-size: 13px; }
.hfds-ul { margin: 0 0 6px 18px; }
.hfds-status { font-size: 12px; opacity: .85; }
.small-note { font-size: 12px; opacity: .7; }
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s pid=%(process)d %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)

api = HfApi()

# 翻訳クライアントは1回だけ作る（毎回作ると遅い）
_TRANSLATOR: Optional[InferenceClient] = None
if HF_TOKEN:
    _TRANSLATOR = InferenceClient(provider="hf-inference", token=HF_TOKEN, timeout=60)


# =========================
# Small helpers
# =========================
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


def author_name(a: Any) -> str:
    if a is None:
        return ""
    if isinstance(a, str):
        return a.strip()
    if isinstance(a, dict):
        for k in ["name", "full_name", "fullname", "author"]:
            v = a.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    for attr in ["name", "full_name", "fullname", "author"]:
        if hasattr(a, attr):
            v = getattr(a, attr)
            if isinstance(v, str) and v.strip():
                return v.strip()
    if hasattr(a, "user") and getattr(a, "user") is not None:
        u = getattr(a, "user")
        for attr in ["fullname", "full_name", "name", "username"]:
            if hasattr(u, attr):
                v = getattr(u, attr)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    s = str(a).strip()
    m = re.search(r"name='([^']+)'", s)
    return m.group(1).strip() if m else ""


def format_authors(authors_any: Any, max_names: int = 8) -> Tuple[str, str]:
    names: List[str] = []
    if isinstance(authors_any, str):
        names = [x.strip() for x in authors_any.split(",") if x.strip()]
    elif isinstance(authors_any, list):
        for a in authors_any:
            n = author_name(a)
            if n:
                names.append(n)
    else:
        n = author_name(authors_any)
        if n:
            names = [n]

    seen = set()
    uniq = []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)

    full = ", ".join(uniq)
    if len(uniq) <= max_names:
        return full, full
    short = ", ".join(uniq[:max_names]) + f", et al. (+{len(uniq) - max_names})"
    return short, full


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
    return first_nonempty(
        p,
        ["summary", "abstract", "paperAbstract", "description", "content", "text"],
        default="",
    ) or ""


# =========================
# Extractive summarization (simple TF-IDF-ish)
# =========================
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
    df: Dict[str, int] = {}
    for ts in toks:
        for t in set(ts):
            df[t] = df.get(t, 0) + 1

    def idf(t: str) -> float:
        return math.log((N + 1) / (df.get(t, 0) + 1)) + 1.0

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

    scores: List[float] = []
    for vec in sent_vecs:
        s = 0.0
        for t, w in vec.items():
            s += w * doc_vec.get(t, 0.0)
        scores.append(s)

    top_idx = sorted(range(N), key=lambda i: scores[i], reverse=True)[:k]
    top_idx = sorted(top_idx)
    return [sents[i] for i in top_idx]


# =========================
# HF API (cached)
# =========================
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


# =========================
# Translation (NLLB) + cleanup
# =========================
def _cleanup_ja(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    # remove spaces before punctuation (English & Japanese)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\s+([、。])", r"\1", s)
    s = re.sub(r"([、。])\s+", r"\1", s)
    # remove spaces between Japanese characters
    s = re.sub(fr"(?<=[{JP_CHARS}])\s+(?=[{JP_CHARS}])", "", s)
    # collapse multiple spaces
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


@lru_cache(maxsize=8192)
def translate_en_to_ja(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if _TRANSLATOR is None:
        raise gr.Error(
            "日本語翻訳を使うには、Space の Settings → Variables and secrets で "
            "`HF_TOKEN` を Secret として設定してください（アカウントにトークンを作っただけでは環境変数に入りません）。"
        )

    out = _TRANSLATOR.translation(
        text,
        model=TRANSLATION_MODEL,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
        clean_up_tokenization_spaces=True,
        generate_parameters={"max_new_tokens": 256, "num_beams": 4},
    )
    if isinstance(out, str):
        return _cleanup_ja(out)
    return _cleanup_ja(getattr(out, "translation_text", None) or str(out))


# =========================
# Rendering + Export
# =========================
def render_cards(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "<div class='hfds-status'>No results.</div>"

    parts = ["<div class='hfds-wrap'>"]
    for r in rows:
        title = html_escape(r.get("title", ""))
        authors_short = html_escape(r.get("authors_short", ""))
        authors_full = html_escape(r.get("authors_full", ""))
        published = html_escape(str(r.get("published", "")) or "")

        links = []
        if r.get("hf_url"):
            links.append(f"<a href='{r['hf_url']}' target='_blank' rel='noopener'>HF</a>")
        if r.get("arxiv_abs_url"):
            links.append(f"<a href='{r['arxiv_abs_url']}' target='_blank' rel='noopener'>arXiv</a>")
        if r.get("pdf_url"):
            links.append(f"<a href='{r['pdf_url']}' target='_blank' rel='noopener'>PDF</a>")

        parts.append("<div class='hfds-card'>")
        parts.append(f"<div class='hfds-title'>{r.get('rank','')}. {title}</div>")

        meta_bits = []
        if authors_short:
            meta_bits.append(f"<b>Authors:</b> {authors_short}")
        if published:
            meta_bits.append(f"<b>Published:</b> {published}")
        if meta_bits:
            parts.append(f"<div class='hfds-meta'>{' • '.join(meta_bits)}</div>")
        if authors_full and authors_full != authors_short:
            parts.append(f"<div class='small-note'>Full authors: {authors_full}</div>")
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


def render_markdown(rows: List[Dict[str, Any]], date_str: str) -> str:
    lines = [f"# {APP_TITLE} — {date_str}", ""]
    for r in rows:
        lines += [f"## {r.get('rank')}. {r.get('title','')}", ""]
        lines += [
            f"- HF: {r.get('hf_url','')}",
            f"- arXiv: {r.get('arxiv_abs_url','')}",
            f"- PDF: {r.get('pdf_url','')}",
        ]
        if r.get("authors_full"):
            lines.append(f"- Authors: {r['authors_full']}")
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


def export_files(rows: List[Dict[str, Any]], date_str: str) -> Tuple[str, str, str]:
    stamp = f"{date_str}_{rid()}"
    csv_path = EXPORT_DIR / f"hfdailysummary_{stamp}.csv"
    jsonl_path = EXPORT_DIR / f"hfdailysummary_{stamp}.jsonl"
    md_path = EXPORT_DIR / f"hfdailysummary_{stamp}.md"

    df = pd.DataFrame(rows).copy()
    for col in ["bullets_en", "bullets_ja"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: "\n".join(x) if isinstance(x, list) else (x or ""))
    df.to_csv(csv_path, index=False)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    md = render_markdown(rows, date_str)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    for p in [csv_path, jsonl_path, md_path]:
        if not p.is_file():
            raise gr.Error(f"Export failed: not a file: {p}")

    return str(csv_path.resolve()), str(jsonl_path.resolve()), str(md_path.resolve())


# =========================
# Pipeline
# =========================
def run_pipeline(
    date_str: str,
    limit: int,
    sort: str,
    k: int,
    output_lang: str,
    translate_top_n: int,
    progress=gr.Progress(track_tqdm=False),
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, str, str]:
    _rid = rid()
    date_str = (date_str or "").strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        raise gr.Error("date は YYYY-MM-DD 形式で指定してください。")

    limit = int(limit)
    k = int(k)
    translate_top_n = int(translate_top_n)

    progress(0, desc="Fetching Daily Papers…")
    base = list_daily(date_str, limit=limit, sort=sort)

    rows: List[Dict[str, Any]] = []
    total = max(1, len(base))

    for i, p in enumerate(base, start=1):
        arxiv_id = guess_arxiv_id(p)
        title = first_nonempty(p, ["title"], default="")
        authors_any = first_nonempty(p, ["authors", "authorNames"], default=[])
        authors_short, authors_full = format_authors(authors_any, max_names=8)
        published = first_nonempty(p, ["publishedAt", "published_at", "published"], default="")

        text = best_text(p)
        bullets_en = extractive_bullets(text, k=k) if text else []

        rows.append(
            {
                "date": date_str,
                "rank": len(rows) + 1,
                "arxiv_id": arxiv_id or "",
                "title": title,
                "authors_short": authors_short,
                "authors_full": authors_full,
                "published": str(published),
                "hf_url": first_nonempty(p, ["url", "paper_url", "hf_url"], default="") or hf_paper_url(arxiv_id),
                "arxiv_abs_url": arxiv_abs_url(arxiv_id),
                "pdf_url": arxiv_pdf_url(arxiv_id),
                "bullets_en": bullets_en,
                "bullets_ja": [],
            }
        )

        progress(i / total)

    if not rows:
        return [], pd.DataFrame(), "<div class='hfds-status'>No results.</div>", f"rid={_rid} no results"

    if output_lang == "日本語":
        top_n = min(max(0, translate_top_n), len(rows))
        if top_n > 0:
            progress(0, desc=f"Translating top {top_n} (NLLB)…")
            for idx in range(top_n):
                rows[idx]["bullets_ja"] = [translate_en_to_ja(b) for b in (rows[idx].get("bullets_en") or [])]
                progress((idx + 1) / top_n)

    cards_html = render_cards(rows)

    df = pd.DataFrame(
        [
            {
                "rank": r["rank"],
                "title": r["title"],
                "authors": r["authors_short"],
                "published": r["published"],
                "hf_url": r["hf_url"],
                "arxiv": r["arxiv_abs_url"],
                "pdf": r["pdf_url"],
                "bullets_en": "\n".join(r.get("bullets_en") or []),
                "bullets_ja": "\n".join(r.get("bullets_ja") or []),
            }
            for r in rows
        ]
    )

    status = f"rid={_rid} rows={len(rows)} lang={output_lang}"
    return rows, df, cards_html, status


def set_latest_date() -> str:
    d = find_latest_date(lookback_days=30)
    if not d:
        raise gr.Error("直近30日で daily papers が見つかりませんでした。")
    return d


def do_export(rows: List[Dict[str, Any]], date_str: str) -> Tuple[str, str, str]:
    if not rows:
        raise gr.Error("Export するデータがありません（先に Run してください）。")
    return export_files(rows, date_str)


# =========================
# UI
# =========================
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

    with gr.Row():
        output_lang = gr.Dropdown(label="Output language", choices=["English", "日本語"], value="English")
        translate_top_n = gr.Slider(label="Translate top N papers", minimum=0, maximum=30, step=1, value=10)
        run_btn = gr.Button("Run", variant="primary")

    status = gr.Markdown("")

    with gr.Tab("Cards"):
        cards = gr.HTML()

    with gr.Tab("Table"):
        table = gr.Dataframe(wrap=True, interactive=False)

    with gr.Tab("Export"):
        gr.Markdown("Run 後に Export を押すと、CSV / JSONL / Markdown を生成します。")
        export_btn = gr.Button("Export (CSV/JSONL/MD)")
        out_csv = gr.File(label="CSV")
        out_jsonl = gr.File(label="JSONL")
        out_md = gr.File(label="Markdown")

    btn_latest.click(fn=set_latest_date, inputs=None, outputs=[date])

    run_btn.click(
        fn=run_pipeline,
        inputs=[date, limit, sort, k, output_lang, translate_top_n],
        outputs=[state_rows, table, cards, status],
    )

    export_btn.click(fn=do_export, inputs=[state_rows, date], outputs=[out_csv, out_jsonl, out_md])

demo.queue()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=DEBUG,
        css=CSS,
        theme=gr.themes.Soft(),
        ssr_mode=False,
    )
