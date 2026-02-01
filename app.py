# app.py (simple / stable)
# Hugging Face Daily Summary
#
# - Date -> fetch daily papers via huggingface_hub.HfApi
# - Extractive bullets (pure python TF-IDF; no torch / no sklearn)
# - Optional EN->JA translation via HF Inference (NLLB; fallback mBART)
# - Cards + Table + Export
#
# Space Secret:
# - HF_TOKEN (optional): enables translation; without it, translation is skipped (never crashes)

import os
import re
import json
import time
import uuid
import math
import csv
import datetime as dt
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from html import escape as html_escape

import gradio as gr
from huggingface_hub import HfApi, InferenceClient

APP_TITLE = "Hugging Face Daily Summary"
TAGLINE = "Daily Papers → key points → (optional) Japanese"

HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()
DEBUG = os.getenv("DEBUG", "0") == "1"

EXPORT_DIR = Path("/tmp/hfds_exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(v\d+)?\b")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")
JA_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
JA_BETWEEN_SPACE_RE = re.compile(r"(?<=[\u3040-\u30ff\u3400-\u9fff])\s+(?=[\u3040-\u30ff\u3400-\u9fff])")

# Better-than-OPUS defaults for technical EN->JA
MT_PRIMARY = ("facebook/nllb-200-distilled-600M", "eng_Latn", "jpn_Jpan")
MT_FALLBACK = ("facebook/mbart-large-50-many-to-many-mmt", "en_XX", "ja_XX")

CSS = """
.hfds-wrap{display:grid;gap:12px}
.hfds-card{border:1px solid rgba(0,0,0,.12);border-radius:14px;padding:14px;background:rgba(255,255,255,.78)}
.hfds-title{font-size:16px;font-weight:750;margin:0 0 6px 0;line-height:1.35}
.hfds-meta{font-size:12px;opacity:.82;margin:0 0 8px 0;word-break:break-word}
.hfds-links a{font-size:12px;margin-right:10px;text-decoration:none}
.hfds-sec{margin:10px 0 4px 0;font-weight:700;font-size:13px}
.hfds-ul{margin:0 0 6px 18px}
.hfds-status{font-size:12px;opacity:.85}
"""

api = HfApi()


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
        v = d.get(k)
        if v is None:
            continue
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and v:
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
    s = str(a)
    m = re.search(r"name='([^']+)'", s)
    return m.group(1).strip() if m else ""


def format_authors(authors_any: Any, max_names: int = 8) -> Tuple[str, str]:
    names: List[str] = []
    if isinstance(authors_any, str):
        names = [x.strip() for x in authors_any.split(",") if x.strip()]
    elif isinstance(authors_any, list):
        names = [author_name(x) for x in authors_any if author_name(x)]
    else:
        n = author_name(authors_any)
        names = [n] if n else []

    # de-dup preserve order
    seen, uniq = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)

    full = ", ".join(uniq)
    if len(uniq) <= max_names:
        return full, full
    short = ", ".join(uniq[:max_names]) + f", et al. (+{len(uniq)-max_names})"
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
    s = first_nonempty(p, ["abstract", "paperAbstract", "summary", "description", "content", "text"], default="") or ""
    return s.strip()[:5000]


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()][:64]


def tokenize(s: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(s or "")]


def extractive_bullets(text: str, k: int) -> List[str]:
    sents = split_sentences(text)
    if not sents:
        return []
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

    doc_vec: Dict[str, float] = {}
    sent_vecs: List[Dict[str, float]] = []
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

    scores = []
    for vec in sent_vecs:
        s = 0.0
        for t, w in vec.items():
            s += w * doc_vec.get(t, 0.0)
        scores.append(s)

    top_idx = sorted(range(N), key=lambda i: scores[i], reverse=True)[:k]
    top_idx = sorted(top_idx)
    return [sents[i] for i in top_idx]


def _postprocess_ja(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    s = JA_BETWEEN_SPACE_RE.sub("", s)
    s = re.sub(r"\s+([、。！？])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _ja_ratio(s: str) -> float:
    if not s:
        return 0.0
    return len(JA_CHAR_RE.findall(s)) / max(1, len(s))


def translate_en_to_ja(text: str, client: InferenceClient) -> Optional[str]:
    text = (text or "").strip()
    if not text:
        return ""
    for model, src, tgt in [MT_PRIMARY, MT_FALLBACK]:
        out = client.translation(text, model=model, src_lang=src, tgt_lang=tgt)
        if isinstance(out, str):
            ja = out
        else:
            ja = getattr(out, "translation_text", None) or str(out)
        ja = _postprocess_ja(ja)
        if _ja_ratio(ja) < 0.02:
            continue
        return ja
    return None


@lru_cache(maxsize=512)
def list_daily(date_str: str, limit: int, sort: str) -> List[Dict[str, Any]]:
    items = list(api.list_daily_papers(date=date_str, limit=limit, sort=sort))
    return [to_dict(x) for x in items]


def find_latest_date(lookback_days: int = 45) -> Optional[str]:
    today = dt.date.today()
    for i in range(lookback_days + 1):
        d = (today - dt.timedelta(days=i)).isoformat()
        try:
            if len(list(api.list_daily_papers(date=d, limit=1, sort="publishedAt"))) > 0:
                return d
        except Exception:
            pass
    return None


def render_cards(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "<div class='hfds-status'>No results.</div>"
    parts = ["<div class='hfds-wrap'>"]
    for r in rows:
        title = html_escape(r.get("title", ""))
        authors_short = html_escape(r.get("authors_short", ""))
        authors_full = html_escape(r.get("authors_full", ""))
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
        if authors_short:
            meta.append(f"<b>Authors:</b> {authors_short}")
        if published:
            meta.append(f"<b>Published:</b> {published}")
        if meta:
            parts.append(f"<div class='hfds-meta'>{' • '.join(meta)}</div>")
        if authors_full and authors_full != authors_short:
            parts.append(f"<div class='hfds-meta'>Full authors: {authors_full}</div>")
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
    out = []
    for r in rows:
        out.append(
            [
                r.get("rank", ""),
                r.get("title", ""),
                r.get("authors_short", ""),
                r.get("published", ""),
                r.get("hf_url", ""),
                r.get("arxiv_abs_url", ""),
                r.get("pdf_url", ""),
                "\n".join(r.get("bullets_en") or []),
                "\n".join(r.get("bullets_ja") or []),
            ]
        )
    return out


def export_files(rows: List[Dict[str, Any]], date_str: str) -> Tuple[str, str, str]:
    stamp = f"{date_str}_{rid()}"
    csv_path = EXPORT_DIR / f"hfdailysummary_{stamp}.csv"
    jsonl_path = EXPORT_DIR / f"hfdailysummary_{stamp}.jsonl"
    md_path = EXPORT_DIR / f"hfdailysummary_{stamp}.md"

    cols = [
        "date", "rank", "arxiv_id", "title", "authors_short", "authors_full", "published",
        "hf_url", "arxiv_abs_url", "pdf_url", "bullets_en", "bullets_ja"
    ]

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

    lines = [f"# {APP_TITLE} — {date_str}", ""]
    for r in rows:
        lines.append(f"## {r.get('rank')}. {r.get('title','')}")
        lines.append(f"- HF: {r.get('hf_url','')}")
        lines.append(f"- arXiv: {r.get('arxiv_abs_url','')}")
        lines.append(f"- PDF: {r.get('pdf_url','')}")
        if r.get("authors_full"):
            lines.append(f"- Authors: {r['authors_full']}")
        if r.get("published"):
            lines.append(f"- Published: {r['published']}")
        lines.append("")
        lines.append("**Key points**")
        for b in (r.get("bullets_en") or []):
            lines.append(f"- {b}")
        if r.get("bullets_ja"):
            lines.append("")
            lines.append("**要点（日本語）**")
            for b in (r.get("bullets_ja") or []):
                lines.append(f"- {b}")
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    for p in [csv_path, jsonl_path, md_path]:
        if not p.is_file():
            raise gr.Error(f"Export failed: not a file: {p}")
    return str(csv_path.resolve()), str(jsonl_path.resolve()), str(md_path.resolve())


def run(date_str: str, sort: str, limit: int, k: int, lang: str, top_n: int) -> Tuple[List[Dict[str, Any]], List[List[Any]], str, str]:
    _rid = rid()
    date_str = (date_str or "").strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        raise gr.Error("date は YYYY-MM-DD 形式で指定してください。")

    limit = int(limit)
    k = int(k)
    top_n = int(top_n)

    base = list_daily(date_str, limit=limit, sort=sort)

    rows: List[Dict[str, Any]] = []
    for p in base:
        arxiv_id = guess_arxiv_id(p)
        title = safe = (first_nonempty(p, ["title"], default="") or "").strip()
        authors_any = first_nonempty(p, ["authors", "authorNames"], default=[])
        authors_short, authors_full = format_authors(authors_any, max_names=8)
        published = (first_nonempty(p, ["publishedAt", "published_at", "published"], default="") or "").strip()

        text = best_text(p)
        bullets_en = extractive_bullets(text, k=k)

        rows.append(
            {
                "date": date_str,
                "rank": len(rows) + 1,
                "arxiv_id": arxiv_id or "",
                "title": title,
                "authors_short": authors_short,
                "authors_full": authors_full,
                "published": published,
                "hf_url": (first_nonempty(p, ["url", "paper_url", "hf_url"], default="") or "").strip() or hf_paper_url(arxiv_id),
                "arxiv_abs_url": arxiv_abs_url(arxiv_id),
                "pdf_url": arxiv_pdf_url(arxiv_id),
                "bullets_en": bullets_en,
                "bullets_ja": [],
            }
        )

    # optional translation
    note = ""
    if lang == "日本語":
        if not HF_TOKEN:
            note = "⚠️ HF_TOKEN 未設定のため翻訳はスキップ"
        else:
            client = InferenceClient(provider="hf-inference", token=HF_TOKEN, timeout=60)
            n = min(max(0, top_n), len(rows))
            for i in range(n):
                ja_list = []
                for b in rows[i]["bullets_en"]:
                    try:
                        ja = translate_en_to_ja(b, client)
                        ja_list.append(ja if ja is not None else b)
                    except Exception:
                        ja_list.append(b)
                rows[i]["bullets_ja"] = ja_list

    cards = render_cards(rows)
    table = rows_to_table(rows)
    status = f"rid={_rid} rows={len(rows)} lang={lang}"
    if note:
        status += f"\n\n{note}"
    return rows, table, cards, status


def latest() -> str:
    d = find_latest_date(lookback_days=45)
    if not d:
        raise gr.Error("直近45日で daily papers が見つかりませんでした。")
    return d


def do_export(rows: List[Dict[str, Any]], date_str: str) -> Tuple[str, str, str]:
    if not rows:
        raise gr.Error("Export するデータがありません（先に Run してください）。")
    return export_files(rows, date_str)


with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}\n\n{TAGLINE}")

    state_rows = gr.State([])

    with gr.Row():
        date = gr.Textbox(label="Date (YYYY-MM-DD)", value=dt.date.today().isoformat(), scale=2)
        btn_latest = gr.Button("Latest")
        sort = gr.Dropdown(label="Sort", choices=["trending", "publishedAt"], value="trending")

    with gr.Row():
        limit = gr.Slider(label="Limit", minimum=5, maximum=200, step=5, value=50)
        k = gr.Slider(label="Bullets", minimum=2, maximum=8, step=1, value=4)

    with gr.Row():
        lang = gr.Dropdown(label="Language", choices=["English", "日本語"], value="English")
        top_n = gr.Slider(label="Translate top N (if Japanese)", minimum=0, maximum=30, step=1, value=3)

    run_btn = gr.Button("Run", variant="primary")
    status = gr.Markdown("")

    with gr.Tab("Cards"):
        cards = gr.HTML()

    with gr.Tab("Table"):
        table = gr.Dataframe(
            headers=["rank", "title", "authors", "published", "hf_url", "arxiv", "pdf", "bullets_en", "bullets_ja"],
            datatype=["number", "str", "str", "str", "str", "str", "str", "str", "str"],
            wrap=True,
            interactive=False,
            row_count=(0, "dynamic"),
            col_count=(9, "fixed"),
        )

    with gr.Tab("Export"):
        export_btn = gr.Button("Export (CSV/JSONL/MD)")
        out_csv = gr.File(label="CSV")
        out_jsonl = gr.File(label="JSONL")
        out_md = gr.File(label="Markdown")

    btn_latest.click(fn=latest, inputs=None, outputs=[date])

    run_btn.click(
        fn=run,
        inputs=[date, sort, limit, k, lang, top_n],
        outputs=[state_rows, table, cards, status],
    )

    export_btn.click(
        fn=do_export,
        inputs=[state_rows, date],
        outputs=[out_csv, out_jsonl, out_md],
    )

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
