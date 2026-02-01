# app.py
# Hugging Face Daily Summary (Gradio Space)
#
# ✅ What it does
# - Pick a date (YYYY-MM-DD) for https://huggingface.co/papers/date/
# - Fetch daily papers via huggingface_hub.HfApi
# - Build extractive key points (pure-Python TF-IDF, no torch/sklearn)
# - Optional EN→JA translation via HF Inference (NLLB + fallback)
# - Clean Cards UI + stable Table + Export (CSV/JSONL/MD)
#
# ✅ Secrets (Space Settings → Variables and secrets)
# - HF_TOKEN (optional): needed ONLY for translation; without it, translation is skipped (no crash)
#
# Recommended README YAML (Space):
# ---
# title: Hugging Face Daily Summary
# sdk: gradio
# sdk_version: 6.5.1
# app_file: app.py
# ---

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


# =========================
# Config
# =========================
APP_TITLE = "Hugging Face Daily Summary"
TAGLINE = "Daily Papers → extractive key points → (optional) Japanese translation"

DEBUG = os.getenv("DEBUG", "0") == "1"
HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()

EXPORT_DIR = Path("/tmp/hfds_exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(v\d+)?\b")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")
JA_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
JA_BETWEEN_SPACE_RE = re.compile(r"(?<=[\u3040-\u30ff\u3400-\u9fff])\s+(?=[\u3040-\u30ff\u3400-\u9fff])")

# Translation models (HF Inference)
# Primary: NLLB (better for technical text)
MT_PRIMARY = ("facebook/nllb-200-distilled-600M", "eng_Latn", "jpn_Jpan")
# Fallback: mBART
MT_FALLBACK = ("facebook/mbart-large-50-many-to-many-mmt", "en_XX", "ja_XX")

# CSS for clean Cards UI (passed to launch() in Gradio 6)
CSS = """
:root { --r: 14px; }
.hfds-wrap { display: grid; gap: 12px; }
.hfds-card {
  border: 1px solid rgba(0,0,0,.12);
  border-radius: var(--r);
  padding: 14px;
  background: rgba(255,255,255,.78);
}
.hfds-title { font-size: 16px; font-weight: 750; margin: 0 0 6px 0; line-height: 1.35; }
.hfds-meta { font-size: 12px; opacity: .82; margin: 0 0 8px 0; white-space: normal; word-break: break-word; }
.hfds-links { margin: 4px 0 0 0; }
.hfds-links a { font-size: 12px; margin-right: 10px; text-decoration: none; }
.hfds-sec { margin: 10px 0 4px 0; font-weight: 700; font-size: 13px; }
.hfds-ul { margin: 0 0 6px 18px; }
.hfds-status { font-size: 12px; opacity: .85; }
.hfds-pill { display:inline-block; font-size: 12px; padding: 3px 8px; border-radius: 999px;
             border: 1px solid rgba(0,0,0,.12); background: rgba(0,0,0,.03); margin-right: 8px; }
.small-note { font-size: 12px; opacity: .72; }
"""


api = HfApi()


# =========================
# Utilities
# =========================
def rid() -> str:
    return uuid.uuid4().hex[:8]


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


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


def to_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    # huggingface_hub objects usually have __dict__
    if hasattr(x, "__dict__"):
        return dict(x.__dict__)
    return {"value": str(x)}


def author_name(a: Any) -> str:
    """Normalize PaperAuthor-like objects/dicts/strings → a clean name."""
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

    # object attributes
    for attr in ["name", "full_name", "fullname", "author"]:
        if hasattr(a, attr):
            v = getattr(a, attr)
            if isinstance(v, str) and v.strip():
                return v.strip()

    # nested user
    if hasattr(a, "user") and getattr(a, "user") is not None:
        u = getattr(a, "user")
        for attr in ["fullname", "full_name", "name", "username"]:
            if hasattr(u, attr):
                v = getattr(u, attr)
                if isinstance(v, str) and v.strip():
                    return v.strip()

    # last-resort parse
    s = str(a)
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

    # de-dup preserve order
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
    # Prefer abstract-like fields; cap length to reduce noise
    s = first_nonempty(p, ["abstract", "paperAbstract", "summary", "description", "content", "text"], default="") or ""
    s = s.strip()
    return s[:5000]


# =========================
# Extractive summarization (pure python TF-IDF)
# =========================
def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # Basic sentence split: good enough for abstracts
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if s.strip()]
    # Hard cap: avoid huge abstracts destroying runtime
    return sents[:64]


def tokenize(s: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(s or "")]


def extractive_bullets(text: str, k: int = 4) -> List[str]:
    sents = split_sentences(text)
    if not sents:
        return []
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
        return math.log((N + 1) / (df.get(t, 0) + 1)) + 1.0

    # sentence vecs + pseudo-centroid
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
    top_idx = sorted(top_idx)  # keep reading order
    return [sents[i] for i in top_idx]


# =========================
# HF API access (cached)
# =========================
@lru_cache(maxsize=1024)
def list_daily(date_str: str, limit: int, sort: str) -> List[Dict[str, Any]]:
    items = list(api.list_daily_papers(date=date_str, limit=limit, sort=sort))
    return [to_dict(x) for x in items]


@lru_cache(maxsize=4096)
def paper_info(arxiv_id: str) -> Dict[str, Any]:
    return to_dict(api.paper_info(id=arxiv_id))


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


# =========================
# Translation (Inference API)
# =========================
def _postprocess_ja(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    # remove spaces between Japanese chars
    s = JA_BETWEEN_SPACE_RE.sub("", s)
    # cleanup common punctuation spacing
    s = re.sub(r"\s+([、。！？])", r"\1", s)
    s = re.sub(r"([（(])\s+", r"\1", s)
    s = re.sub(r"\s+([）)])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _ja_ratio(s: str) -> float:
    if not s:
        return 0.0
    return len(JA_CHAR_RE.findall(s)) / max(1, len(s))


def translate_en_to_ja(text: str, client: InferenceClient) -> Optional[str]:
    """
    Returns translated Japanese string, or None if it looks invalid.
    Never throws (caller handles exceptions).
    """
    text = (text or "").strip()
    if not text:
        return ""

    for (model, src, tgt) in [MT_PRIMARY, MT_FALLBACK]:
        out = client.translation(text, model=model, src_lang=src, tgt_lang=tgt)
        if isinstance(out, str):
            ja = out
        else:
            ja = getattr(out, "translation_text", None) or str(out)
        ja = _postprocess_ja(ja)
        # reject clearly-bad output
        if _ja_ratio(ja) < 0.02:
            continue
        return ja

    return None


# =========================
# Rendering
# =========================
def render_cards(rows: List[Dict[str, Any]], page: int, per_page: int) -> str:
    if not rows:
        return "<div class='hfds-status'>No results.</div>"

    n = len(rows)
    page = max(1, int(page))
    per_page = max(1, int(per_page))
    start = (page - 1) * per_page
    end = min(start + per_page, n)
    if start >= n:
        start = max(0, n - per_page)
        end = n

    parts = [f"<div class='hfds-status'><span class='hfds-pill'>papers: {n}</span><span class='hfds-pill'>showing: {start+1}-{end}</span></div>"]
    parts.append("<div class='hfds-wrap'>")

    for r in rows[start:end]:
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

        parts.append("</div>")  # card

    parts.append("</div>")  # wrap
    return "\n".join(parts)


def rows_to_table(rows: List[Dict[str, Any]]) -> List[List[Any]]:
    cols = ["rank", "title", "authors", "published", "hf_url", "arxiv", "pdf", "bullets_en", "bullets_ja"]
    out: List[List[Any]] = []
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


def rows_to_markdown(rows: List[Dict[str, Any]], date_str: str) -> str:
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


# =========================
# Export (never return directories)
# =========================
def export_files(rows: List[Dict[str, Any]], date_str: str) -> Tuple[str, str, str]:
    if not rows:
        raise gr.Error("Export するデータがありません（先に Run してください）。")

    stamp = f"{date_str}_{rid()}"
    csv_path = EXPORT_DIR / f"hfdailysummary_{stamp}.csv"
    jsonl_path = EXPORT_DIR / f"hfdailysummary_{stamp}.jsonl"
    md_path = EXPORT_DIR / f"hfdailysummary_{stamp}.md"

    # CSV
    cols = [
        "date",
        "rank",
        "arxiv_id",
        "title",
        "authors_short",
        "authors_full",
        "published",
        "hf_url",
        "arxiv_abs_url",
        "pdf_url",
        "bullets_en",
        "bullets_ja",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["bullets_en"] = "\n".join(rr.get("bullets_en") or [])
            rr["bullets_ja"] = "\n".join(rr.get("bullets_ja") or [])
            w.writerow({c: rr.get(c, "") for c in cols})

    # JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Markdown
    md = rows_to_markdown(rows, date_str=date_str)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    for p in [csv_path, jsonl_path, md_path]:
        if not p.is_file():
            raise gr.Error(f"Export failed: not a file: {p}")
    return str(csv_path.resolve()), str(jsonl_path.resolve()), str(md_path.resolve())


# =========================
# Main pipeline
# =========================
def run_pipeline(
    date_str: str,
    sort: str,
    limit: int,
    enrich: bool,
    k: int,
    query: str,
    output_lang: str,
    translate_top_n: int,
    polite_sleep: float,
    page: int,
    per_page: int,
    progress=gr.Progress(track_tqdm=False),
) -> Tuple[List[Dict[str, Any]], List[List[Any]], str, str, str]:
    """
    Returns:
      - rows_state (list[dict]) for export
      - table_rows (list[list])
      - cards_html
      - markdown_text
      - status_text
    """
    _rid = rid()
    try:
        date_str = (date_str or "").strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            raise gr.Error("date は YYYY-MM-DD 形式で指定してください。")

        limit = int(limit)
        k = int(k)
        translate_top_n = int(translate_top_n)
        page = int(page)
        per_page = int(per_page)
        query = (query or "").strip().lower()
        polite_sleep = float(polite_sleep)

        progress(0, desc="Fetching daily papers…")
        base = list_daily(date_str, limit=limit, sort=sort)

        rows: List[Dict[str, Any]] = []
        total = max(1, len(base))

        for i, p in enumerate(base, start=1):
            arxiv_id = guess_arxiv_id(p)
            merged = dict(p)

            # optionally enrich
            if enrich and arxiv_id:
                try:
                    merged.update(paper_info(arxiv_id))
                except Exception:
                    pass

            title = safe_str(first_nonempty(merged, ["title"], default=""))
            authors_any = first_nonempty(merged, ["authors", "authorNames"], default=[])
            authors_short, authors_full = format_authors(authors_any, max_names=8)
            published = safe_str(first_nonempty(merged, ["publishedAt", "published_at", "published"], default=""))

            # filter
            if query:
                if (query not in title.lower()) and (query not in authors_full.lower()):
                    progress(i / total)
                    continue

            # summary
            text = best_text(merged)
            bullets_en = extractive_bullets(text, k=k) if text else []

            # urls
            hf_url = safe_str(first_nonempty(merged, ["url", "paper_url", "hf_url"], default="")) or hf_paper_url(arxiv_id)

            rows.append(
                {
                    "date": date_str,
                    "rank": len(rows) + 1,
                    "arxiv_id": arxiv_id or "",
                    "title": title,
                    "authors_short": authors_short,
                    "authors_full": authors_full,
                    "published": published,
                    "hf_url": hf_url,
                    "arxiv_abs_url": arxiv_abs_url(arxiv_id),
                    "pdf_url": arxiv_pdf_url(arxiv_id),
                    "bullets_en": bullets_en,
                    "bullets_ja": [],
                }
            )

            if polite_sleep > 0:
                time.sleep(polite_sleep)

            progress(i / total)

        if not rows:
            return [], [], "<div class='hfds-status'>No results.</div>", "No results.", f"rid={_rid} (no results)"

        # optional translation
        translation_note = ""
        if output_lang == "日本語":
            if not HF_TOKEN:
                translation_note = "⚠️ HF_TOKEN 未設定のため翻訳はスキップ（Space Settings → Secrets に HF_TOKEN を追加）"
            else:
                top_n = min(max(0, translate_top_n), len(rows))
                if top_n > 0:
                    progress(0, desc=f"Translating top {top_n}…")
                    client = InferenceClient(provider="hf-inference", token=HF_TOKEN, timeout=60)
                    for idx in range(top_n):
                        en_list = rows[idx].get("bullets_en") or []
                        ja_list: List[str] = []
                        for b in en_list:
                            try:
                                ja = translate_en_to_ja(b, client)
                                ja_list.append(ja if ja is not None else b)  # fallback to EN if invalid
                            except Exception:
                                ja_list.append(b)  # hard fallback: show EN rather than garbage
                        rows[idx]["bullets_ja"] = ja_list
                        progress((idx + 1) / top_n)

        table_rows = rows_to_table(rows)
        cards_html = render_cards(rows, page=page, per_page=per_page)
        md = rows_to_markdown(rows, date_str=date_str)

        status = f"rid={_rid} rows={len(rows)} sort={sort} lang={output_lang}"
        if translation_note:
            status += f"\n\n{translation_note}"

        return rows, table_rows, cards_html, md, status

    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Error (rid={_rid}): {type(e).__name__}: {e}")


def set_latest_date() -> str:
    d = find_latest_date(lookback_days=45)
    if not d:
        raise gr.Error("直近45日で daily papers が見つかりませんでした。")
    return d


def do_export(rows: List[Dict[str, Any]], date_str: str) -> Tuple[str, str, str]:
    date_str = (date_str or "").strip()
    if not date_str:
        date_str = dt.date.today().isoformat()
    return export_files(rows, date_str=date_str)


# =========================
# UI
# =========================
with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}\n\n{TAGLINE}")

    token_badge = "✅ Translation enabled" if HF_TOKEN else "⚠️ Translation disabled (set Space Secret HF_TOKEN to enable)"
    gr.Markdown(f"<span class='hfds-pill'>{token_badge}</span>", elem_id="token_badge")

    state_rows = gr.State([])

    with gr.Row():
        date = gr.Textbox(label="Date (YYYY-MM-DD)", value=dt.date.today().isoformat(), scale=2)
        btn_latest = gr.Button("Latest (API)", scale=1)
        sort = gr.Dropdown(label="Sort", choices=["trending", "publishedAt"], value="trending", scale=1)

    with gr.Accordion("Options", open=True):
        with gr.Row():
            limit = gr.Slider(label="Fetch limit", minimum=5, maximum=200, step=5, value=50)
            k = gr.Slider(label="Bullets per paper", minimum=2, maximum=8, step=1, value=4)
            enrich = gr.Checkbox(label="Enrich via paper_info (slower)", value=True)

        with gr.Row():
            query = gr.Textbox(label="Filter (title/authors contains)", value="", placeholder="e.g., diffusion, RLHF, protein…")
            polite_sleep = gr.Slider(label="Polite sleep (sec) per paper", minimum=0.0, maximum=0.5, step=0.05, value=0.05)

        with gr.Row():
            output_lang = gr.Dropdown(label="Output language", choices=["English", "日本語"], value="English")
            translate_top_n = gr.Slider(label="Translate top N papers", minimum=0, maximum=30, step=1, value=3)

        with gr.Row():
            page = gr.Number(label="Cards: page", value=1, precision=0)
            per_page = gr.Number(label="Cards: per page", value=5, precision=0)

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

    with gr.Tab("Markdown"):
        md_out = gr.Markdown()

    with gr.Tab("Export"):
        gr.Markdown("Run 後に Export を押すと、CSV / JSONL / Markdown を生成します。")
        export_btn = gr.Button("Export (CSV/JSONL/MD)")
        out_csv = gr.File(label="CSV")
        out_jsonl = gr.File(label="JSONL")
        out_md = gr.File(label="Markdown")

    btn_latest.click(fn=set_latest_date, inputs=None, outputs=[date])

    run_btn.click(
        fn=run_pipeline,
        inputs=[date, sort, limit, enrich, k, query, output_lang, translate_top_n, polite_sleep, page, per_page],
        outputs=[state_rows, table, cards, md_out, status],
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
