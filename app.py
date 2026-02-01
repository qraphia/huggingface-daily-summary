import os
import re
import html
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import requests
import gradio as gr
from huggingface_hub import InferenceClient

APP_TITLE = "HuggingFace Daily Summary"

DAILY_PAPERS_API = "https://huggingface.co/api/daily_papers"
HF_PAPER_URL = "https://huggingface.co/papers/{paper_id}"
ARXIV_ABS_URL = "https://arxiv.org/abs/{paper_id}"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{paper_id}.pdf"

# ✅ NLLB はあなたの環境で 404 になっているので使わない
TRANSLATION_MODEL = "LiquidAI/LFM2-350M-ENJP-MT"

# Space Secrets (and local env) should provide this
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

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
.title { font-size: 1.05rem; font-weight: 700; margin: 0 0 6px; }
.meta { color: var(--muted); font-size: 0.92rem; margin: 0 0 8px; }
.links a { margin-right: 10px; font-size: 0.92rem; }
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

# ---------- text utils ----------
_JP_CHAR = r"\u3040-\u30FF\u4E00-\u9FFF\u3000-\u303F"
def _normalize_summary(s: str) -> str:
    if not s:
        return ""
    # Remove "L1:" like markers from HF daily_papers
    s = re.sub(r"\bL\d+:\s*", "", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def _extract_bullets(summary: str, max_bullets: int = 4) -> List[str]:
    """
    Heuristic:
    - If pattern like (1) ... (2) ... exists, split by it
    - else split by sentences and take first few
    """
    summary = _normalize_summary(summary)
    if not summary:
        return []

    # Try "(1) ... (2) ..." style
    parts = re.split(r"\(\d+\)\s*", summary)
    if len(parts) >= 3:
        # parts[0] is preface
        bullets = [p.strip(" .;\n") for p in parts[1:] if p.strip()]
        bullets = bullets[:max_bullets]
        return bullets

    # Fallback: sentence-ish split
    s = re.sub(r"\s+", " ", summary).strip()
    # split by ., !, ?, 。, ！, ？
    sent = re.split(r"(?<=[\.\!\?\。\！\？])\s+", s)
    sent = [x.strip() for x in sent if x.strip()]
    return sent[:max_bullets]

def _format_authors(authors: List[Dict[str, Any]], max_names: int = 8) -> Tuple[str, str]:
    names = [a.get("name", "").strip() for a in (authors or []) if a.get("name")]
    if not names:
        return ("", "")
    short = ", ".join(names[:max_names]) + (f", et al. (+{len(names)-max_names})" if len(names) > max_names else "")
    full = ", ".join(names)
    return (short, full)

def _clean_ja_spacing(text: str) -> str:
    """
    Remove spurious spaces between Japanese characters/punctuations (common MT artifact).
    Won't fix semantic garbage, but improves readability when MT is basically correct.
    """
    if not text:
        return ""
    # remove spaces between Japanese chars
    text = re.sub(rf"(?<=[{_JP_CHAR}])\s+(?=[{_JP_CHAR}])", "", text)
    # remove spaces before Japanese punctuation
    text = re.sub(r"\s+([、。．，！？：；）】』」〉》])", r"\1", text)
    return text.strip()

# ---------- HF translation ----------
_TRANSLATOR: Optional[InferenceClient] = None
_TRANSLATION_CACHE: Dict[str, str] = {}

def _get_translator() -> Optional[InferenceClient]:
    global _TRANSLATOR
    if not HF_TOKEN:
        return None
    if _TRANSLATOR is None:
        # provider must be hf-inference for translation() in huggingface_hub
        _TRANSLATOR = InferenceClient(provider="hf-inference", token=HF_TOKEN, timeout=60)
    return _TRANSLATOR

def translate_en_to_ja(text: str) -> str:
    """
    Robust: never crash the app. If translation fails, return empty string.
    """
    text = (text or "").strip()
    if not text:
        return ""
    if text in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[text]

    client = _get_translator()
    if client is None:
        return ""

    try:
        out = client.translation(text, model=TRANSLATION_MODEL)
        # out can be str / dict-like / object depending on client version
        if isinstance(out, str):
            ja = out
        elif isinstance(out, dict):
            ja = out.get("translation_text") or out.get("generated_text") or str(out)
        else:
            ja = getattr(out, "translation_text", None) or getattr(out, "generated_text", None) or str(out)

        ja = _clean_ja_spacing(ja)
        _TRANSLATION_CACHE[text] = ja
        return ja
    except Exception:
        # Fail closed: no Japanese rather than broken app
        return ""

def translate_bullets(bullets: List[str]) -> List[str]:
    if not bullets:
        return []
    # Batch translate as a single block to reduce calls
    block = "\n".join([f"- {b}" for b in bullets])
    ja_block = translate_en_to_ja(block)
    if not ja_block:
        return []
    # Try to split back into lines
    lines = [re.sub(r"^\s*[-•]\s*", "", ln).strip() for ln in ja_block.splitlines() if ln.strip()]
    return lines[: len(bullets)] if lines else []

# ---------- data fetch ----------
def fetch_daily_papers(date_str: str, limit: int) -> List[Dict[str, Any]]:
    params = {"date": date_str, "limit": int(limit), "sort": "trending"}
    r = requests.get(DAILY_PAPERS_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        return []
    return data

def build_cards(date_str: str, limit: int, lang: str) -> Tuple[str, str]:
    """
    Returns: (html, status_md)
    """
    # Validate date
    try:
        dt.date.fromisoformat(date_str)
    except Exception:
        return ("", f"❌ 日付形式が不正です: `{date_str}`（YYYY-MM-DD で入力してください）")

    try:
        items = fetch_daily_papers(date_str, limit)
    except Exception as e:
        return ("", f"❌ daily_papers API の取得に失敗: `{type(e).__name__}`")

    want_ja = (lang == "日本語（要約+翻訳）")
    if want_ja and not HF_TOKEN:
        # Don’t crash; just warn and show English only
        status = "⚠️ `HF_TOKEN` が未設定なので日本語翻訳はスキップしました（Space Secrets に `HF_TOKEN` を入れてください）。"
        want_ja = False
    else:
        status = "✅ 取得完了"

    cards_html = ["<div class='container'>"]
    for idx, it in enumerate(items, start=1):
        paper = (it or {}).get("paper") or {}
        paper_id = paper.get("id") or it.get("id") or ""
        title = (it.get("title") or paper.get("title") or "").strip()
        summary = it.get("summary") or paper.get("summary") or ""
        ai_summary = paper.get("ai_summary") or ""
        keywords = paper.get("ai_keywords") or []
        upvotes = paper.get("upvotes")
        authors = paper.get("authors") or []
        authors_short, authors_full = _format_authors(authors)

        bullets_en = _extract_bullets(summary, max_bullets=4)
        if not bullets_en and ai_summary:
            bullets_en = _extract_bullets(ai_summary, max_bullets=3)

        bullets_ja = translate_bullets(bullets_en) if want_ja else []

        hf_url = HF_PAPER_URL.format(paper_id=paper_id) if paper_id else ""
        arxiv_abs = ARXIV_ABS_URL.format(paper_id=paper_id) if paper_id else ""
        arxiv_pdf = ARXIV_PDF_URL.format(paper_id=paper_id) if paper_id else ""

        # Escape for HTML safety
        title_h = html.escape(title) if title else "(no title)"
        authors_short_h = html.escape(authors_short)
        authors_full_h = html.escape(authors_full)

        links = []
        if hf_url:
            links.append(f"<a href='{hf_url}' target='_blank' rel='noopener'>HF Paper</a>")
        if arxiv_abs:
            links.append(f"<a href='{arxiv_abs}' target='_blank' rel='noopener'>arXiv</a>")
        if arxiv_pdf:
            links.append(f"<a href='{arxiv_pdf}' target='_blank' rel='noopener'>PDF</a>")

        cards_html.append("<div class='card'>")
        cards_html.append(f"<div class='title'>{idx}. {title_h}</div>")
        meta_bits = []
        if upvotes is not None:
            meta_bits.append(f"▲ {upvotes}")
        if authors_short_h:
            meta_bits.append(authors_short_h)
        cards_html.append(f"<div class='meta'>{' · '.join(meta_bits)}</div>")

        if links:
            cards_html.append(f"<div class='links'>{' '.join(links)}</div>")

        if keywords and isinstance(keywords, list):
            badges = "".join([f"<span class='badge'>{html.escape(str(k))}</span>" for k in keywords[:12]])
            cards_html.append(f"<div class='badges'>{badges}</div>")

        # English bullets
        if bullets_en:
            cards_html.append("<div class='section'><h4>Key points (EN)</h4><ul>")
            for b in bullets_en:
                cards_html.append(f"<li>{html.escape(b)}</li>")
            cards_html.append("</ul></div>")

        # Japanese bullets
        if bullets_ja:
            cards_html.append("<div class='section'><h4>要点（日本語）</h4><ul>")
            for b in bullets_ja:
                cards_html.append(f"<li>{html.escape(b)}</li>")
            cards_html.append("</ul></div>")
        elif want_ja:
            cards_html.append("<small class='note'>※ 翻訳が失敗したため日本語は省略（token 権限/モデル提供状況/一時障害を確認）</small>")

        # Full authors collapsible
        if authors_full_h and len(authors_full_h) > len(authors_short_h) + 10:
            cards_html.append(
                f"<details style='margin-top:10px;'><summary><small>Full authors</small></summary>"
                f"<div class='meta' style='margin-top:6px;'>{authors_full_h}</div></details>"
            )

        cards_html.append("</div>")  # card

    cards_html.append("</div>")  # container
    return ("\n".join(cards_html), status)

# ---------- UI ----------
def today_iso() -> str:
    # HF Spaces runs in UTC by default; we keep it simple: today in UTC.
    return dt.date.today().isoformat()

with gr.Blocks() as demo:
    gr.Markdown(f"# {APP_TITLE}\nHugging Face Daily Papers を指定日付で取得し、要点を表示します。")
    with gr.Row():
        date_in = gr.Textbox(label="Date (YYYY-MM-DD)", value=today_iso(), max_lines=1)
        limit_in = gr.Slider(label="Limit", minimum=1, maximum=20, step=1, value=5)
        lang_in = gr.Dropdown(
            label="Output language",
            choices=["English", "日本語（要約+翻訳）"],
            value="English",
        )
        run_btn = gr.Button("Run", variant="primary")

    status_out = gr.Markdown()
    html_out = gr.HTML()

    run_btn.click(
        fn=build_cards,
        inputs=[date_in, limit_in, lang_in],
        outputs=[html_out, status_out],
    )

# ✅ Gradio 6.x: theme/css are passed to launch(), not Blocks()
demo.launch(theme=gr.themes.Soft(), css=CSS, ssr_mode=False)
