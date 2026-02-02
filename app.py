import datetime as dt
import html
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import gradio as gr
import requests


# =========================
# Config
# =========================

APP_TITLE = "Hugging Face Daily Summary"
APP_SUBTITLE = "Daily Papers, glance-first. Full abstract + BibTeX on demand."

HF_DAILY_PAPERS_API = "https://huggingface.co/api/daily_papers"
HF_PAPER_API = "https://huggingface.co/api/papers/{paper_id}"
HF_PAPER_PAGE = "https://huggingface.co/papers/{paper_id}"

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_ABS = "https://arxiv.org/abs/{paper_id}"
ARXIV_PDF = "https://arxiv.org/pdf/{paper_id}.pdf"

HTTP_TIMEOUT_S = int(os.getenv("HTTP_TIMEOUT_S", "25"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
BACKOFF_BASE_S = float(os.getenv("BACKOFF_BASE_S", "1.0"))

HF_CACHE_TTL_S = int(os.getenv("HF_CACHE_TTL_S", "300"))          # 5 min
ARXIV_CACHE_TTL_S = int(os.getenv("ARXIV_CACHE_TTL_S", "86400"))  # 24 h
PAPER_EXTRAS_TTL_S = int(os.getenv("PAPER_EXTRAS_TTL_S", "86400"))  # 24 h

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "6"))

HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "qraphia/huggingface-daily-summary (glance-first; gradio)",
        "Accept": "application/json",
    }
)
if HF_TOKEN:
    SESSION.headers["Authorization"] = f"Bearer {HF_TOKEN}"

HF_CACHE: Dict[Tuple[str, int], Tuple[float, List[Dict[str, Any]], Dict[str, str]]] = {}
ARXIV_CACHE: Dict[str, Tuple[float, "ArxivMeta"]] = {}
PAPER_EXTRAS_CACHE: Dict[str, Tuple[float, "PaperExtras"]] = {}

RE_ARXIV_NEW = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
RE_ARXIV_OLD = re.compile(r"^[a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$", re.IGNORECASE)

RE_MAX_DATE = re.compile(r'must be less than or equal to "([^"]+)"')

# Paper page HTML extraction (best-effort, minimal)
RE_PROJECT_PAGE = re.compile(r'href="([^"]+)"[^>]*>\s*Project page\s*<', re.IGNORECASE)
RE_GITHUB = re.compile(r'href="([^"]+github\.com[^"]+)"[^>]*>\s*GitHub\s*<', re.IGNORECASE)
RE_STARS = re.compile(r">\s*GitHub\s*(\d+)\s*<", re.IGNORECASE)

RE_MODELS = re.compile(r"Models citing this paper\s+(\d+)", re.IGNORECASE)
RE_DATASETS = re.compile(r"Datasets citing this paper\s+(\d+)", re.IGNORECASE)
RE_SPACES = re.compile(r"Spaces citing this paper\s+(\d+)", re.IGNORECASE)

CSS = """
:root{
  --bg:#f7f8fb;
  --card:#ffffff;
  --bd:#e6e8ef;
  --text:#111827;
  --muted:#6b7280;
  --chip:#f3f4f6;
  --link:#2563eb;
  --shadow: 0 1px 1px rgba(17,24,39,0.05);
}

body{ background: var(--bg); }
.gradio-container{ max-width: 1080px !important; }

.header{ margin: 10px 0 12px; }
.header h1{ margin: 0 0 0.25rem; font-size: 1.70rem; letter-spacing: -0.01em; }
.subtle{ color: var(--muted); font-size: 0.96rem; line-height: 1.45; }

.panel{
  background: var(--card);
  border: 1px solid var(--bd);
  border-radius: 16px;
  padding: 12px 12px;
  box-shadow: var(--shadow);
  margin-bottom: 12px;
}

.note{
  margin-top: 6px;
  color: var(--muted);
  font-size: 0.88rem;
  line-height: 1.35;
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

.topline{
  display:flex;
  gap: 10px;
  align-items:flex-start;
  justify-content:space-between;
}

.title{
  font-size: 1.08rem;
  font-weight: 850;
  color: var(--text);
  line-height: 1.35;
  margin: 0;
  flex: 1;
}

.badge{
  display:inline-flex;
  align-items:center;
  padding: 4px 10px;
  background: #111827;
  color: #fff;
  border-radius: 999px;
  font-size: 0.86rem;
  font-weight: 800;
  white-space: nowrap;
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
  padding: 2px 10px;
  background: var(--chip);
  border: 1px solid var(--bd);
  border-radius: 999px;
  color: #374151;
  font-size: 0.86rem;
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

details.more summary{
  cursor: pointer;
  color: #374151;
  font-weight: 850;
  font-size: 0.92rem;
  margin-top: 10px;
}

.more-block{
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid var(--bd);
}

pre.bibtex{
  margin-top: 8px;
  padding: 10px 10px;
  background: #0b1220;
  color: #e5e7eb;
  border-radius: 12px;
  border: 1px solid #111827;
  overflow-x: auto;
  font-size: 0.86rem;
  line-height: 1.45;
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
  font-weight: 800;
}

.links a:hover{ text-decoration: underline; }

.error-box{
  background: #fff1f2;
  border: 1px solid #fecdd3;
  border-radius: 16px;
  padding: 12px 12px;
  color: #7f1d1d;
}
"""


# =========================
# Models
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

@dataclass
class PaperExtras:
    project_page_url: Optional[str]
    github_repo_url: Optional[str]
    github_stars: Optional[int]
    citing_models: Optional[int]
    citing_datasets: Optional[int]
    citing_spaces: Optional[int]


# =========================
# Helpers
# =========================

def esc(s: str) -> str:
    return html.escape(s or "", quote=True)

def utc_today_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()

def normalize_text(s: str) -> str:
    if not s:
        return ""
    out = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    for i in range(1, 120):
        out = out.replace(f"L{i}:", "")
    while "\n\n\n" in out:
        out = out.replace("\n\n\n", "\n\n")
    return out.strip()

def is_arxiv_id(paper_id: str) -> bool:
    pid = (paper_id or "").strip()
    return bool(RE_ARXIV_NEW.match(pid) or RE_ARXIV_OLD.match(pid))

def ensure_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    if u.startswith("//"):
        return "https:" + u
    if u.startswith("/"):
        return "https://huggingface.co" + u
    if u.startswith("http://") or u.startswith("https://"):
        return u
    return "https://" + u

def parse_max_allowed_date_from_error(body_text: str) -> Optional[str]:
    try:
        obj = json.loads(body_text)
        msg = obj.get("error") if isinstance(obj, dict) else ""
        msg = str(msg)
    except Exception:
        msg = body_text or ""
    m = RE_MAX_DATE.search(msg)
    if not m:
        return None
    iso_dt = m.group(1).strip()
    if "T" in iso_dt:
        return iso_dt.split("T", 1)[0]
    return iso_dt[:10] if len(iso_dt) >= 10 else None

def safe_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[ApiError], str]:
    last_text = ""
    last_err: Optional[ApiError] = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            r = SESSION.get(url, params=params, timeout=HTTP_TIMEOUT_S)
            last_text = r.text or ""
        except requests.RequestException as e:
            last_err = ApiError(None, type(e).__name__, str(e)[:600], {})
            time.sleep(min(20.0, BACKOFF_BASE_S * (2 ** attempt)))
            continue

        if r.status_code == 200:
            try:
                return r.json(), None, ""
            except Exception:
                return None, ApiError(200, "Invalid JSON", (r.text or "")[:600], {}), (r.text or "")

        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(20.0, BACKOFF_BASE_S * (2 ** attempt)))
            continue

        return None, ApiError(r.status_code, r.reason or "HTTP error", (r.text or "")[:600], {}), (r.text or "")

    return None, last_err, last_text

def safe_get_text(url: str) -> Optional[str]:
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = SESSION.get(url, timeout=HTTP_TIMEOUT_S)
            if r.status_code == 200:
                return r.text or ""
        except requests.RequestException:
            pass
        time.sleep(min(15.0, BACKOFF_BASE_S * (2 ** attempt)))
    return None

def abstract_preview(full: str, max_chars: int = 520) -> Tuple[str, bool]:
    s = (full or "").strip()
    if not s:
        return "", False
    if len(s) <= max_chars:
        return s, False
    return s[: max_chars - 1].rstrip() + "…", True


# =========================
# HF daily papers (with auto clamp for future dates)
# =========================

def fetch_hf_daily(date_iso: str, limit: int) -> Tuple[Optional[List[Dict[str, Any]]], Optional[ApiError], str]:
    requested = (date_iso or "").strip()
    effective = requested

    key = (effective, int(limit))
    now = time.time()
    hit = HF_CACHE.get(key)
    if hit and (now - hit[0] < HF_CACHE_TTL_S):
        return hit[1], None, effective

    params = {"date": effective, "limit": int(limit), "sort": "trending"}
    data, err, raw_text = safe_get_json(HF_DAILY_PAPERS_API, params)

    if err is not None and err.status_code == 400:
        max_date = parse_max_allowed_date_from_error(raw_text)
        if max_date and max_date != effective:
            effective = max_date
            key2 = (effective, int(limit))
            hit2 = HF_CACHE.get(key2)
            if hit2 and (now - hit2[0] < HF_CACHE_TTL_S):
                return hit2[1], None, effective

            params2 = {"date": effective, "limit": int(limit), "sort": "trending"}
            data2, err2, _ = safe_get_json(HF_DAILY_PAPERS_API, params2)
            if err2 is None and isinstance(data2, list):
                HF_CACHE[key2] = (now, data2, {})
                return data2, None, effective
            return None, err2, effective

    if err is not None:
        return None, err, effective

    if not isinstance(data, list):
        return None, ApiError(200, "Unexpected payload", str(type(data))[:200], {}), effective

    HF_CACHE[key] = (now, data, {})
    return data, None, effective


# =========================
# Paper extras (Project page / GitHub / Hub assets counts)
# =========================

def _cache_get_extras(pid: str) -> Optional[PaperExtras]:
    now = time.time()
    hit = PAPER_EXTRAS_CACHE.get(pid)
    if hit and (now - hit[0] < PAPER_EXTRAS_TTL_S):
        return hit[1]
    return None

def _cache_put_extras(pid: str, extras: PaperExtras) -> None:
    PAPER_EXTRAS_CACHE[pid] = (time.time(), extras)

def _extract_counts(text: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    def g(rx: re.Pattern) -> Optional[int]:
        m = rx.search(text or "")
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None
    return g(RE_MODELS), g(RE_DATASETS), g(RE_SPACES)

def fetch_paper_extras(pid: str) -> PaperExtras:
    pid = (pid or "").strip()
    cached = _cache_get_extras(pid)
    if cached:
        return cached

    project_page_url: Optional[str] = None
    github_repo_url: Optional[str] = None
    github_stars: Optional[int] = None
    citing_models: Optional[int] = None
    citing_datasets: Optional[int] = None
    citing_spaces: Optional[int] = None

    # 1) API (best-effort)
    obj, err, _ = safe_get_json(HF_PAPER_API.format(paper_id=pid), None)
    if err is None and isinstance(obj, dict) and "error" not in obj:
        # GitHub fields are known to exist for some papers (e.g., githubRepo/githubStars on HF paper API pages). :contentReference[oaicite:4]{index=4}
        gr = obj.get("githubRepo")
        if isinstance(gr, str) and "github.com" in gr and gr.startswith("http"):
            github_repo_url = gr.strip()
        gs = obj.get("githubStars")
        try:
            if gs is not None:
                github_stars = int(gs)
        except Exception:
            github_stars = None

        # Project page field names are not guaranteed; fall back to HTML if missing.
        for k in ("projectPage", "projectPageUrl", "project_url", "projectUrl", "project_page_url"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                project_page_url = ensure_url(v.strip())
                break

    # 2) HTML (authoritative for “Project page” button existence) :contentReference[oaicite:5]{index=5}
    page_text = safe_get_text(HF_PAPER_PAGE.format(paper_id=pid))
    if page_text:
        if project_page_url is None:
            m = RE_PROJECT_PAGE.search(page_text)
            if m:
                project_page_url = ensure_url(m.group(1))
        if github_repo_url is None:
            m = RE_GITHUB.search(page_text)
            if m:
                github_repo_url = ensure_url(m.group(1))
        if github_stars is None:
            m = RE_STARS.search(page_text)
            if m:
                try:
                    github_stars = int(m.group(1))
                except Exception:
                    github_stars = None

        cm, cd, cs = _extract_counts(page_text)
        citing_models, citing_datasets, citing_spaces = cm, cd, cs

    extras = PaperExtras(
        project_page_url=project_page_url,
        github_repo_url=github_repo_url,
        github_stars=github_stars,
        citing_models=citing_models,
        citing_datasets=citing_datasets,
        citing_spaces=citing_spaces,
    )
    _cache_put_extras(pid, extras)
    return extras


# =========================
# arXiv meta (abstract + bibtex)
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
    return [xs[i:i + n] for i in range(0, len(xs), n)]

def _parse_arxiv_atom(xml_text: str) -> Dict[str, ArxivMeta]:
    out: Dict[str, ArxivMeta] = {}
    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return out

    for entry in root.findall("a:entry", ns):
        raw_id = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
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
    wanted: List[str] = []
    result: Dict[str, ArxivMeta] = {}

    for aid in arxiv_ids:
        aid = (aid or "").strip()
        if not aid or not is_arxiv_id(aid):
            continue
        cached = _cache_get_arxiv(aid)
        if cached:
            result[aid] = cached
        else:
            wanted.append(aid)

    if not wanted:
        return result

    for group in _chunk(wanted, 25):
        try:
            r = SESSION.get(ARXIV_API, params={"id_list": ",".join(group)}, timeout=HTTP_TIMEOUT_S)
            if r.status_code == 200:
                parsed = _parse_arxiv_atom(r.text or "")
                for k, meta in parsed.items():
                    result[k] = meta
                    _cache_put_arxiv(meta)
        except requests.RequestException:
            pass

    return result

def build_bibtex(meta: ArxivMeta) -> str:
    key = meta.arxiv_id.replace("/", "_")
    authors = " and ".join(meta.authors) if meta.authors else "Unknown"
    title = meta.title.replace("{", "\\{").replace("}", "\\}")
    year = meta.year or "????"
    url = ARXIV_ABS.format(paper_id=meta.arxiv_id)
    return (
        f"@misc{{arxiv:{key},\n"
        f"  title={{ {title} }},\n"
        f"  author={{ {authors} }},\n"
        f"  year={{ {year} }},\n"
        f"  eprint={{ {meta.arxiv_id} }},\n"
        f"  archivePrefix={{arXiv}},\n"
        f"  url={{ {url} }}\n"
        f"}}"
    )


# =========================
# Render
# =========================

def get_hf_authors(paper: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    authors = paper.get("authors") or []
    if isinstance(authors, list):
        for a in authors:
            if isinstance(a, dict) and a.get("name"):
                out.append(str(a["name"]).strip())
    return out

def format_authors(names: List[str], max_names: int = 8) -> str:
    names = [n.strip() for n in (names or []) if n and n.strip()]
    if not names:
        return "—"
    if len(names) <= max_names:
        return ", ".join(names)
    return ", ".join(names[:max_names]) + f", et al. (+{len(names) - max_names})"

def apply_search(items: List[Dict[str, Any]], q: str) -> List[Dict[str, Any]]:
    q = (q or "").strip().lower()
    if not q:
        return items
    out: List[Dict[str, Any]] = []
    for it in items:
        paper = (it or {}).get("paper") or {}
        title = str(it.get("title") or paper.get("title") or "")
        authors = " ".join(get_hf_authors(paper))
        hay = f"{title}\n{authors}".lower()
        if q in hay:
            out.append(it)
    return out

def render(date_iso: str, limit: int, query: str) -> Tuple[str, str]:
    date_iso = (date_iso or "").strip()
    try:
        dt.date.fromisoformat(date_iso)
    except Exception:
        return "", "❌ Invalid date. Use YYYY-MM-DD (UTC)."

    limit = max(1, min(50, int(limit)))

    items, err, effective_date = fetch_hf_daily(date_iso, limit)
    if err is not None:
        box = (
            "<div class='error-box'>"
            "<b>API request failed</b><br/>"
            f"Status: {esc(str(err.status_code))}<br/>"
            f"Reason: {esc(err.reason)}<br/>"
            f"Body (snippet): {esc(err.body_snippet)}"
            "</div>"
        )
        return box, "❌ Failed to fetch daily papers."

    assert items is not None
    items = apply_search(items, query)

    if not items:
        return "<div class='paper-list'></div>", "No results."

    # ids
    paper_ids: List[str] = []
    for it in items:
        paper = (it or {}).get("paper") or {}
        pid = str(paper.get("id") or it.get("id") or "").strip()
        if pid:
            paper_ids.append(pid)

    # extras concurrent
    extras_map: Dict[str, PaperExtras] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_paper_extras, pid): pid for pid in paper_ids}
        for f in as_completed(futs):
            pid = futs[f]
            try:
                extras_map[pid] = f.result()
            except Exception:
                extras_map[pid] = PaperExtras(None, None, None, None, None, None)

    # arXiv meta
    arxiv_meta_map = fetch_arxiv_batch([pid for pid in paper_ids if is_arxiv_id(pid)])

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

        # abstract
        abstract_full = meta.abstract if (meta and meta.abstract) else normalize_text(str(it.get("summary") or paper.get("summary") or ""))
        preview, truncated = abstract_preview(abstract_full, max_chars=520)

        exx = extras_map.get(pid)

        # signals (explicit labels)
        pills: List[str] = []
        if exx and exx.project_page_url:
            pills.append("<span class='pill'>Project page</span>")
        if exx and exx.github_repo_url:
            if exx.github_stars is not None:
                pills.append(f"<span class='pill'>GitHub ★ {esc(str(exx.github_stars))}</span>")
            else:
                pills.append("<span class='pill'>GitHub</span>")

        # hub assets counts (only if we could extract)
        if exx and (exx.citing_models is not None or exx.citing_datasets is not None or exx.citing_spaces is not None):
            cm = "—" if exx.citing_models is None else str(exx.citing_models)
            cd = "—" if exx.citing_datasets is None else str(exx.citing_datasets)
            cs = "—" if exx.citing_spaces is None else str(exx.citing_spaces)
            pills.append(f"<span class='pill'>Hub assets: Models {esc(cm)} · Datasets {esc(cd)} · Spaces {esc(cs)}</span>")

        signal_row = f"<div class='meta'>{''.join(pills)}</div>" if pills else ""

        # links (only confirmed)
        links: List[str] = []
        if pid:
            links.append(f"<a href='{esc(HF_PAPER_PAGE.format(paper_id=pid))}' target='_blank' rel='noopener'>HF page</a>")
        if pid and is_arxiv_id(pid):
            links.append(f"<a href='{esc(ARXIV_ABS.format(paper_id=pid))}' target='_blank' rel='noopener'>arXiv</a>")
            links.append(f"<a href='{esc(ARXIV_PDF.format(paper_id=pid))}' target='_blank' rel='noopener'>PDF</a>")
        if exx and exx.project_page_url:
            links.append(f"<a href='{esc(exx.project_page_url)}' target='_blank' rel='noopener'>Project page</a>")
        if exx and exx.github_repo_url:
            links.append(f"<a href='{esc(exx.github_repo_url)}' target='_blank' rel='noopener'>GitHub</a>")

        link_row = f"<div class='links'>{''.join(links)}</div>"

        # details: full abstract + bibtex
        details = ""
        if truncated or (meta is not None):
            bib = build_bibtex(meta) if meta else ""
            details = (
                "<details class='more'>"
                "<summary>Details</summary>"
                "<div class='more-block'>"
                "<div><b>Full abstract</b></div>"
                f"<div class='abstract-preview'>{esc(abstract_full) if abstract_full else esc('Abstract not available.')}</div>"
                + (f"<div style='margin-top:10px;'><b>BibTeX</b></div><pre class='bibtex'>{esc(bib)}</pre>" if bib else "")
                + "</div></details>"
            )

        blocks.append(
            "<article class='card'>"
            "<div class='topline'>"
            f"<div class='title'>{idx}. {esc(title)}</div>"
            f"<div class='badge'>Upvotes {esc(upvotes_text)}</div>"
            "</div>"
            f"<div class='meta'><span class='pill'>Authors: {esc(authors_text)}</span></div>"
            f"{signal_row}"
            "<div class='abstract'>"
            "<h4>Abstract</h4>"
            f"<div class='abstract-preview'>{esc(preview) if preview else esc('Abstract not available.')}</div>"
            f"{details}"
            "</div>"
            f"{link_row}"
            "</article>"
        )

    blocks.append("</div>")

    status = f"✅ Loaded {len(items)} paper(s) for {effective_date} (UTC)."
    if effective_date != date_iso:
        status += f" (requested {date_iso} was not available yet; auto-clamped)"
    return "\n".join(blocks), status


# =========================
# Gradio UI (no accordion / simple search)
# =========================

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
                date_in = gr.Textbox(label="Date (YYYY-MM-DD, UTC)", value=utc_today_iso(), max_lines=1)
                limit_in = gr.Slider(label="Limit", minimum=1, maximum=50, step=1, value=10)
                query_in = gr.Textbox(label="Search (title / authors)", value="", max_lines=1)
                fetch_btn = gr.Button("Fetch", variant="primary")

            gr.HTML(
                "<div class='note'>"
                "<b>Note:</b> Date is interpreted in <b>UTC</b>. If you enter a future UTC date, the app will use the latest available date."
                "</div>"
            )

        status_out = gr.Markdown()
        html_out = gr.HTML()

        fetch_btn.click(fn=render, inputs=[date_in, limit_in, query_in], outputs=[html_out, status_out])
        demo.load(fn=render, inputs=[date_in, limit_in, query_in], outputs=[html_out, status_out])

    return demo


demo = build_app()

if __name__ == "__main__":
    demo.queue().launch(css=CSS, theme=gr.themes.Soft(), ssr_mode=False)
