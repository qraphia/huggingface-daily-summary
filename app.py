import datetime as dt
import html
import json
import os
import re
import time
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import gradio as gr
import requests

# =========================
# Config
# =========================

APP_TITLE = "Hugging Face Daily Summary"
APP_SUBTITLE = "Glance-first daily briefing with full abstract + BibTeX."

HF_DAILY_PAPERS_API = "https://huggingface.co/api/daily_papers"
HF_PAPER_API = "https://huggingface.co/api/papers/{paper_id}"
HF_PAPER_PAGE = "https://huggingface.co/papers/{paper_id}"

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_ABS = "https://arxiv.org/abs/{paper_id}"
ARXIV_PDF = "https://arxiv.org/pdf/{paper_id}.pdf"

HTTP_TIMEOUT_S = int(os.getenv("HTTP_TIMEOUT_S", "25"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
BACKOFF_BASE_S = float(os.getenv("BACKOFF_BASE_S", "1.0"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "6"))

HF_CACHE_TTL_S = int(os.getenv("HF_CACHE_TTL_S", "300"))          # 5 min
ARXIV_CACHE_TTL_S = int(os.getenv("ARXIV_CACHE_TTL_S", "86400"))  # 24 h
EXTRAS_CACHE_TTL_S = int(os.getenv("PAPER_EXTRAS_TTL_S", "86400"))# 24 h

HF_CACHE_MAX = int(os.getenv("HF_CACHE_MAX", "64"))
ARXIV_CACHE_MAX = int(os.getenv("ARXIV_CACHE_MAX", "512"))
EXTRAS_CACHE_MAX = int(os.getenv("EXTRAS_CACHE_MAX", "512"))

HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "qraphia/huggingface-daily-summary (gradio; requests)",
        "Accept": "application/json",
    }
)
if HF_TOKEN:
    SESSION.headers["Authorization"] = f"Bearer {HF_TOKEN}"

RE_ARXIV_NEW = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
RE_ARXIV_OLD = re.compile(r"^[a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$", re.IGNORECASE)
RE_MAX_DATE = re.compile(r'must be less than or equal to "([^"]+)"')

# HF paper page HTML extraction (best-effort fallback)
RE_PROJECT_PAGE = re.compile(r'href="([^"]+)"[^>]*>\s*Project page\s*<', re.IGNORECASE)
RE_GITHUB = re.compile(r'href="([^"]+github\.com[^"]+)"[^>]*>\s*GitHub\s*<', re.IGNORECASE)

# =========================
# CSS (BibTeX: dark + selectable; no-JS accurate hint)
# =========================

CSS = """
:root{
  --hf-yellow:#FFD21E; --hf-orange:#FF9D00; --hf-gray:#6B7280;
  --bg:#ffffff; --surface:#ffffff; --muted-bg:#f6f7f9;
  --border:#e5e7eb; --border-strong:#d1d5db;
  --text:#111827; --muted:#6B7280; --link:#0b5fff;
  --shadow: 0 10px 30px rgba(17,24,39,0.10);
  --shadow-sm: 0 4px 14px rgba(17,24,39,0.08);
}

body{
  background:
    radial-gradient(900px 320px at 15% 0%, rgba(255,157,0,0.22), transparent 58%),
    radial-gradient(900px 320px at 85% 0%, rgba(255,210,30,0.20), transparent 58%),
    var(--bg);
  color: var(--text);
}

.gradio-container{ max-width: 1120px !important; }
footer{ display:none !important; }

.hero{
  margin:14px 0 10px; padding:16px;
  border:1px solid var(--border-strong); border-radius:18px;
  background: linear-gradient(180deg, #fff, #fff8e0);
  box-shadow: var(--shadow-sm);
}
.hero h1{ margin:0; font-size:1.55rem; letter-spacing:-0.02em; }
.hero .sub{ margin-top:6px; color:var(--muted); font-size:0.98rem; line-height:1.35; }

#toolbar{
  position: sticky; top:10px; z-index:30;
  margin:10px 0; padding:10px;
  border:1px solid var(--border-strong); border-radius:18px;
  background: rgba(255,255,255,0.96);
  backdrop-filter: blur(10px);
  box-shadow: var(--shadow);
}

#toolbar input, #toolbar select{
  border-radius:14px !important;
  border:1px solid var(--border-strong) !important;
  background:#fff !important;
}
#toolbar input:focus, #toolbar select:focus{
  border-color: rgba(255,157,0,0.60) !important;
  box-shadow: 0 0 0 4px rgba(255,157,0,0.18) !important;
}

#fetch_btn{
  border-radius:14px !important;
  border:1px solid rgba(255,157,0,0.65) !important;
  background: linear-gradient(180deg, var(--hf-orange), #ffb84a) !important;
  color:#111827 !important;
  font-weight: 950 !important;
  box-shadow: 0 12px 22px rgba(255,157,0,0.24) !important;
}
#fetch_btn:hover{ transform: translateY(-1px); box-shadow: 0 16px 28px rgba(255,157,0,0.28) !important; }

.kpis{ margin-top:10px; display:flex; flex-wrap:wrap; gap:8px; }
.kpi{
  display:inline-flex; gap:8px; align-items:center;
  padding:6px 10px; border:1px solid var(--border-strong);
  border-radius:999px; background: rgba(255,255,255,0.70);
  box-shadow: 0 1px 0 rgba(17,24,39,0.03);
  font-size:0.90rem; font-weight:950;
}
.kpi span{ color: var(--muted); font-weight:950; }

.highlights{ margin-top:12px; display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:12px; }
@media (max-width:980px){ .highlights{ grid-template-columns:1fr; } }

.hcard{
  border:1px solid var(--border-strong);
  border-radius:18px; padding:12px;
  background:var(--surface); box-shadow: var(--shadow-sm);
}
.hcard h3{ margin:0 0 8px; font-size:0.95rem; letter-spacing:-0.01em; }
.hitem{
  display:flex; justify-content:space-between; gap:10px;
  padding:8px 0; border-top:1px solid var(--border);
}
.hitem:first-of-type{ border-top:none; }
.hitem a{
  color: var(--text); text-decoration:none; font-weight:950;
  font-size:0.92rem; line-height:1.25;
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;
}
.hitem a:hover{ text-decoration:underline; }
.hval{ color:var(--muted); font-weight:950; white-space:nowrap; }

.feed{ margin-top:14px; display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:12px; }
@media (max-width:980px){ .feed{ grid-template-columns:1fr; } }

.card{
  border:1px solid var(--border);
  border-radius:18px; padding:14px;
  background: var(--surface);
  box-shadow: var(--shadow-sm);
}
.card.hot{
  border-color: rgba(255,157,0,0.55);
  box-shadow: 0 16px 34px rgba(255,157,0,0.18);
}
.card.oss{
  background: linear-gradient(180deg, #fff, #f8fbff);
  border-color: rgba(11,95,255,0.18);
}

.topline{ display:flex; justify-content:space-between; gap:12px; align-items:flex-start; }
.title{ font-weight: 980; font-size:1.02rem; line-height:1.25; }
.badge{
  border-radius: 999px; padding: 6px 10px;
  border: 1px solid var(--border);
  background: var(--muted-bg);
  font-weight: 980; font-size: 0.84rem; white-space:nowrap;
}

.meta{ margin-top:10px; display:flex; flex-wrap:wrap; gap:8px; }
.pill{
  display:inline-flex; align-items:center; gap:6px;
  padding:6px 10px; border-radius:999px;
  background: rgba(255,210,30,0.25);
  border:1px solid rgba(255,210,30,0.50);
  color: var(--text); font-weight: 950; font-size:0.84rem;
}
.pill.dim{
  background: rgba(107,114,128,0.10);
  border:1px solid rgba(107,114,128,0.22);
  color: var(--muted);
}

.abs{ margin-top:12px; padding-top:12px; border-top:1px solid var(--border); }
.abs .label{
  color: var(--muted); font-weight:980; font-size:0.84rem;
  letter-spacing:0.02em; text-transform:uppercase;
}
.abs .txt{
  margin-top:8px; color: var(--text); font-size:0.95rem; line-height:1.55;
  display:-webkit-box; -webkit-line-clamp:4; -webkit-box-orient:vertical; overflow:hidden;
}

.links{ margin-top:12px; display:flex; flex-wrap:wrap; gap:12px; align-items:center; }
.links a{ color: var(--link); text-decoration:none; font-weight:980; font-size:0.92rem; }
.links a:hover{ text-decoration:underline; }

/* BibTeX popover: dark + selectable; no-JS hint that is true */
details.biblink{ position: relative; display:inline-block; }
details.biblink summary{
  list-style:none; cursor:pointer; color: var(--link);
  font-weight:980; font-size:0.92rem;
}
details.biblink summary::-webkit-details-marker{ display:none; }
details.biblink[open] summary{ text-decoration: underline; }

.bibpop{
  position:absolute; right:0; top:26px; z-index:50;
  width: min(560px, 78vw);
  background:#0b1220; color:#e5e7eb;
  border:1px solid rgba(255,255,255,0.10);
  border-radius:14px; box-shadow: 0 16px 34px rgba(0,0,0,0.25);
  padding:10px;
}
.bibpop .hint{
  color: rgba(229,231,235,0.72);
  font-weight: 900; font-size:0.80rem;
  margin-bottom:8px;
}
pre.bib{
  margin:0; padding:10px; border-radius:12px;
  background:#000; border:1px solid rgba(255,255,255,0.12);
  color:#fff; overflow-x:auto;
  font-size:0.86rem; line-height:1.45;
  user-select:text; -webkit-user-select:text;
  cursor:text;
}

details.more summary{
  cursor:pointer; margin-top:12px; color: var(--link);
  font-weight:980; font-size:0.90rem;
}
.moreblock{ margin-top:10px; padding-top:10px; border-top:1px solid var(--border); }
.full{ white-space: pre-wrap; line-height:1.6; }

.error{
  border:1px solid rgba(239,68,68,0.30);
  background: rgba(239,68,68,0.06);
  border-radius: 14px;
  padding: 12px;
  color:#991b1b;
}
"""

# =========================
# Models
# =========================

@dataclass(frozen=True)
class ApiError:
    status_code: Optional[int]
    reason: str
    body_snippet: str

@dataclass(frozen=True)
class ArxivMeta:
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    year: str

@dataclass(frozen=True)
class PaperExtras:
    project_page_url: Optional[str]
    github_repo_url: Optional[str]
    github_stars: Optional[int]

# =========================
# Thread-safe TTL Cache with max-size bound
# =========================

class TTLCache:
    def __init__(self, ttl_s: int, max_items: int):
        self.ttl_s = max(1, int(ttl_s))
        self.max_items = max(1, int(max_items))
        self._lock = threading.Lock()
        self._data: "OrderedDict[Any, Tuple[float, Any]]" = OrderedDict()

    def get(self, key: Any) -> Optional[Any]:
        now = time.time()
        with self._lock:
            hit = self._data.get(key)
            if not hit:
                return None
            ts, val = hit
            if now - ts > self.ttl_s:
                self._data.pop(key, None)
                return None
            # refresh LRU position
            self._data.move_to_end(key, last=True)
            return val

    def set(self, key: Any, value: Any) -> None:
        now = time.time()
        with self._lock:
            self._data[key] = (now, value)
            self._data.move_to_end(key, last=True)
            self._evict_locked(now)

    def _evict_locked(self, now: float) -> None:
        # 1) drop expired (from oldest)
        while self._data:
            (k, (ts, _)) = next(iter(self._data.items()))
            if now - ts <= self.ttl_s:
                break
            self._data.popitem(last=False)

        # 2) enforce max size
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)

HF_CACHE = TTLCache(HF_CACHE_TTL_S, HF_CACHE_MAX)
ARXIV_CACHE = TTLCache(ARXIV_CACHE_TTL_S, ARXIV_CACHE_MAX)
EXTRAS_CACHE = TTLCache(EXTRAS_CACHE_TTL_S, EXTRAS_CACHE_MAX)

# =========================
# Helpers
# =========================

def esc(s: str) -> str:
    return html.escape(s or "", quote=True)

def utc_today_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()

def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def normalize_text(s: str) -> str:
    if not s:
        return ""
    out = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    while "\n\n\n" in out:
        out = out.replace("\n\n\n", "\n\n")
    return out.strip()

def is_arxiv_id(x: str) -> bool:
    x = (x or "").strip()
    return bool(RE_ARXIV_NEW.match(x) or RE_ARXIV_OLD.match(x))

def strip_arxiv_version(aid: str) -> str:
    # keep BibTeX eprint stable (no vN)
    return re.sub(r"v\d+$", "", (aid or "").strip())

def normalize_ids(raw_id: str) -> Tuple[str, Optional[str]]:
    """
    Returns:
      hf_id    : used for HF URLs (/papers/{hf_id})
      arxiv_id : used for arXiv API / BibTeX, if available
    """
    rid = (raw_id or "").strip()
    if rid.startswith("arxiv:"):
        aid = rid.split(":", 1)[1].strip()
        return aid, (aid if is_arxiv_id(aid) else None)
    return rid, (rid if is_arxiv_id(rid) else None)

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
    msg = ""
    try:
        obj = json.loads(body_text)
        if isinstance(obj, dict):
            msg = str(obj.get("error") or "")
    except Exception:
        msg = body_text or ""
    m = RE_MAX_DATE.search(msg)
    if not m:
        return None
    iso_dt = m.group(1).strip()
    if "T" in iso_dt:
        return iso_dt.split("T", 1)[0]
    return iso_dt[:10] if len(iso_dt) >= 10 else None

def backoff_sleep(attempt: int, cap_s: float = 20.0) -> None:
    t = min(cap_s, BACKOFF_BASE_S * (2 ** attempt))
    time.sleep(t)

# =========================
# HTTP (retry-normalized)
# =========================

_TRANSIENT = {429, 500, 502, 503, 504}

def http_get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: int = HTTP_TIMEOUT_S,
) -> Tuple[Optional[requests.Response], str]:
    last_text = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = SESSION.get(url, params=params, headers=headers, timeout=timeout_s)
            last_text = r.text or ""
        except requests.RequestException:
            backoff_sleep(attempt, cap_s=20.0)
            continue

        if r.status_code == 200:
            return r, last_text

        if r.status_code in _TRANSIENT:
            backoff_sleep(attempt, cap_s=20.0)
            continue

        # non-transient error
        return r, last_text

    return None, last_text

def safe_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[ApiError], str]:
    r, text = http_get(url, params=params, headers={"Accept": "application/json"})
    if r is None:
        return None, ApiError(None, "Request failed", (text or "")[:600]), text
    if r.status_code != 200:
        return None, ApiError(r.status_code, r.reason or "HTTP error", (text or "")[:600]), text
    try:
        return r.json(), None, ""
    except Exception:
        return None, ApiError(200, "Invalid JSON", (text or "")[:600]), text

def safe_get_text(url: str) -> Optional[str]:
    # IMPORTANT: this actually retries (the old code did not).
    r, text = http_get(url, params=None, headers={"Accept": "text/html,application/xhtml+xml"})
    if r is None or r.status_code != 200:
        return None
    return text or ""

def preview_text(s: str, max_chars: int = 640) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return s if len(s) <= max_chars else s[: max_chars - 1].rstrip() + "…"

# =========================
# Fetch HF daily list (auto clamp)
# =========================

def fetch_hf_daily(date_iso: str, limit: int) -> Tuple[Optional[List[Dict[str, Any]]], Optional[ApiError], str]:
    date_iso = (date_iso or "").strip()
    limit = int(limit)

    cache_key = (date_iso, limit)
    cached = HF_CACHE.get(cache_key)
    if cached is not None:
        return cached, None, date_iso

    params = {"date": date_iso, "limit": limit, "sort": "trending"}
    data, err, raw = safe_get_json(HF_DAILY_PAPERS_API, params=params)

    if err is not None and err.status_code == 400:
        max_date = parse_max_allowed_date_from_error(raw)
        if max_date and max_date != date_iso:
            params2 = {"date": max_date, "limit": limit, "sort": "trending"}
            data2, err2, _ = safe_get_json(HF_DAILY_PAPERS_API, params=params2)
            if err2 is None and isinstance(data2, list):
                HF_CACHE.set((max_date, limit), data2)
                return data2, None, max_date
            return None, err2, max_date

    if err is not None:
        return None, err, date_iso
    if not isinstance(data, list):
        return None, ApiError(200, "Unexpected payload", str(type(data))[:200]), date_iso

    HF_CACHE.set(cache_key, data)
    return data, None, date_iso

# =========================
# Paper extras (Project/GitHub)
# =========================

def fetch_paper_extras(hf_id: str) -> PaperExtras:
    hf_id = (hf_id or "").strip()
    if not hf_id:
        return PaperExtras(None, None, None)

    cached = EXTRAS_CACHE.get(hf_id)
    if cached is not None:
        return cached

    project_page_url: Optional[str] = None
    github_repo_url: Optional[str] = None
    github_stars: Optional[int] = None

    # Prefer HF JSON API
    obj, err, _ = safe_get_json(HF_PAPER_API.format(paper_id=hf_id), params=None)
    if err is None and isinstance(obj, dict) and "error" not in obj:
        gr_repo = obj.get("githubRepo")
        if isinstance(gr_repo, str) and gr_repo.startswith("http") and "github.com" in gr_repo:
            github_repo_url = gr_repo.strip()

        gs = obj.get("githubStars")
        try:
            if gs is not None:
                github_stars = int(gs)
        except Exception:
            github_stars = None

        for k in ("projectPage", "projectPageUrl", "project_url", "projectUrl", "project_page_url"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                project_page_url = ensure_url(v.strip())
                break

    # HTML fallback (works only if safe_get_text actually retries)
    page = safe_get_text(HF_PAPER_PAGE.format(paper_id=hf_id))
    if page:
        if project_page_url is None:
            m = RE_PROJECT_PAGE.search(page)
            if m:
                project_page_url = ensure_url(m.group(1))
        if github_repo_url is None:
            m = RE_GITHUB.search(page)
            if m:
                github_repo_url = ensure_url(m.group(1))

    ex = PaperExtras(project_page_url, github_repo_url, github_stars)
    EXTRAS_CACHE.set(hf_id, ex)
    return ex

# =========================
# arXiv meta (abstract + BibTeX)
# =========================

def parse_arxiv_atom(xml_text: str) -> Dict[str, ArxivMeta]:
    out: Dict[str, ArxivMeta] = {}
    ns = {"a": "http://www.w3.org/2005/Atom"}

    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return out

    for entry in root.findall("a:entry", ns):
        raw_id = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1].strip() if "/abs/" in raw_id else ""
        if not arxiv_id:
            continue

        title = normalize_text(entry.findtext("a:title", default="", namespaces=ns) or "")
        abstract = normalize_text(entry.findtext("a:summary", default="", namespaces=ns) or "")
        published = (entry.findtext("a:published", default="", namespaces=ns) or "").strip()
        year = published[:4] if len(published) >= 4 else ""

        authors: List[str] = []
        for a in entry.findall("a:author", ns):
            name = (a.findtext("a:name", default="", namespaces=ns) or "").strip()
            if name:
                authors.append(name)

        out[arxiv_id] = ArxivMeta(arxiv_id=arxiv_id, title=title, abstract=abstract, authors=authors, year=year)

    return out

def chunked(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]

def fetch_arxiv_batch(arxiv_ids: List[str]) -> Dict[str, ArxivMeta]:
    result: Dict[str, ArxivMeta] = {}
    wanted: List[str] = []

    for aid in arxiv_ids:
        aid = (aid or "").strip()
        if not aid or not is_arxiv_id(aid):
            continue
        cached = ARXIV_CACHE.get(aid)
        if cached is not None:
            result[aid] = cached
        else:
            wanted.append(aid)

    for group in chunked(wanted, 25):
        r, text = http_get(ARXIV_API, params={"id_list": ",".join(group)}, headers={"Accept": "application/atom+xml"})
        if r is None or r.status_code != 200:
            continue
        parsed = parse_arxiv_atom(text or "")
        for k, v in parsed.items():
            result[k] = v
            ARXIV_CACHE.set(k, v)

    return result

def build_bibtex(meta: ArxivMeta) -> str:
    base_id = strip_arxiv_version(meta.arxiv_id)
    key = base_id.replace("/", "_")
    authors = " and ".join(meta.authors) if meta.authors else "Unknown"
    title = (meta.title or "").replace("{", "\\{").replace("}", "\\}")
    year = meta.year or "????"
    url = ARXIV_ABS.format(paper_id=base_id)

    return (
        f"@misc{{arxiv:{key},\n"
        f"  title={{ {title} }},\n"
        f"  author={{ {authors} }},\n"
        f"  year={{ {year} }},\n"
        f"  eprint={{ {base_id} }},\n"
        f"  archivePrefix={{arXiv}},\n"
        f"  url={{ {url} }}\n"
        f"}}"
    )

# =========================
# Rendering
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

def build_highlight_card(title: str, rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return f"<div class='hcard'><h3>{esc(title)}</h3><div style='color:var(--muted)'>—</div></div>"
    lines: List[str] = []
    for r in rows:
        href = HF_PAPER_PAGE.format(paper_id=r["hf_id"])
        lines.append(
            "<div class='hitem'>"
            f"<a href='{esc(href)}' target='_blank' rel='noopener'>{esc(r['title'])}</a>"
            f"<div class='hval'>{esc(str(r['upvotes']))}</div>"
            "</div>"
        )
    return f"<div class='hcard'><h3>{esc(title)}</h3>{''.join(lines)}</div>"

def match_query(title: str, abstract: str, keywords: List[str], query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return True
    hay = " ".join([title or "", abstract or "", " ".join(keywords or [])]).lower()
    # OR semantics: any token hit
    tokens = [t for t in re.split(r"[,\s]+", q) if t]
    return any(t in hay for t in tokens)

def render(date_iso: str, limit_choice: str, query: str) -> Tuple[str, str]:
    date_iso = (date_iso or "").strip()
    try:
        dt.date.fromisoformat(date_iso)
    except Exception:
        return "", "❌ Invalid date. Use YYYY-MM-DD (UTC)."

    limit = max(1, min(50, safe_int(limit_choice, 10)))

    items, err, effective = fetch_hf_daily(date_iso, limit)
    if err is not None:
        return (
            "<div class='error'><b>API request failed</b><br/>"
            f"Status: {esc(str(err.status_code))}<br/>"
            f"Reason: {esc(err.reason)}<br/>"
            f"Body (snippet): {esc(err.body_snippet)}</div>",
            "❌ Failed to fetch daily papers.",
        )
    assert items is not None
    if not items:
        return "<div class='feed'></div>", "No results."

    # Collect rows
    rows: List[Dict[str, Any]] = []
    hf_ids: List[str] = []
    arxiv_ids: List[str] = []
    upvotes_all: List[int] = []
    topic_counter: Dict[str, int] = {}

    for it in items:
        paper = (it or {}).get("paper") or {}
        raw_id = str(paper.get("id") or it.get("id") or "").strip()
        hf_id, arxiv_id = normalize_ids(raw_id)

        title = str(it.get("title") or paper.get("title") or "").strip() or "(no title)"
        up = safe_int(paper.get("upvotes"), 0)
        upvotes_all.append(up)

        kws = paper.get("ai_keywords") or []
        kw_list: List[str] = []
        if isinstance(kws, list):
            for k in kws[:10]:
                ks = str(k).strip()
                if ks:
                    kw_list.append(ks)
                    topic_counter[ks] = topic_counter.get(ks, 0) + 1

        rows.append(
            {
                "hf_id": hf_id,
                "arxiv_id": arxiv_id,
                "title": title,
                "upvotes": up,
                "paper": paper,
                "item": it,
                "keywords": kw_list,
            }
        )
        if hf_id:
            hf_ids.append(hf_id)
        if arxiv_id:
            arxiv_ids.append(arxiv_id)

    # De-dup ids to avoid redundant calls
    hf_ids_u = sorted(set(hf_ids))
    arxiv_ids_u = sorted(set(arxiv_ids))

    # Extras concurrent
    extras_map: Dict[str, PaperExtras] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_paper_extras, hid): hid for hid in hf_ids_u}
        for f in as_completed(futs):
            hid = futs[f]
            try:
                extras_map[hid] = f.result()
            except Exception:
                extras_map[hid] = PaperExtras(None, None, None)

    # arXiv meta
    arxiv_map = fetch_arxiv_batch(arxiv_ids_u)

    # Filter after meta is available (query can match abstract too)
    filtered: List[Dict[str, Any]] = []
    for r in rows:
        meta = arxiv_map.get(r["arxiv_id"]) if r["arxiv_id"] else None
        abs_full = meta.abstract if (meta and meta.abstract) else normalize_text(str(r["item"].get("summary") or r["paper"].get("summary") or ""))
        if match_query(r["title"], abs_full, r["keywords"], query):
            filtered.append(r)

    if not filtered:
        return "<div class='feed'></div>", "No results (after filter)."

    # KPI / brief
    total = len(filtered)
    with_project = 0
    with_github = 0
    for r in filtered:
        exx = extras_map.get(r["hf_id"])
        if exx and exx.project_page_url:
            with_project += 1
        if exx and exx.github_repo_url:
            with_github += 1

    up_sorted = sorted([r["upvotes"] for r in filtered])
    hot_threshold = up_sorted[max(0, int(0.80 * (len(up_sorted) - 1)))] if up_sorted else 0
    top_topics = sorted(topic_counter.items(), key=lambda kv: kv[1], reverse=True)[:6]
    topics_str = ", ".join([t for t, _ in top_topics]) if top_topics else "—"

    hero = (
        "<div class='hero'>"
        f"<h1>{esc(APP_TITLE)} — {esc(effective)} (UTC)</h1>"
        f"<div class='sub'>{esc(APP_SUBTITLE)}</div>"
        "<div class='kpis'>"
        f"<div class='kpi'><span>Total</span>{total}</div>"
        f"<div class='kpi'><span>Project page</span>{with_project}</div>"
        f"<div class='kpi'><span>GitHub</span>{with_github}</div>"
        f"<div class='kpi'><span>HOT ≥</span>{hot_threshold}</div>"
        f"<div class='kpi'><span>Top topics</span>{esc(topics_str)}</div>"
        "</div>"
        "</div>"
    )

    # Highlights
    top_up = sorted(filtered, key=lambda r: r["upvotes"], reverse=True)[:3]
    with_proj: List[Dict[str, Any]] = []
    with_gh: List[Dict[str, Any]] = []
    for r in sorted(filtered, key=lambda r: r["upvotes"], reverse=True):
        exx = extras_map.get(r["hf_id"])
        if exx and exx.project_page_url and len(with_proj) < 3:
            with_proj.append(r)
        if exx and exx.github_repo_url and len(with_gh) < 3:
            with_gh.append(r)
        if len(with_proj) >= 3 and len(with_gh) >= 3:
            break

    highlights = (
        "<div class='highlights'>"
        + build_highlight_card("Top Upvotes", top_up)
        + build_highlight_card("Top with Project page", with_proj)
        + build_highlight_card("Top with GitHub", with_gh)
        + "</div>"
    )

    # Cards
    cards: List[str] = ["<div class='feed'>"]
    for r in filtered:
        hf_id = r["hf_id"]
        arxiv_id = r["arxiv_id"]
        title = r["title"]
        upvotes = r["upvotes"]
        paper = r["paper"]
        it = r["item"]
        exx = extras_map.get(hf_id)

        meta = arxiv_map.get(arxiv_id) if arxiv_id else None
        authors = meta.authors if (meta and meta.authors) else get_hf_authors(paper)
        authors_text = format_authors(authors)

        full_abs = meta.abstract if (meta and meta.abstract) else normalize_text(str(it.get("summary") or paper.get("summary") or ""))
        prev_abs = preview_text(full_abs, 640) if full_abs else ""

        pills = [
            "<span class='pill dim'>Authors</span>",
            f"<span class='pill'>{esc(authors_text)}</span>",
        ]
        if exx and exx.project_page_url:
            pills.append("<span class='pill'>Project page</span>")
        if exx and exx.github_repo_url:
            if exx.github_stars is not None:
                pills.append(f"<span class='pill'>GitHub ★ {esc(str(exx.github_stars))}</span>")
            else:
                pills.append("<span class='pill'>GitHub</span>")

        cls = "card"
        if hot_threshold > 0 and upvotes >= hot_threshold:
            cls += " hot"
        if exx and (exx.project_page_url or exx.github_repo_url):
            cls += " oss"

        links: List[str] = []
        if hf_id:
            links.append(f"<a href='{esc(HF_PAPER_PAGE.format(paper_id=hf_id))}' target='_blank' rel='noopener'>HF page</a>")
        if arxiv_id:
            base_id = strip_arxiv_version(arxiv_id)
            links.append(f"<a href='{esc(ARXIV_ABS.format(paper_id=base_id))}' target='_blank' rel='noopener'>arXiv</a>")
            links.append(f"<a href='{esc(ARXIV_PDF.format(paper_id=base_id))}' target='_blank' rel='noopener'>PDF</a>")

        if exx and exx.project_page_url:
            links.append(f"<a href='{esc(exx.project_page_url)}' target='_blank' rel='noopener'>Project page</a>")

        if exx and exx.github_repo_url:
            links.append(f"<a href='{esc(exx.github_repo_url)}' target='_blank' rel='noopener'>GitHub</a>")

        if meta:
            bib = build_bibtex(meta)
            # no JS: user selects manually; closing is by clicking BibTeX again.
            links.append(
                "<details class='biblink'>"
                "<summary>BibTeX</summary>"
                "<div class='bibpop'>"
                "<div class='hint'>Select &amp; copy (no JS). Close: click “BibTeX” again.</div>"
                f"<pre class='bib'>{esc(bib)}</pre>"
                "</div></details>"
            )

        details = ""
        if full_abs:
            details = (
                "<details class='more'>"
                "<summary>Read full abstract</summary>"
                "<div class='moreblock'>"
                f"<div class='full'>{esc(full_abs)}</div>"
                "</div></details>"
            )

        cards.append(
            f"<article class='{cls}'>"
            "<div class='topline'>"
            f"<div class='title'>{esc(title)}</div>"
            f"<div class='badge'>Upvotes {esc(str(upvotes))}</div>"
            "</div>"
            f"<div class='meta'>{''.join(pills)}</div>"
            "<div class='abs'>"
            "<div class='label'>Abstract</div>"
            f"<div class='txt'>{esc(prev_abs) if prev_abs else esc('Abstract not available.')}</div>"
            f"{details}"
            "</div>"
            f"<div class='links'>{''.join(links)}</div>"
            "</article>"
        )

    cards.append("</div>")

    page = hero + highlights + "".join(cards)

    status = f"✅ Loaded {total} paper(s) for {effective} (UTC)."
    if effective != date_iso:
        status += f" (requested {date_iso} was not available yet; auto-clamped)"
    if (query or "").strip():
        status += f"  | filter: “{query.strip()}”"

    return page, status

# =========================
# Gradio UI
# =========================

def build_app() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE, css=CSS) as demo:
        gr.HTML(f"<div class='hero'><h1>{esc(APP_TITLE)}</h1><div class='sub'>{esc(APP_SUBTITLE)}</div></div>")

        with gr.Row(elem_id="toolbar"):
            date_in = gr.Textbox(
                label="",
                show_label=False,
                value=utc_today_iso(),
                max_lines=1,
                placeholder="Date (YYYY-MM-DD, UTC) e.g., 2026-01-30",
                scale=5,
            )
            limit_in = gr.Dropdown(
                label="",
                show_label=False,
                choices=["5", "10", "20", "30", "50"],
                value="10",
                allow_custom_value=False,
                scale=2,
            )
            query_in = gr.Textbox(
                label="",
                show_label=False,
                value="",
                max_lines=1,
                placeholder="Filter (optional): tokens separated by space/comma (matches title/abstract/keywords)",
                scale=6,
            )
            fetch_btn = gr.Button("Fetch", variant="primary", elem_id="fetch_btn", scale=2)

        status_out = gr.Markdown()
        html_out = gr.HTML()

        fetch_btn.click(fn=render, inputs=[date_in, limit_in, query_in], outputs=[html_out, status_out])
        demo.load(fn=render, inputs=[date_in, limit_in, query_in], outputs=[html_out, status_out])

    return demo

demo = build_app()

if __name__ == "__main__":
    demo.queue().launch()
