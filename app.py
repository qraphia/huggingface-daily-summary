import datetime as dt
import html
import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests

# =========================================================
# Config
# =========================================================

APP_TITLE = "Hugging Face Daily Summary"
APP_SUBTITLE = "Glance-first daily briefing with full abstract + BibTeX (no translation)."

HF_DAILY_PAPERS_API = "https://huggingface.co/api/daily_papers"
HF_PAPER_API = "https://huggingface.co/api/papers/{paper_id}"
HF_PAPER_PAGE = "https://huggingface.co/papers/{paper_id}"

# arXiv API docs show http endpoint; be robust in hosted env.
ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_ABS = "https://arxiv.org/abs/{paper_id}"
ARXIV_PDF = "https://arxiv.org/pdf/{paper_id}.pdf"

HTTP_TIMEOUT_S = int(os.getenv("HTTP_TIMEOUT_S", "25"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
BACKOFF_BASE_S = float(os.getenv("BACKOFF_BASE_S", "1.0"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "6"))

HF_CACHE_TTL_S = int(os.getenv("HF_CACHE_TTL_S", "300"))          # 5 min
ARXIV_CACHE_TTL_S = int(os.getenv("ARXIV_CACHE_TTL_S", "86400"))  # 24 h
EXTRAS_CACHE_TTL_S = int(os.getenv("EXTRAS_CACHE_TTL_S", "86400"))  # 24 h

HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "qraphia/huggingface-daily-summary (gradio; no-translation)",
        "Accept": "application/json",
    }
)
if HF_TOKEN:
    SESSION.headers["Authorization"] = f"Bearer {HF_TOKEN}"

RE_DATE_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RE_MAX_DATE = re.compile(r'must be less than or equal to "([^"]+)"')

RE_ARXIV_NEW = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
RE_ARXIV_OLD = re.compile(r"^[a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$", re.IGNORECASE)

# HF paper page HTML extraction (best-effort)
RE_PROJECT_PAGE = re.compile(r'href="([^"]+)"[^>]*>\s*Project page\s*<', re.IGNORECASE)
RE_GITHUB = re.compile(r'href="([^"]+github\.com[^"]+)"[^>]*>\s*GitHub\s*<', re.IGNORECASE)

# =========================================================
# Styling (HF-ish, higher contrast)
# =========================================================

CSS = """
:root{
  --hf-yellow:#FFD21E;
  --hf-orange:#FF9D00;
  --hf-black:#111827;
  --hf-gray:#6B7280;

  --bg:#f7f7fb;
  --surface:#ffffff;
  --surface-2:#fff7dc;
  --border:#d1d5db;
  --border-2:#bfc5cf;

  --text:#111827;
  --muted:#6B7280;
  --link:#0b5fff;

  --shadow: 0 14px 34px rgba(17,24,39,0.12);
  --shadow-sm: 0 6px 18px rgba(17,24,39,0.10);
}

body{
  background:
    radial-gradient(900px 380px at 15% -5%, rgba(255,157,0,0.32), transparent 60%),
    radial-gradient(900px 380px at 85% -5%, rgba(255,210,30,0.28), transparent 60%),
    var(--bg);
  color: var(--text);
}

.gradio-container{ max-width: 1120px !important; }
footer{ display:none !important; }

.hero{
  margin: 14px 0 10px;
  padding: 16px 16px;
  border: 1px solid var(--border-2);
  border-radius: 18px;
  background: linear-gradient(180deg, #fff, var(--surface-2));
  box-shadow: var(--shadow-sm);
  position: relative;
  overflow: hidden;
}
.hero:before{
  content:"";
  position:absolute;
  left:0; top:0; right:0;
  height: 6px;
  background: linear-gradient(90deg, var(--hf-orange), var(--hf-yellow));
}
.hero h1{
  margin:0;
  font-size: 1.55rem;
  letter-spacing:-0.02em;
  color: var(--hf-black);
  font-weight: 950;
}
.hero .sub{
  margin-top: 6px;
  color: var(--muted);
  font-size: 0.98rem;
  line-height: 1.35;
  font-weight: 600;
}

#toolbar{
  position: sticky;
  top: 10px;
  z-index: 30;
  margin: 10px 0 12px;
  padding: 12px 12px;
  border: 1px solid var(--border-2);
  border-radius: 18px;
  background: rgba(255,255,255,0.98);
  backdrop-filter: blur(10px);
  box-shadow: var(--shadow);
}

#toolbar .gr-form{
  gap: 10px !important;
}

#date_in input, #limit_in select{
  border-radius: 14px !important;
  border: 1px solid var(--border-2) !important;
  background: #fff !important;
  font-weight: 750 !important;
  color: var(--hf-black) !important;
}
#date_in input:focus, #limit_in select:focus{
  border-color: rgba(255,157,0,0.85) !important;
  box-shadow: 0 0 0 4px rgba(255,157,0,0.22) !important;
}

#fetch_btn{
  border-radius: 14px !important;
  border: 1px solid rgba(17,24,39,0.25) !important;
  background: linear-gradient(180deg, var(--hf-yellow), #ffe37b) !important;
  color: var(--hf-black) !important;
  font-weight: 950 !important;
  box-shadow: 0 14px 24px rgba(255,210,30,0.32) !important;
}
#fetch_btn:hover{
  transform: translateY(-1px);
  box-shadow: 0 18px 30px rgba(255,210,30,0.38) !important;
}

.status{
  margin-top: 8px;
  padding: 10px 12px;
  border-radius: 14px;
  border: 1px solid var(--border-2);
  background: rgba(255,255,255,0.95);
  font-weight: 750;
}

.kpis{
  margin-top: 10px;
  display:flex;
  flex-wrap:wrap;
  gap: 8px;
}
.kpi{
  display:inline-flex;
  gap: 8px;
  align-items:center;
  padding: 6px 10px;
  border: 1px solid var(--border-2);
  border-radius: 999px;
  background: rgba(255,255,255,0.75);
  box-shadow: 0 1px 0 rgba(17,24,39,0.03);
  font-size: 0.90rem;
  font-weight: 950;
}
.kpi span{ color: var(--muted); font-weight: 950; }

.highlights{
  margin-top: 12px;
  display:grid;
  grid-template-columns: repeat(3, minmax(0,1fr));
  gap: 12px;
}
@media (max-width: 980px){
  .highlights{ grid-template-columns: 1fr; }
}

.hcard{
  border: 1px solid var(--border-2);
  border-radius: 18px;
  padding: 12px 12px;
  background: var(--surface);
  box-shadow: var(--shadow-sm);
}
.hcard h3{
  margin:0 0 8px;
  font-size: 0.95rem;
  letter-spacing: -0.01em;
  font-weight: 950;
}
.hitem{
  display:flex;
  justify-content: space-between;
  gap: 10px;
  padding: 8px 0;
  border-top: 1px solid rgba(209,213,219,0.8);
}
.hitem:first-of-type{ border-top: none; }
.hitem a{
  color: var(--text);
  text-decoration:none;
  font-weight: 900;
  font-size: 0.92rem;
  line-height: 1.25;
  display:-webkit-box;
  -webkit-line-clamp:2;
  -webkit-box-orient:vertical;
  overflow:hidden;
}
.hitem a:hover{ text-decoration: underline; }
.hval{
  color: var(--muted);
  font-weight: 950;
  white-space: nowrap;
}

.feed{
  margin-top: 14px;
  display:grid;
  grid-template-columns: repeat(2, minmax(0,1fr));
  gap: 12px;
}
@media (max-width: 980px){
  .feed{ grid-template-columns: 1fr; }
}

.card{
  border: 1px solid var(--border-2);
  border-radius: 20px;
  padding: 14px 14px;
  background: var(--surface);
  box-shadow: var(--shadow-sm);
  position: relative;
  overflow: hidden;
}
.card:hover{ box-shadow: var(--shadow); }

.card::before{
  content:"";
  position:absolute;
  left:0; top:0; bottom:0;
  width: 6px;
  background: rgba(17,24,39,0.12);
}
.card.hot::before{ background: linear-gradient(180deg, var(--hf-orange), var(--hf-yellow)); }
.card.oss::before{ background: linear-gradient(180deg, #0b5fff, #7aa7ff); }

.topline{
  display:flex;
  align-items:flex-start;
  justify-content: space-between;
  gap: 10px;
}
.title{
  font-size: 1.03rem;
  font-weight: 950;
  letter-spacing: -0.01em;
  line-height: 1.25;
}
.badge{
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--border-2);
  background: rgba(255,255,255,0.88);
  font-weight: 950;
  color: var(--hf-black);
  white-space: nowrap;
}

.meta{
  margin-top: 10px;
  display:flex;
  flex-wrap:wrap;
  gap: 8px;
}
.pill{
  display:inline-flex;
  align-items:center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--border-2);
  background: rgba(255,255,255,0.80);
  font-size: 0.86rem;
  font-weight: 900;
}
.pill.dim{ color: var(--muted); font-weight: 950; }

.abs{
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px solid rgba(209,213,219,0.8);
}
.label{
  font-size: 0.78rem;
  color: var(--muted);
  font-weight: 950;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.txt{
  color: var(--text);
  font-size: 0.94rem;
  line-height: 1.42;
  font-weight: 550;
  white-space: pre-wrap;
}

details.more{
  margin-top: 10px;
  border: 1px dashed rgba(107,114,128,0.45);
  border-radius: 14px;
  padding: 10px 10px;
  background: rgba(255,255,255,0.70);
}
details.more summary{
  cursor:pointer;
  font-weight: 950;
  color: var(--hf-black);
}
.moreblock{ margin-top: 10px; }
.full{ white-space: pre-wrap; line-height: 1.50; }

.links{
  margin-top: 12px;
  display:flex;
  flex-wrap:wrap;
  gap: 10px;
  align-items:center;
}
.links a{
  color: var(--link);
  text-decoration:none;
  font-weight: 950;
}
.links a:hover{ text-decoration: underline; }

details.biblink{
  display:inline-block;
}
details.biblink summary{
  cursor:pointer;
  list-style:none;
  color: var(--link);
  font-weight: 950;
}
details.biblink summary::-webkit-details-marker{ display:none; }

.bibpop{
  margin-top: 10px;
  border: 1px solid var(--border-2);
  border-radius: 14px;
  background: #0b1020;
  color: #e5e7eb;
  padding: 10px 10px;
  box-shadow: var(--shadow);
  min-width: min(520px, 92vw);
}
.bibhint{
  font-size: 0.80rem;
  color: rgba(229,231,235,0.75);
  margin-bottom: 8px;
  font-weight: 700;
}
pre.bib{
  margin:0;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 0.84rem;
  line-height: 1.35;
}
"""

# =========================================================
# Data structures & caches
# =========================================================

@dataclass(frozen=True)
class ApiError:
    status_code: Optional[int]
    reason: str
    body_snippet: str

@dataclass(frozen=True)
class PaperExtras:
    project_page_url: Optional[str] = None
    github_repo_url: Optional[str] = None
    github_stars: Optional[int] = None

@dataclass(frozen=True)
class ArxivMeta:
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    year: str

HF_CACHE: Dict[Tuple[str, int], Tuple[float, List[Dict[str, Any]]]] = {}
ARXIV_CACHE: Dict[str, Tuple[float, ArxivMeta]] = {}
EXTRAS_CACHE: Dict[str, Tuple[float, PaperExtras]] = {}

# =========================================================
# Utilities
# =========================================================

def esc(s: Any) -> str:
    return html.escape(str(s), quote=True)

def normalize_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def utc_today_iso() -> str:
    return dt.datetime.utcnow().date().isoformat()

def parse_iso_date_or_today(s: str) -> str:
    s = (s or "").strip()
    if not RE_DATE_ISO.match(s):
        return utc_today_iso()
    try:
        dt.date.fromisoformat(s)
        return s
    except Exception:
        return utc_today_iso()

def is_arxiv_id(s: str) -> bool:
    s = (s or "").strip()
    return bool(RE_ARXIV_NEW.match(s) or RE_ARXIV_OLD.match(s))

def extract_arxiv_id(value: Any) -> str:
    """
    HF may store arXiv ids in various places. We try to extract a valid arXiv id from:
    - exact string
    - strings like "arxiv:2601.21821"
    - URLs containing the id
    """
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    s = s.replace("arXiv:", "arxiv:").replace("ARXIV:", "arxiv:")
    if s.startswith("arxiv:"):
        s = s.split(":", 1)[1].strip()
    # pull first likely arXiv identifier from anywhere in string
    m = re.search(r"(\d{4}\.\d{4,5}(v\d+)?)", s)
    if m:
        return m.group(1)
    m2 = re.search(r"([a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?)", s, re.IGNORECASE)
    if m2:
        return m2.group(1)
    return s if is_arxiv_id(s) else ""

def parse_year_from_iso(ts: Any) -> str:
    if ts is None:
        return ""
    s = str(ts).strip()
    if len(s) >= 4 and s[:4].isdigit():
        return s[:4]
    return ""

def percentile(values: List[int], p: float) -> int:
    if not values:
        return 0
    xs = sorted(values)
    k = int(round((len(xs) - 1) * p))
    k = max(0, min(len(xs) - 1, k))
    return xs[k]

def chunk(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]

def ensure_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    if u.startswith("//"):
        return "https:" + u
    if u.startswith("http://") or u.startswith("https://"):
        return u
    return "https://" + u

def parse_max_allowed_date_from_error(body_text: str) -> Optional[str]:
    body_text = body_text or ""
    m = RE_MAX_DATE.search(body_text)
    if not m:
        return None
    iso = m.group(1)
    # API returns "...Z"; keep date portion
    return iso[:10] if len(iso) >= 10 else None

# =========================================================
# HTTP helpers
# =========================================================

def safe_get_json(url: str, params: Optional[Dict[str, Any]]) -> Tuple[Optional[Any], Optional[ApiError], str]:
    last_text = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = SESSION.get(url, params=params, timeout=HTTP_TIMEOUT_S)
            last_text = r.text or ""
        except requests.RequestException:
            time.sleep(min(20.0, BACKOFF_BASE_S * (2 ** attempt)))
            continue

        if r.status_code == 200:
            try:
                return r.json(), None, ""
            except Exception:
                return None, ApiError(200, "Invalid JSON", (r.text or "")[:600]), (r.text or "")

        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(20.0, BACKOFF_BASE_S * (2 ** attempt)))
            continue

        return None, ApiError(r.status_code, r.reason or "HTTP error", (r.text or "")[:600]), (r.text or "")

    return None, ApiError(None, "Request failed", last_text[:600]), last_text

def safe_get_text(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Optional[str]:
    last_text = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = SESSION.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT_S)
            last_text = r.text or ""
            if r.status_code == 200:
                return last_text
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(20.0, BACKOFF_BASE_S * (2 ** attempt)))
                continue
        except requests.RequestException:
            time.sleep(min(15.0, BACKOFF_BASE_S * (2 ** attempt)))
            continue
        break
    return None

# =========================================================
# HF daily papers
# =========================================================

def fetch_hf_daily(date_iso: str, limit: int) -> Tuple[Optional[List[Dict[str, Any]]], Optional[ApiError], str]:
    date_iso = parse_iso_date_or_today(date_iso)
    limit = int(limit)

    now = time.time()
    key = (date_iso, limit)
    if key in HF_CACHE and (now - HF_CACHE[key][0] < HF_CACHE_TTL_S):
        return HF_CACHE[key][1], None, date_iso

    params = {"date": date_iso, "limit": limit, "sort": "trending"}
    data, err, raw = safe_get_json(HF_DAILY_PAPERS_API, params)

    if err is not None and err.status_code == 400:
        max_date = parse_max_allowed_date_from_error(raw)
        if max_date and max_date != date_iso:
            params2 = {"date": max_date, "limit": limit, "sort": "trending"}
            data2, err2, _ = safe_get_json(HF_DAILY_PAPERS_API, params2)
            if err2 is None and isinstance(data2, list):
                HF_CACHE[(max_date, limit)] = (now, data2)
                return data2, None, max_date
            return None, err2, max_date
        return None, err, date_iso

    if err is not None:
        return None, err, date_iso

    if not isinstance(data, list):
        return None, ApiError(200, "Unexpected payload", str(type(data))[:200]), date_iso

    HF_CACHE[key] = (now, data)
    return data, None, date_iso

# =========================================================
# Paper extras (Project page / GitHub)
# =========================================================

def extras_cache_get(pid: str) -> Optional[PaperExtras]:
    hit = EXTRAS_CACHE.get(pid)
    if not hit:
        return None
    ts, obj = hit
    return obj if (time.time() - ts < EXTRAS_CACHE_TTL_S) else None

def extras_cache_put(pid: str, ex: PaperExtras) -> None:
    EXTRAS_CACHE[pid] = (time.time(), ex)

def fetch_paper_extras(hf_id: str) -> PaperExtras:
    hf_id = (hf_id or "").strip()
    cached = extras_cache_get(hf_id)
    if cached:
        return cached

    project_page_url: Optional[str] = None
    github_repo_url: Optional[str] = None
    github_stars: Optional[int] = None

    obj, err, _ = safe_get_json(HF_PAPER_API.format(paper_id=hf_id), None)
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

    # best-effort HTML scrape (HF page changes sometimes)
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

    ex = PaperExtras(project_page_url=project_page_url, github_repo_url=github_repo_url, github_stars=github_stars)
    extras_cache_put(hf_id, ex)
    return ex

# =========================================================
# arXiv meta (abstract + optional enrichment)
# =========================================================

def arxiv_cache_get(aid: str) -> Optional[ArxivMeta]:
    hit = ARXIV_CACHE.get(aid)
    if not hit:
        return None
    ts, obj = hit
    return obj if (time.time() - ts < ARXIV_CACHE_TTL_S) else None

def arxiv_cache_put(meta: ArxivMeta) -> None:
    ARXIV_CACHE[meta.arxiv_id] = (time.time(), meta)

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

def fetch_arxiv_batch(arxiv_ids: List[str]) -> Dict[str, ArxivMeta]:
    result: Dict[str, ArxivMeta] = {}
    wanted: List[str] = []

    for aid in arxiv_ids:
        aid = (aid or "").strip()
        if not aid or not is_arxiv_id(aid):
            continue
        cached = arxiv_cache_get(aid)
        if cached:
            result[aid] = cached
        else:
            wanted.append(aid)

    if not wanted:
        return result

    # arXiv API etiquette: keep requests small and retryable
    headers = {"Accept": "application/atom+xml, text/xml, application/xml;q=0.9, */*;q=0.8"}
    for group in chunk(wanted, 20):
        xml_text = safe_get_text(ARXIV_API, params={"id_list": ",".join(group)}, headers=headers)
        if not xml_text:
            continue
        parsed = parse_arxiv_atom(xml_text)
        for k, v in parsed.items():
            result[k] = v
            arxiv_cache_put(v)

    return result

# =========================================================
# BibTeX (ALWAYS show when arXiv id exists)
# =========================================================

def bib_escape(s: str) -> str:
    s = (s or "").replace("{", "\\{").replace("}", "\\}")
    return s

def build_bibtex_fallback(arxiv_id: str, title: str, authors: List[str], year: str) -> str:
    """
    Create a robust BibTeX even if arXiv API is unavailable.
    """
    arxiv_id = (arxiv_id or "").strip()
    key = arxiv_id.replace("/", "_")
    au = " and ".join([a for a in (authors or []) if a.strip()]) or "Unknown"
    ti = bib_escape(title or "Untitled")
    yr = year or "????"
    url = ARXIV_ABS.format(paper_id=arxiv_id)

    return (
        f"@misc{{arxiv:{key},\n"
        f"  title={{ {ti} }},\n"
        f"  author={{ {au} }},\n"
        f"  year={{ {yr} }},\n"
        f"  eprint={{ {arxiv_id} }},\n"
        f"  archivePrefix={{arXiv}},\n"
        f"  url={{ {url} }}\n"
        f"}}"
    )

def build_bibtex(meta: Optional[ArxivMeta], arxiv_id: str, title: str, authors: List[str], year: str) -> str:
    if meta is not None:
        return build_bibtex_fallback(
            arxiv_id=meta.arxiv_id,
            title=meta.title or title,
            authors=meta.authors or authors,
            year=meta.year or year,
        )
    return build_bibtex_fallback(arxiv_id=arxiv_id, title=title, authors=authors, year=year)

# =========================================================
# Rendering helpers
# =========================================================

def get_hf_authors(paper: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    authors = paper.get("authors") or []
    if isinstance(authors, list):
        for a in authors:
            if isinstance(a, dict) and a.get("name"):
                out.append(str(a["name"]).strip())
            elif isinstance(a, str) and a.strip():
                out.append(a.strip())
    return out

def format_authors(names: List[str], max_names: int = 8) -> str:
    names = [n.strip() for n in (names or []) if n and n.strip()]
    if not names:
        return "—"
    if len(names) <= max_names:
        return ", ".join(names)
    return ", ".join(names[:max_names]) + f", et al. (+{len(names) - max_names})"

def preview_text(s: str, max_chars: int = 640) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return s if len(s) <= max_chars else s[: max_chars - 1].rstrip() + "…"

def build_highlight_card(title: str, items_: List[Dict[str, Any]]) -> str:
    if not items_:
        return f"<div class='hcard'><h3>{esc(title)}</h3><div style='color:var(--muted)'>—</div></div>"
    lines = []
    for r in items_:
        href = HF_PAPER_PAGE.format(paper_id=r["hf_id"])
        lines.append(
            "<div class='hitem'>"
            f"<a href='{esc(href)}' target='_blank' rel='noopener'>{esc(r['title'])}</a>"
            f"<div class='hval'>{esc(str(r['upvotes']))}</div>"
            "</div>"
        )
    return f"<div class='hcard'><h3>{esc(title)}</h3>{''.join(lines)}</div>"

# =========================================================
# Main render
# =========================================================

def render(date_iso: str, limit_str: str) -> Tuple[str, str]:
    date_iso = parse_iso_date_or_today(date_iso)
    try:
        limit = int(limit_str)
    except Exception:
        limit = 10

    items, err, effective = fetch_hf_daily(date_iso, limit)
    if err is not None or items is None:
        status = (
            "❌ Failed to fetch daily papers.\n\n"
            f"- Status: {err.status_code}\n"
            f"- Reason: {err.reason}\n"
            f"- Body (snippet): {err.body_snippet}"
        )
        return "", status

    rows: List[Dict[str, Any]] = []
    tag_counter: Dict[str, int] = {}

    # Normalize payload
    for it in items:
        if not isinstance(it, dict):
            continue
        paper = it.get("paper") if isinstance(it.get("paper"), dict) else it
        hf_id = str(paper.get("id") or it.get("id") or "").strip()

        title = str(paper.get("title") or it.get("title") or "").strip()
        title = title or "Untitled"

        # Upvotes: HF daily list surfaces a score-like field (varies)
        upvotes_raw = it.get("score")
        if upvotes_raw is None:
            upvotes_raw = it.get("upvotes")
        if upvotes_raw is None:
            upvotes_raw = paper.get("upvotes")
        try:
            upvotes = int(upvotes_raw) if upvotes_raw is not None else 0
        except Exception:
            upvotes = 0

        # Abstract candidate (HF)
        hf_abs = str(it.get("summary") or paper.get("summary") or paper.get("abstract") or "").strip()

        # Tags/topics
        tags = paper.get("tags") or it.get("tags") or []
        tag_list: List[str] = []
        if isinstance(tags, list):
            for t in tags:
                if isinstance(t, str) and t.strip():
                    tag_list.append(t.strip())
        for t in tag_list:
            tag_counter[t] = tag_counter.get(t, 0) + 1

        # arXiv id extraction
        arxiv_id = ""
        for cand in (
            paper.get("arxivId"),
            paper.get("arxiv_id"),
            paper.get("paperId"),
            paper.get("paper_id"),
            hf_id,
            paper.get("url"),
        ):
            arxiv_id = extract_arxiv_id(cand)
            if arxiv_id:
                break

        # Year guess from HF fields
        year = ""
        for cand in (paper.get("publishedAt"), paper.get("published_at"), it.get("publishedAt"), it.get("publishedAtDate")):
            year = parse_year_from_iso(cand)
            if year:
                break

        rows.append(
            {
                "hf_id": hf_id,
                "title": title,
                "upvotes": upvotes,
                "paper": paper,
                "item": it,
                "hf_abs": hf_abs,
                "tags": tag_list,
                "arxiv_id": arxiv_id,
                "year": year,
            }
        )

    # Fetch extras concurrently
    extras_map: Dict[str, PaperExtras] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {}
        for r in rows:
            if r["hf_id"]:
                futs[ex.submit(fetch_paper_extras, r["hf_id"])] = r["hf_id"]
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                extras_map[pid] = fut.result()
            except Exception:
                extras_map[pid] = PaperExtras()

    # Fetch arXiv meta (best-effort)
    arxiv_ids = sorted({r["arxiv_id"] for r in rows if r["arxiv_id"] and is_arxiv_id(r["arxiv_id"])})
    arxiv_map = fetch_arxiv_batch(arxiv_ids) if arxiv_ids else {}

    # KPIs
    total = len(rows)
    with_project = sum(1 for r in rows if (extras_map.get(r["hf_id"]) and extras_map[r["hf_id"]].project_page_url))
    with_github = sum(1 for r in rows if (extras_map.get(r["hf_id"]) and extras_map[r["hf_id"]].github_repo_url))

    upv_list = [r["upvotes"] for r in rows]
    hot_threshold = percentile(upv_list, 0.90) if upv_list else 0

    top_topics = sorted(tag_counter.items(), key=lambda kv: kv[1], reverse=True)[:6]
    topics_str = ", ".join([t for t, _ in top_topics]) if top_topics else "—"

    arxiv_cov = f"{len(arxiv_map)}/{len(arxiv_ids)}" if arxiv_ids else "0/0"

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
        f"<div class='kpi'><span>arXiv meta</span>{esc(arxiv_cov)}</div>"
        "</div>"
        "</div>"
    )

    # Highlights
    top_up = sorted(rows, key=lambda r: r["upvotes"], reverse=True)[:3]
    with_proj = []
    with_gh = []
    for r in sorted(rows, key=lambda r: r["upvotes"], reverse=True):
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
    for r in rows:
        hf_id = r["hf_id"]
        arxiv_id = r["arxiv_id"]
        title = r["title"]
        upvotes = r["upvotes"]
        paper = r["paper"]
        hf_abs = r["hf_abs"]
        year_guess = r["year"]

        exx = extras_map.get(hf_id)
        meta = arxiv_map.get(arxiv_id) if arxiv_id else None

        authors = meta.authors if (meta and meta.authors) else get_hf_authors(paper)
        authors_text = format_authors(authors)

        full_abs = meta.abstract if (meta and meta.abstract) else normalize_text(hf_abs)
        prev_abs = preview_text(full_abs, 640) if full_abs else ""

        pills = [
            "<span class='pill dim'>Authors</span>",
            f"<span class='pill'>{esc(authors_text)}</span>",
        ]

        # Show up to 3 tags as chips (glance value)
        tags = [t for t in (r.get("tags") or []) if t.strip()]
        for t in tags[:3]:
            pills.append(f"<span class='pill'>{esc(t)}</span>")

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
            links.append(f"<a href='{esc(ARXIV_ABS.format(paper_id=arxiv_id))}' target='_blank' rel='noopener'>arXiv</a>")
            links.append(f"<a href='{esc(ARXIV_PDF.format(paper_id=arxiv_id))}' target='_blank' rel='noopener'>PDF</a>")
        if exx and exx.project_page_url:
            links.append(f"<a href='{esc(exx.project_page_url)}' target='_blank' rel='noopener'>Project page</a>")
        if exx and exx.github_repo_url:
            links.append(f"<a href='{esc(exx.github_repo_url)}' target='_blank' rel='noopener'>GitHub</a>")

        # BibTeX next to GitHub (or at end if no GitHub)
        if arxiv_id:
            bib = build_bibtex(
                meta=meta,
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                year=(meta.year if meta else "") or year_guess,
            )
            bib_block = (
                "<details class='biblink'>"
                "<summary>BibTeX</summary>"
                "<div class='bibpop'>"
                "<div class='bibhint'>Select & copy (no JS).</div>"
                f"<pre class='bib'>{esc(bib)}</pre>"
                "</div>"
                "</details>"
            )
            # try to insert right after GitHub, else append
            inserted = False
            for i, a in enumerate(links):
                if ">GitHub<" in a:
                    links.insert(i + 1, bib_block)
                    inserted = True
                    break
            if not inserted:
                links.append(bib_block)

        details = (
            "<details class='more'>"
            "<summary>Details (full abstract)</summary>"
            "<div class='moreblock'>"
            f"<div class='full'>{esc(full_abs) if full_abs else esc('Abstract not available.')}</div>"
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

    status = f"✅ Loaded {len(rows)} paper(s) for {effective} (UTC).  arXiv meta: {arxiv_cov}."
    if effective != date_iso:
        status += f" (requested {date_iso} was not available yet; auto-clamped)"

    # add a visible status bar styling
    status = f"<div class='status'>{esc(status)}</div>"

    return page, status

# =========================================================
# Gradio UI (date / limit / fetch only)
# =========================================================

def build_app() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.HTML(
            f"<div class='hero'><h1>{esc(APP_TITLE)}</h1>"
            f"<div class='sub'>{esc(APP_SUBTITLE)}</div></div>"
        )

        with gr.Row(elem_id="toolbar"):
            date_in = gr.Textbox(
                label="",
                show_label=False,
                value=utc_today_iso(),
                max_lines=1,
                placeholder="Date (YYYY-MM-DD, UTC) e.g., 2026-01-30",
                scale=6,
                elem_id="date_in",
            )
            limit_in = gr.Dropdown(
                label="",
                show_label=False,
                choices=["5", "10", "20", "30", "50"],
                value="10",
                allow_custom_value=False,
                scale=2,
                elem_id="limit_in",
            )
            fetch_btn = gr.Button("Fetch", variant="primary", elem_id="fetch_btn", scale=2)

        html_out = gr.HTML()
        status_out = gr.HTML()

        fetch_btn.click(fn=render, inputs=[date_in, limit_in], outputs=[html_out, status_out])
        demo.load(fn=render, inputs=[date_in, limit_in], outputs=[html_out, status_out])

    return demo

demo = build_app()

if __name__ == "__main__":
    demo.queue().launch(css=CSS, theme=gr.themes.Soft(), ssr_mode=False)
