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
APP_SUBTITLE = "A glance-first briefing with full abstract + BibTeX."

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

HF_CACHE_TTL_S = int(os.getenv("HF_CACHE_TTL_S", "300"))              # 5 min
ARXIV_CACHE_TTL_S = int(os.getenv("ARXIV_CACHE_TTL_S", "86400"))      # 24 h
PAPER_EXTRAS_TTL_S = int(os.getenv("PAPER_EXTRAS_TTL_S", "86400"))    # 24 h

HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "qraphia/huggingface-daily-summary (briefing; gradio)",
        "Accept": "application/json",
    }
)
if HF_TOKEN:
    SESSION.headers["Authorization"] = f"Bearer {HF_TOKEN}"


HF_CACHE: Dict[Tuple[str, int], Tuple[float, List[Dict[str, Any]]]] = {}
ARXIV_CACHE: Dict[str, Tuple[float, "ArxivMeta"]] = {}
PAPER_EXTRAS_CACHE: Dict[str, Tuple[float, "PaperExtras"]] = {}

RE_ARXIV_NEW = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
RE_ARXIV_OLD = re.compile(r"^[a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$", re.IGNORECASE)
RE_MAX_DATE = re.compile(r'must be less than or equal to "([^"]+)"')

# Paper page HTML extraction (best-effort)
RE_PROJECT_PAGE = re.compile(r'href="([^"]+)"[^>]*>\s*Project page\s*<', re.IGNORECASE)
RE_GITHUB = re.compile(r'href="([^"]+github\.com[^"]+)"[^>]*>\s*GitHub\s*<', re.IGNORECASE)
RE_MODELS = re.compile(r"Models citing this paper\s+(\d+)", re.IGNORECASE)
RE_DATASETS = re.compile(r"Datasets citing this paper\s+(\d+)", re.IGNORECASE)
RE_SPACES = re.compile(r"Spaces citing this paper\s+(\d+)", re.IGNORECASE)

# HF brand colors (official): #FFD21E / #FF9D00 / #6B7280
CSS = """
:root{
  --hf-yellow:#FFD21E;
  --hf-orange:#FF9D00;
  --hf-gray:#6B7280;

  --bg:#ffffff;
  --surface:#ffffff;
  --muted-bg:#f6f7f9;
  --border:#e6e8ee;
  --text:#111827;
  --muted:#6B7280;

  --shadow: 0 8px 28px rgba(17,24,39,0.08);
  --shadow-sm: 0 3px 10px rgba(17,24,39,0.06);
}

body{
  background: radial-gradient(900px 340px at 20% 0%, rgba(255,157,0,0.18), transparent 55%),
              radial-gradient(800px 320px at 80% 0%, rgba(255,210,30,0.18), transparent 55%),
              var(--bg);
  color: var(--text);
}

.gradio-container{ max-width: 1120px !important; }
footer{ display:none !important; }

/* Header */
.hero{
  margin: 14px 0 12px;
  padding: 16px 16px;
  border: 1px solid var(--border);
  border-radius: 16px;
  background: linear-gradient(180deg, #fff, #fff8e0);
  box-shadow: var(--shadow-sm);
}
.hero h1{
  margin:0;
  font-size: 1.55rem;
  letter-spacing:-0.02em;
  color: var(--text);
}
.hero .sub{
  margin-top: 6px;
  color: var(--muted);
  font-size: 0.98rem;
  line-height: 1.35;
}

/* Toolbar (sticky, compact, stylish) */
#toolbar{
  position: sticky;
  top: 10px;
  z-index: 30;
  margin: 10px 0 10px;
  padding: 12px 12px;
  border: 1px solid var(--border);
  border-radius: 16px;
  background: rgba(255,255,255,0.94);
  backdrop-filter: blur(10px);
  box-shadow: var(--shadow);
}

#toolbar .gr-form{ background: transparent !important; border: none !important; box-shadow:none !important; }

#toolbar input, #toolbar select{
  border-radius: 12px !important;
  border: 1px solid var(--border) !important;
  background: #fff !important;
}
#toolbar input:focus, #toolbar select:focus{
  border-color: rgba(255,157,0,0.55) !important;
  box-shadow: 0 0 0 4px rgba(255,157,0,0.18) !important;
}

#fetch_btn{
  border-radius: 12px !important;
  border: 1px solid rgba(255,157,0,0.55) !important;
  background: linear-gradient(180deg, var(--hf-orange), #ffb84a) !important;
  color: #111827 !important;
  font-weight: 900 !important;
  box-shadow: 0 10px 18px rgba(255,157,0,0.22) !important;
}
#fetch_btn:hover{
  transform: translateY(-1px);
  box-shadow: 0 14px 24px rgba(255,157,0,0.26) !important;
}

/* KPI row */
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
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--muted-bg);
  color: var(--text);
  font-size: 0.90rem;
  font-weight: 900;
}
.kpi span{ color: var(--muted); font-weight: 900; }

/* Highlights */
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
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 12px 12px;
  background: var(--surface);
  box-shadow: var(--shadow-sm);
}
.hcard h3{
  margin:0 0 8px;
  font-size: 0.95rem;
  letter-spacing: -0.01em;
  color: var(--text);
}
.hitem{
  display:flex;
  justify-content: space-between;
  gap: 10px;
  padding: 8px 0;
  border-top: 1px solid var(--border);
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
  font-weight: 900;
  white-space: nowrap;
}

/* Feed cards */
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
  border: 1px solid var(--border);
  border-radius: 18px;
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
  background: var(--hf-orange);
}
.card.hot::before{ background: #ff3b30; }
.card.oss::before{ background: #22c55e; }

.topline{
  display:flex;
  justify-content: space-between;
  gap: 10px;
  align-items:flex-start;
}
.title{
  margin:0;
  color: var(--text);
  font-weight: 950;
  font-size: 1.02rem;
  line-height: 1.28;
  display:-webkit-box;
  -webkit-line-clamp:2;
  -webkit-box-orient:vertical;
  overflow:hidden;
}
.badge{
  padding: 6px 10px;
  border-radius: 999px;
  background: #fff7e6;
  border: 1px solid rgba(255,157,0,0.35);
  color: #7a3d00;
  font-weight: 950;
  font-size: 0.86rem;
  white-space: nowrap;
}

.meta{
  margin-top: 10px;
  display:flex;
  flex-wrap:wrap;
  gap: 8px;
}
.pill{
  padding: 4px 10px;
  border-radius: 999px;
  background: var(--muted-bg);
  border: 1px solid var(--border);
  color: var(--text);
  font-weight: 900;
  font-size: 0.84rem;
}
.pill.dim{ color: var(--muted); }

.abs{
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--border);
}
.abs .label{
  color: var(--muted);
  font-weight: 950;
  font-size: 0.84rem;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}
.abs .txt{
  margin-top: 8px;
  color: var(--text);
  font-size: 0.95rem;
  line-height: 1.55;
  display:-webkit-box;
  -webkit-line-clamp:4;
  -webkit-box-orient:vertical;
  overflow:hidden;
}

details.more summary{
  cursor: pointer;
  margin-top: 12px;
  color: #0b5fff;
  font-weight: 950;
  font-size: 0.90rem;
}
.moreblock{
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid var(--border);
}
.moreblock .full{
  white-space: pre-wrap;
  color: var(--text);
  line-height: 1.55;
  font-size: 0.95rem;
}
pre.bib{
  margin-top: 10px;
  padding: 10px;
  border-radius: 14px;
  background: #0b1220;
  border: 1px solid rgba(17,24,39,0.20);
  color: #e5e7eb;
  overflow-x: auto;
  font-size: 0.86rem;
  line-height: 1.45;
}

.links{
  margin-top: 12px;
  display:flex;
  flex-wrap:wrap;
  gap: 10px;
}
.links a{
  color: #0b5fff;
  text-decoration:none;
  font-weight: 950;
  font-size: 0.92rem;
}
.links a:hover{ text-decoration: underline; }

.error{
  border: 1px solid rgba(255,59,48,0.35);
  background: rgba(255,59,48,0.08);
  color: #7a0b0b;
  padding: 12px;
  border-radius: 16px;
}
"""


# =========================
# Data models
# =========================

@dataclass
class ApiError:
    status_code: Optional[int]
    reason: str
    body_snippet: str

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
    return iso_dt.split("T", 1)[0] if "T" in iso_dt else (iso_dt[:10] if len(iso_dt) >= 10 else None)

def safe_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[ApiError], str]:
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

def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


# =========================
# Fetch daily list (auto clamp future date)
# =========================

def fetch_hf_daily(date_iso: str, limit: int) -> Tuple[Optional[List[Dict[str, Any]]], Optional[ApiError], str]:
    date_iso = (date_iso or "").strip()
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

    if err is not None:
        return None, err, date_iso

    if not isinstance(data, list):
        return None, ApiError(200, "Unexpected payload", str(type(data))[:200]), date_iso

    HF_CACHE[key] = (now, data)
    return data, None, date_iso


# =========================
# Paper extras (Project/GitHub/Hub assets counts)
# =========================

def cache_get_extras(pid: str) -> Optional[PaperExtras]:
    hit = PAPER_EXTRAS_CACHE.get(pid)
    if not hit:
        return None
    ts, obj = hit
    return obj if (time.time() - ts < PAPER_EXTRAS_TTL_S) else None

def cache_put_extras(pid: str, ex: PaperExtras) -> None:
    PAPER_EXTRAS_CACHE[pid] = (time.time(), ex)

def extract_counts(text: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
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
    cached = cache_get_extras(pid)
    if cached:
        return cached

    project_page_url: Optional[str] = None
    github_repo_url: Optional[str] = None
    github_stars: Optional[int] = None
    cm = cd = cs = None

    obj, err, _ = safe_get_json(HF_PAPER_API.format(paper_id=pid), None)
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

    page = safe_get_text(HF_PAPER_PAGE.format(paper_id=pid))
    if page:
        if project_page_url is None:
            m = RE_PROJECT_PAGE.search(page)
            if m:
                project_page_url = ensure_url(m.group(1))
        if github_repo_url is None:
            m = RE_GITHUB.search(page)
            if m:
                github_repo_url = ensure_url(m.group(1))
        cm, cd, cs = extract_counts(page)

    ex = PaperExtras(
        project_page_url=project_page_url,
        github_repo_url=github_repo_url,
        github_stars=github_stars,
        citing_models=cm,
        citing_datasets=cd,
        citing_spaces=cs,
    )
    cache_put_extras(pid, ex)
    return ex


# =========================
# arXiv meta (abstract + BibTeX)
# =========================

def cache_get_arxiv(aid: str) -> Optional[ArxivMeta]:
    hit = ARXIV_CACHE.get(aid)
    if not hit:
        return None
    ts, obj = hit
    return obj if (time.time() - ts < ARXIV_CACHE_TTL_S) else None

def cache_put_arxiv(meta: ArxivMeta) -> None:
    ARXIV_CACHE[meta.arxiv_id] = (time.time(), meta)

def chunk(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]

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
            out[arxiv_id] = ArxivMeta(arxiv_id=arxiv_id, title=title, abstract=abstract, authors=authors, year=year)
    return out

def fetch_arxiv_batch(arxiv_ids: List[str]) -> Dict[str, ArxivMeta]:
    result: Dict[str, ArxivMeta] = {}
    wanted: List[str] = []
    for aid in arxiv_ids:
        if not aid or not is_arxiv_id(aid):
            continue
        cached = cache_get_arxiv(aid)
        if cached:
            result[aid] = cached
        else:
            wanted.append(aid)

    for group in chunk(wanted, 25):
        try:
            r = SESSION.get(ARXIV_API, params={"id_list": ",".join(group)}, timeout=HTTP_TIMEOUT_S)
            if r.status_code == 200:
                parsed = parse_arxiv_atom(r.text or "")
                for k, v in parsed.items():
                    result[k] = v
                    cache_put_arxiv(v)
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

def preview_text(s: str, max_chars: int = 620) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return s if len(s) <= max_chars else s[: max_chars - 1].rstrip() + "…"

def build_highlight_card(title: str, items_: List[Dict[str, Any]], metric: str) -> str:
    if not items_:
        return f"<div class='hcard'><h3>{esc(title)}</h3><div class='sub'>—</div></div>"
    lines = []
    for r in items_:
        pid = r["pid"]
        href = HF_PAPER_PAGE.format(paper_id=pid) if pid else "#"
        if metric == "upvotes":
            val = str(r["upvotes"])
        elif metric == "stars":
            val = f"★ {r['stars']}"
        else:
            val = str(r["assets"])
        lines.append(
            "<div class='hitem'>"
            f"<a href='{esc(href)}' target='_blank' rel='noopener'>{esc(r['title'])}</a>"
            f"<div class='hval'>{esc(val)}</div>"
            "</div>"
        )
    return f"<div class='hcard'><h3>{esc(title)}</h3>{''.join(lines)}</div>"

def render(date_iso: str, limit_choice: str) -> Tuple[str, str]:
    date_iso = (date_iso or "").strip()
    try:
        dt.date.fromisoformat(date_iso)
    except Exception:
        return "", "❌ Invalid date. Use YYYY-MM-DD (UTC)."

    limit = safe_int(limit_choice, 10)
    limit = max(1, min(50, limit))

    items, err, effective = fetch_hf_daily(date_iso, limit)
    if err is not None:
        return (
            f"<div class='error'><b>API request failed</b><br/>"
            f"Status: {esc(str(err.status_code))}<br/>"
            f"Reason: {esc(err.reason)}<br/>"
            f"Body (snippet): {esc(err.body_snippet)}</div>",
            "❌ Failed to fetch daily papers."
        )

    assert items is not None
    if not items:
        return "<div class='feed'></div>", "No results."

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
    arxiv_ids = [pid for pid in paper_ids if is_arxiv_id(pid)]
    arxiv_map = fetch_arxiv_batch(arxiv_ids)

    # derived metrics
    rows: List[Dict[str, Any]] = []
    upvotes_all: List[int] = []
    topic_counter: Dict[str, int] = {}

    for it in items:
        paper = (it or {}).get("paper") or {}
        pid = str(paper.get("id") or it.get("id") or "").strip()
        title = str(it.get("title") or paper.get("title") or "").strip() or "(no title)"
        up = safe_int(paper.get("upvotes"), 0)
        upvotes_all.append(up)

        exx = extras_map.get(pid)
        stars = exx.github_stars if (exx and exx.github_stars is not None) else None
        assets = 0
        if exx:
            assets = safe_int(exx.citing_models, 0) + safe_int(exx.citing_datasets, 0) + safe_int(exx.citing_spaces, 0)

        kws = paper.get("ai_keywords") or []
        if isinstance(kws, list):
            for k in kws[:10]:
                ks = str(k).strip()
                if ks:
                    topic_counter[ks] = topic_counter.get(ks, 0) + 1

        rows.append(
            {
                "pid": pid,
                "title": title,
                "upvotes": up,
                "stars": stars,
                "assets": assets,
                "has_project": bool(exx and exx.project_page_url),
                "has_github": bool(exx and exx.github_repo_url),
            }
        )

    # hot threshold (80th percentile)
    up_sorted = sorted(upvotes_all)
    hot_threshold = up_sorted[max(0, int(0.80 * (len(up_sorted) - 1)))] if up_sorted else 0

    total = len(items)
    with_project = sum(1 for r in rows if r["has_project"])
    with_github = sum(1 for r in rows if r["has_github"])
    with_assets = sum(1 for r in rows if r["assets"] > 0)
    top_topics = sorted(topic_counter.items(), key=lambda kv: kv[1], reverse=True)[:6]
    topics_str = ", ".join([t for t, _ in top_topics]) if top_topics else "—"

    # highlights (avoid duplicates across sections)
    used: set = set()

    top_up = []
    for r in sorted(rows, key=lambda r: r["upvotes"], reverse=True):
        if r["pid"] not in used:
            top_up.append(r)
            used.add(r["pid"])
        if len(top_up) == 3:
            break

    top_star = []
    for r in sorted(rows, key=lambda r: (r["stars"] if r["stars"] is not None else -1), reverse=True):
        if r["stars"] is None:
            continue
        if r["pid"] not in used:
            top_star.append(r)
            used.add(r["pid"])
        if len(top_star) == 3:
            break

    top_assets = []
    for r in sorted(rows, key=lambda r: r["assets"], reverse=True):
        if r["assets"] <= 0:
            continue
        if r["pid"] not in used:
            top_assets.append(r)
            used.add(r["pid"])
        if len(top_assets) == 3:
            break

    hero = (
        "<div class='hero'>"
        f"<h1>{esc(APP_TITLE)} — {esc(effective)} (UTC)</h1>"
        f"<div class='sub'>{esc(APP_SUBTITLE)}</div>"
        "<div class='kpis'>"
        f"<div class='kpi'><span>Total</span>{total}</div>"
        f"<div class='kpi'><span>Project page</span>{with_project}</div>"
        f"<div class='kpi'><span>GitHub</span>{with_github}</div>"
        f"<div class='kpi'><span>Hub assets</span>{with_assets}</div>"
        f"<div class='kpi'><span>HOT ≥</span>{hot_threshold}</div>"
        f"<div class='kpi'><span>Top topics</span>{esc(topics_str)}</div>"
        "</div>"
        "</div>"
    )

    highlights = (
        "<div class='highlights'>"
        + build_highlight_card("Top Upvotes", top_up, "upvotes")
        + build_highlight_card("Top GitHub Stars", top_star, "stars")
        + build_highlight_card("Most Hub Assets", top_assets, "assets")
        + "</div>"
    )

    cards: List[str] = ["<div class='feed'>"]
    for it in items:
        paper = (it or {}).get("paper") or {}
        pid = str(paper.get("id") or it.get("id") or "").strip()

        title = str(it.get("title") or paper.get("title") or "").strip() or "(no title)"
        upvotes = safe_int(paper.get("upvotes"), 0)

        meta = arxiv_map.get(pid)
        authors = meta.authors if (meta and meta.authors) else get_hf_authors(paper)
        authors_text = format_authors(authors)

        exx = extras_map.get(pid)

        full_abs = meta.abstract if (meta and meta.abstract) else normalize_text(str(it.get("summary") or paper.get("summary") or ""))
        prev_abs = preview_text(full_abs, 620) if full_abs else ""

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

        if exx and (exx.citing_models is not None or exx.citing_datasets is not None or exx.citing_spaces is not None):
            cm = "—" if exx.citing_models is None else str(exx.citing_models)
            cd = "—" if exx.citing_datasets is None else str(exx.citing_datasets)
            cs = "—" if exx.citing_spaces is None else str(exx.citing_spaces)
            pills.append(f"<span class='pill'>Models {esc(cm)} · Datasets {esc(cd)} · Spaces {esc(cs)}</span>")

        cls = "card"
        if hot_threshold > 0 and upvotes >= hot_threshold:
            cls += " hot"
        if exx and (exx.project_page_url or exx.github_repo_url):
            cls += " oss"

        links = []
        if pid:
            links.append(f"<a href='{esc(HF_PAPER_PAGE.format(paper_id=pid))}' target='_blank' rel='noopener'>HF page</a>")
        if pid and is_arxiv_id(pid):
            links.append(f"<a href='{esc(ARXIV_ABS.format(paper_id=pid))}' target='_blank' rel='noopener'>arXiv</a>")
            links.append(f"<a href='{esc(ARXIV_PDF.format(paper_id=pid))}' target='_blank' rel='noopener'>PDF</a>")
        if exx and exx.project_page_url:
            links.append(f"<a href='{esc(exx.project_page_url)}' target='_blank' rel='noopener'>Project page</a>")
        if exx and exx.github_repo_url:
            links.append(f"<a href='{esc(exx.github_repo_url)}' target='_blank' rel='noopener'>GitHub</a>")

        bib = build_bibtex(meta) if meta else ""
        details = (
            "<details class='more'>"
            "<summary>Details (full abstract + BibTeX)</summary>"
            "<div class='moreblock'>"
            f"<div class='full'>{esc(full_abs) if full_abs else esc('Abstract not available.')}</div>"
            + (f"<pre class='bib'>{esc(bib)}</pre>" if bib else "")
            + "</div></details>"
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

    status = f"✅ Loaded {len(items)} paper(s) for {effective} (UTC)."
    if effective != date_iso:
        status += f" (requested {date_iso} was not available yet; auto-clamped)"
    return page, status


# =========================
# Gradio UI (date / limit / fetch only)
# =========================

def build_app() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.HTML(
            f"<div class='hero'><h1>{esc(APP_TITLE)}</h1>"
            f"<div class='sub'>{esc(APP_SUBTITLE)}</div></div>"
        )

        with gr.Row(elem_id="toolbar"):
            date_in = gr.Textbox(
                label="Date (YYYY-MM-DD, UTC)",
                value=utc_today_iso(),
                max_lines=1,
                placeholder="e.g., 2026-01-30",
            )
            limit_in = gr.Dropdown(
                label="Limit",
                choices=["5", "10", "20", "30", "50"],
                value="10",
                allow_custom_value=False,
            )
            fetch_btn = gr.Button("Fetch", variant="primary", elem_id="fetch_btn")

        status_out = gr.Markdown()
        html_out = gr.HTML()

        fetch_btn.click(fn=render, inputs=[date_in, limit_in], outputs=[html_out, status_out])
        demo.load(fn=render, inputs=[date_in, limit_in], outputs=[html_out, status_out])

    return demo


demo = build_app()

if __name__ == "__main__":
    demo.queue().launch(css=CSS, theme=gr.themes.Soft(), ssr_mode=False)
