import datetime as dt
import html
import json
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


# =========================
# Configuration
# =========================

APP_TITLE = "Hugging Face Daily Summary"
APP_SUBTITLE = (
    "Triage-friendly viewer for Hugging Face Daily Papers. "
    "No translation. No keypoints. Abstract-first. Project page links when available."
)

HF_DAILY_PAPERS_API = "https://huggingface.co/api/daily_papers"
HF_PAPER_API = "https://huggingface.co/api/papers/{paper_id}"
HF_PAPER_PAGE = "https://huggingface.co/papers/{paper_id}"

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_ABS = "https://arxiv.org/abs/{paper_id}"
ARXIV_PDF = "https://arxiv.org/pdf/{paper_id}.pdf"

HTTP_TIMEOUT_S = int(os.getenv("HTTP_TIMEOUT_S", "25"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
BACKOFF_BASE_S = float(os.getenv("BACKOFF_BASE_S", "1.0"))

HF_CACHE_TTL_S = int(os.getenv("HF_CACHE_TTL_S", "300"))              # 5 min
PAPER_EXTRAS_TTL_S = int(os.getenv("PAPER_EXTRAS_TTL_S", "86400"))    # 24 h
ARXIV_CACHE_TTL_S = int(os.getenv("ARXIV_CACHE_TTL_S", "86400"))      # 24 h

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "6"))

# Optional: set as Space secret to reduce HF rate limiting.
HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "qraphia/huggingface-daily-summary (clean; triage)",
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

# HTML extraction: Project page anchor
RE_PROJECT_PAGE = re.compile(r'href="([^"]+)"[^>]*>\s*Project page\s*<', re.IGNORECASE)
# HTML extraction: citing counts
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
  --shadow: 0 1px 1px rgba(17,24,39,0.04);
}

body{ background: var(--bg); }
.gradio-container{ max-width: 1080px !important; }

.header{ margin: 6px 0 10px; }
.header h1{ margin: 0 0 0.25rem; font-size: 1.65rem; letter-spacing: -0.01em; }
.subtle{ color: var(--muted); font-size: 0.96rem; line-height: 1.45; }

.panel{
  background: var(--card);
  border: 1px solid var(--bd);
  border-radius: 16px;
  padding: 12px 12px;
  box-shadow: var(--shadow);
  margin-bottom: 10px;
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

.title{
  font-size: 1.08rem;
  font-weight: 800;
  color: var(--text);
  line-height: 1.35;
  margin: 0;
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
  gap: 6px;
  padding: 2px 10px;
  background: var(--chip);
  border: 1px solid var(--bd);
  border-radius: 999px;
  color: #374151;
  font-size: 0.86rem;
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
  font-weight: 700;
}

.links a:hover{ text-decoration: underline; }

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

details.abstract-full summary{
  cursor: pointer;
  color: #374151;
  font-weight: 800;
  font-size: 0.92rem;
  margin-top: 8px;
}

.abstract-full-text{
  margin-top: 8px;
  color: var(--text);
  font-size: 0.97rem;
  line-height: 1.62;
  white-space: pre-wrap;
}

.chips{
  margin-top: 10px;
  display:flex;
  flex-wrap:wrap;
  gap: 6px;
}

.chip{
  background: var(--chip);
  border: 1px solid var(--bd);
  border-radius:999px;
  padding:2px 10px;
  font-size:0.82rem;
  color:#374151;
}

.error-box{
  background: #fff1f2;
  border: 1px solid #fecdd3;
  border-radius: 16px;
  padding: 12px 12px;
  color: #7f1d1d;
}

.diag{
  margin-top: 10px;
  background: #f9fafb;
  border: 1px solid var(--bd);
  border-radius: 16px;
  padding: 10px 10px;
  color: #374151;
  font-size: 0.88rem;
  white-space: pre-wrap;
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
    published_date: Optional[str]      # YYYY-MM-DD
    submitted_on_daily: Optional[str]  # YYYY-MM-DD
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
    # domain like mmfinereason.github.io
    return "https://" + u

def extract_rate_limit_headers(headers: Dict[str, str]) -> Dict[str, str]:
    keep: Dict[str, str] = {}
    for k, v in (headers or {}).items():
        lk = k.lower()
        if lk.startswith("ratelimit-") or lk in ("retry-after",):
            keep[k] = v
    return keep

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

def safe_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[ApiError], Dict[str, str], str]:
    last_headers: Dict[str, str] = {}
    last_text: str = ""
    last_err: Optional[ApiError] = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            r = SESSION.get(url, params=params, timeout=HTTP_TIMEOUT_S)
            last_headers = dict(r.headers or {})
            last_text = r.text or ""
        except requests.RequestException as e:
            last_err = ApiError(None, type(e).__name__, str(e)[:600], {})
            time.sleep(min(20.0, BACKOFF_BASE_S * (2 ** attempt)))
            continue

        if r.status_code == 200:
            try:
                return r.json(), None, extract_rate_limit_headers(last_headers), ""
            except Exception:
                err = ApiError(200, "Invalid JSON", (r.text or "")[:600], extract_rate_limit_headers(last_headers))
                return None, err, extract_rate_limit_headers(last_headers), (r.text or "")

        if r.status_code in (429, 500, 502, 503, 504):
            retry_after = r.headers.get("Retry-After")
            wait_s = min(20.0, BACKOFF_BASE_S * (2 ** attempt))
            if retry_after:
                try:
                    wait_s = max(wait_s, float(retry_after))
                except Exception:
                    pass
            last_err = ApiError(r.status_code, r.reason or "HTTP error", (r.text or "")[:600], extract_rate_limit_headers(last_headers))
            time.sleep(wait_s)
            continue

        err = ApiError(r.status_code, r.reason or "HTTP error", (r.text or "")[:600], extract_rate_limit_headers(last_headers))
        return None, err, extract_rate_limit_headers(last_headers), (r.text or "")

    return None, last_err, extract_rate_limit_headers(last_headers), last_text

def safe_get_text(url: str) -> Tuple[Optional[str], Optional[ApiError]]:
    last_err: Optional[ApiError] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = SESSION.get(url, timeout=HTTP_TIMEOUT_S)
        except requests.RequestException as e:
            last_err = ApiError(None, type(e).__name__, str(e)[:600], {})
            time.sleep(min(15.0, BACKOFF_BASE_S * (2 ** attempt)))
            continue

        if r.status_code == 200:
            return (r.text or ""), None

        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(15.0, BACKOFF_BASE_S * (2 ** attempt)))
            continue

        return None, ApiError(r.status_code, r.reason or "HTTP error", (r.text or "")[:600], {})

    return None, last_err


# =========================
# HF Daily Papers
# =========================

def fetch_hf_daily(date_iso: str, limit: int) -> Tuple[Optional[List[Dict[str, Any]]], Optional[ApiError], Dict[str, str], bool, str]:
    requested = (date_iso or "").strip()
    effective = requested

    key = (effective, int(limit))
    now = time.time()
    hit = HF_CACHE.get(key)
    if hit and (now - hit[0] < HF_CACHE_TTL_S):
        return hit[1], None, hit[2], True, effective

    params = {"date": effective, "limit": int(limit), "sort": "trending"}
    data, err, rate_headers, raw_text = safe_get_json(HF_DAILY_PAPERS_API, params)

    if err is not None and err.status_code == 400:
        max_date = parse_max_allowed_date_from_error(raw_text)
        if max_date and max_date != effective:
            effective = max_date
            key2 = (effective, int(limit))
            hit2 = HF_CACHE.get(key2)
            if hit2 and (now - hit2[0] < HF_CACHE_TTL_S):
                return hit2[1], None, hit2[2], True, effective

            params2 = {"date": effective, "limit": int(limit), "sort": "trending"}
            data2, err2, rate_headers2, _ = safe_get_json(HF_DAILY_PAPERS_API, params2)
            if err2 is None and isinstance(data2, list):
                HF_CACHE[key2] = (now, data2, rate_headers2)
                return data2, None, rate_headers2, False, effective

            return None, err2, rate_headers2, False, effective

    if err is not None:
        return None, err, rate_headers, False, effective

    if not isinstance(data, list):
        return None, ApiError(200, "Unexpected payload", str(type(data))[:200], rate_headers), rate_headers, False, effective

    HF_CACHE[key] = (now, data, rate_headers)
    return data, None, rate_headers, False, effective


# =========================
# HF Paper extras (project page, github, counts)
# =========================

def _cache_get_extras(pid: str) -> Optional[PaperExtras]:
    now = time.time()
    hit = PAPER_EXTRAS_CACHE.get(pid)
    if hit and (now - hit[0] < PAPER_EXTRAS_TTL_S):
        return hit[1]
    return None

def _cache_put_extras(pid: str, extras: PaperExtras) -> None:
    PAPER_EXTRAS_CACHE[pid] = (time.time(), extras)

def _date_only(iso_dt: Any) -> Optional[str]:
    s = str(iso_dt or "").strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return None

def _extract_project_from_api(obj: Dict[str, Any]) -> Optional[str]:
    # key candidates (future-proof)
    candidates = [
        "projectPage",
        "projectPageUrl",
        "projectPageURL",
        "project_url",
        "projectUrl",
        "project_page",
        "projectPageURL",
        "project_page_url",
        "projectPageURL",
        "projectHomepage",
        "projectHomepageUrl",
    ]
    for k in candidates:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return ensure_url(v.strip())

    # sometimes embedded under "links" or similar
    for k in ("links", "resources", "artifacts"):
        v = obj.get(k)
        if isinstance(v, list):
            for it in v:
                if isinstance(it, dict):
                    name = str(it.get("name") or it.get("label") or "").strip().lower()
                    url = it.get("url") or it.get("href")
                    if isinstance(url, str) and url.strip():
                        u = ensure_url(url.strip())
                        if "project" in name:
                            return u
        if isinstance(v, dict):
            # try common shapes
            for kk, vv in v.items():
                if isinstance(vv, str) and vv.strip():
                    if "project" in str(kk).lower():
                        return ensure_url(vv.strip())

    return None

def _extract_counts_from_html(text: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    def g(rx: re.Pattern) -> Optional[int]:
        m = rx.search(text or "")
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None
    return g(RE_MODELS), g(RE_DATASETS), g(RE_SPACES)

def _extract_project_from_html(text: str) -> Optional[str]:
    m = RE_PROJECT_PAGE.search(text or "")
    if not m:
        return None
    return ensure_url(m.group(1))

def fetch_paper_extras(paper_id: str) -> PaperExtras:
    pid = (paper_id or "").strip()
    cached = _cache_get_extras(pid)
    if cached:
        return cached

    project_page: Optional[str] = None
    github_repo: Optional[str] = None
    github_stars: Optional[int] = None
    published_date: Optional[str] = None
    submitted_on_daily: Optional[str] = None
    citing_models: Optional[int] = None
    citing_datasets: Optional[int] = None
    citing_spaces: Optional[int] = None

    # 1) Try API first
    api_url = HF_PAPER_API.format(paper_id=pid)
    obj, err, _, _ = safe_get_json(api_url, None)
    if err is None and isinstance(obj, dict) and "error" not in obj:
        project_page = _extract_project_from_api(obj)

        gr_url = obj.get("githubRepo")
        if isinstance(gr_url, str) and gr_url.strip().startswith("http") and "github.com" in gr_url:
            github_repo = gr_url.strip()

        gs = obj.get("githubStars")
        try:
            if gs is not None:
                github_stars = int(gs)
        except Exception:
            github_stars = None

        published_date = _date_only(obj.get("publishedAt"))
        submitted_on_daily = _date_only(obj.get("submittedOnDailyAt"))

    # 2) If missing pieces, scrape paper page HTML (best-effort)
    if project_page is None or citing_models is None or citing_datasets is None or citing_spaces is None:
        page_url = HF_PAPER_PAGE.format(paper_id=pid)
        page_text, page_err = safe_get_text(page_url)
        if page_err is None and page_text:
            if project_page is None:
                project_page = _extract_project_from_html(page_text)
            cm, cd, cs = _extract_counts_from_html(page_text)
            if citing_models is None:
                citing_models = cm
            if citing_datasets is None:
                citing_datasets = cd
            if citing_spaces is None:
                citing_spaces = cs

    extras = PaperExtras(
        project_page_url=project_page,
        github_repo_url=github_repo,
        github_stars=github_stars,
        published_date=published_date,
        submitted_on_daily=submitted_on_daily,
        citing_models=citing_models,
        citing_datasets=citing_datasets,
        citing_spaces=citing_spaces,
    )
    _cache_put_extras(pid, extras)
    return extras


# =========================
# arXiv batch fetch (abstract)
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
        params = {"id_list": ",".join(group)}
        for attempt in range(MAX_RETRIES + 1):
            try:
                r = SESSION.get(ARXIV_API, params=params, timeout=HTTP_TIMEOUT_S)
                if r.status_code == 200:
                    parsed = _parse_arxiv_atom(r.text or "")
                    for k, meta in parsed.items():
                        result[k] = meta
                        _cache_put_arxiv(meta)
                    break
            except requests.RequestException:
                pass
            time.sleep(min(10.0, BACKOFF_BASE_S * (2 ** attempt)))

    return result


# =========================
# Rendering helpers
# =========================

def format_authors(names: List[str], max_names: int = 8) -> str:
    names = [n.strip() for n in (names or []) if n and n.strip()]
    if not names:
        return "—"
    if len(names) <= max_names:
        return ", ".join(names)
    return ", ".join(names[:max_names]) + f", et al. (+{len(names) - max_names})"

def get_hf_authors(paper: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    authors = paper.get("authors") or []
    if isinstance(authors, list):
        for a in authors:
            if isinstance(a, dict) and a.get("name"):
                out.append(str(a["name"]).strip())
    return out

def get_keywords(paper: Dict[str, Any]) -> List[str]:
    kws = paper.get("ai_keywords") or []
    if isinstance(kws, list):
        return [str(k).strip() for k in kws if str(k).strip()]
    return []

def abstract_preview(full: str, max_chars: int = 520) -> Tuple[str, bool]:
    s = (full or "").strip()
    if not s:
        return "", False
    if len(s) <= max_chars:
        return s, False
    return s[: max_chars - 1].rstrip() + "…", True

def apply_filter(items: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    q = (query or "").strip().lower()
    if not q:
        return items
    out: List[Dict[str, Any]] = []
    for it in items:
        paper = (it or {}).get("paper") or {}
        title = str(it.get("title") or paper.get("title") or "")
        authors = " ".join(get_hf_authors(paper))
        keywords = " ".join(get_keywords(paper))
        hay = f"{title}\n{authors}\n{keywords}".lower()
        if q in hay:
            out.append(it)
    return out

def apply_sort(items: List[Dict[str, Any]], sort_mode: str, extras_map: Dict[str, PaperExtras]) -> List[Dict[str, Any]]:
    if sort_mode == "API order":
        return items

    def upvotes(it: Dict[str, Any]) -> int:
        paper = (it or {}).get("paper") or {}
        v = paper.get("upvotes")
        try:
            return int(v)
        except Exception:
            return -1

    def gh_stars(it: Dict[str, Any]) -> int:
        paper = (it or {}).get("paper") or {}
        pid = str(paper.get("id") or it.get("id") or "").strip()
        ex = extras_map.get(pid)
        if ex and ex.github_stars is not None:
            return int(ex.github_stars)
        return -1

    def has_project(it: Dict[str, Any]) -> int:
        paper = (it or {}).get("paper") or {}
        pid = str(paper.get("id") or it.get("id") or "").strip()
        ex = extras_map.get(pid)
        return 1 if (ex and ex.project_page_url) else 0

    def title_key(it: Dict[str, Any]) -> str:
        paper = (it or {}).get("paper") or {}
        t = str(it.get("title") or paper.get("title") or "")
        return t.lower()

    if sort_mode == "Upvotes (desc)":
        return sorted(items, key=upvotes, reverse=True)
    if sort_mode == "GitHub stars (desc)":
        return sorted(items, key=gh_stars, reverse=True)
    if sort_mode == "Has project page (desc)":
        return sorted(items, key=has_project, reverse=True)
    if sort_mode == "Title (A→Z)":
        return sorted(items, key=title_key)
    return items


# =========================
# Main render
# =========================

def render(
    date_iso: str,
    limit: int,
    query: str,
    sort_mode: str,
    show_keywords: bool,
    expand_abstracts: bool,
    only_with_project: bool,
    only_with_github: bool,
    show_artifacts_counts: bool,
    diagnostics: bool,
) -> Tuple[str, str]:
    date_iso = (date_iso or "").strip()
    try:
        dt.date.fromisoformat(date_iso)
    except Exception:
        return "", "❌ Invalid date. Use YYYY-MM-DD (UTC)."

    limit = max(1, min(50, int(limit)))

    items, err, rate_headers, cached, effective_date = fetch_hf_daily(date_iso, limit)

    if err is not None:
        diag = ""
        if diagnostics:
            diag_txt = "\n".join(
                [
                    f"status_code: {err.status_code}",
                    f"reason: {err.reason}",
                    f"body_snippet: {err.body_snippet}",
                    f"rate_limit_headers: {err.headers}",
                    "hint: If you see 429, set HF_TOKEN in Space secrets.",
                ]
            )
            diag = f"<div class='diag'>{esc(diag_txt)}</div>"

        box = (
            "<div class='error-box'>"
            "<b>API request failed</b><br/>"
            f"Status: {esc(str(err.status_code))}<br/>"
            f"Reason: {esc(err.reason)}<br/>"
            f"Body (snippet): {esc(err.body_snippet)}"
            "</div>"
            f"{diag}"
        )
        return box, "❌ Failed to fetch daily papers."

    assert items is not None
    items = apply_filter(items, query)

    if not items:
        return "<div class='paper-list'></div>", "No results."

    # Gather ids
    paper_ids: List[str] = []
    for it in items:
        paper = (it or {}).get("paper") or {}
        pid = str(paper.get("id") or it.get("id") or "").strip()
        if pid:
            paper_ids.append(pid)

    # Fetch extras concurrently (project page, github, counts)
    extras_map: Dict[str, PaperExtras] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_paper_extras, pid): pid for pid in paper_ids}
        for f in as_completed(futs):
            pid = futs[f]
            try:
                extras_map[pid] = f.result()
            except Exception:
                extras_map[pid] = PaperExtras(
                    project_page_url=None,
                    github_repo_url=None,
                    github_stars=None,
                    published_date=None,
                    submitted_on_daily=None,
                    citing_models=None,
                    citing_datasets=None,
                    citing_spaces=None,
                )

    # Apply link-based filters (triage)
    if only_with_project or only_with_github:
        filtered: List[Dict[str, Any]] = []
        for it in items:
            paper = (it or {}).get("paper") or {}
            pid = str(paper.get("id") or it.get("id") or "").strip()
            exx = extras_map.get(pid)
            if only_with_project and not (exx and exx.project_page_url):
                continue
            if only_with_github and not (exx and exx.github_repo_url):
                continue
            filtered.append(it)
        items = filtered

    if not items:
        return "<div class='paper-list'></div>", "No results after filters."

    # arXiv abstracts (real abstracts when possible)
    arxiv_meta_map = fetch_arxiv_batch([pid for pid in paper_ids if is_arxiv_id(pid)])

    # Sort after extras are known
    items = apply_sort(items, sort_mode, extras_map)

    blocks: List[str] = ["<div class='paper-list'>"]

    for idx, it in enumerate(items, start=1):
        paper = (it or {}).get("paper") or {}
        pid = str(paper.get("id") or it.get("id") or "").strip()

        title = str(it.get("title") or paper.get("title") or "").strip() or "(no title)"
        upvotes = paper.get("upvotes")
        upvotes_text = str(upvotes) if upvotes is not None else "—"

        exx = extras_map.get(pid)
        arxiv_meta = arxiv_meta_map.get(pid)

        authors = arxiv_meta.authors if (arxiv_meta and arxiv_meta.authors) else get_hf_authors(paper)
        authors_text = format_authors(authors)

        # Abstract: arXiv abstract if possible, else HF summary
        abstract_full = arxiv_meta.abstract if (arxiv_meta and arxiv_meta.abstract) else normalize_text(str(it.get("summary") or paper.get("summary") or ""))
        preview, truncated = abstract_preview(abstract_full, max_chars=520)

        preview_html = (
            f"<div class='abstract-preview'>{esc(preview)}</div>"
            if preview
            else "<div class='abstract-preview'><span class='note'>Abstract not available.</span></div>"
        )

        full_html = ""
        if truncated:
            open_attr = " open" if expand_abstracts else ""
            full_html = (
                f"<details class='abstract-full'{open_attr}>"
                "<summary>Show full abstract</summary>"
                f"<div class='abstract-full-text'>{esc(abstract_full)}</div>"
                "</details>"
            )

        # Meta pills (triage signals)
        pills: List[str] = [
            f"<span class='pill'>Upvotes: {esc(upvotes_text)}</span>",
            f"<span class='pill'>Authors: {esc(authors_text)}</span>",
        ]

        if exx and exx.published_date:
            pills.append(f"<span class='pill'>Published: {esc(exx.published_date)}</span>")
        if exx and exx.submitted_on_daily:
            pills.append(f"<span class='pill'>Daily: {esc(exx.submitted_on_daily)}</span>")
        if exx and exx.github_stars is not None and exx.github_repo_url:
            pills.append(f"<span class='pill'>GitHub ★ {esc(str(exx.github_stars))}</span>")

        if show_artifacts_counts and exx:
            cm = "—" if exx.citing_models is None else str(exx.citing_models)
            cd = "—" if exx.citing_datasets is None else str(exx.citing_datasets)
            cs = "—" if exx.citing_spaces is None else str(exx.citing_spaces)
            pills.append(f"<span class='pill'>Artifacts: M {esc(cm)} · D {esc(cd)} · S {esc(cs)}</span>")

        meta_row = "<div class='meta'>" + "".join(pills) + "</div>"

        # Links: HF page / arXiv / PDF / Project page / GitHub (NO search fallback)
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

        # Keywords (optional)
        kw_html = ""
        if show_keywords:
            kws = get_keywords(paper)
            if kws:
                chips = "".join([f"<span class='chip'>{esc(k)}</span>" for k in kws[:18]])
                kw_html = f"<div class='chips'>{chips}</div>"

        blocks.append(
            "<article class='card'>"
            f"<div class='title'>{idx}. {esc(title)}</div>"
            f"{meta_row}"
            "<div class='abstract'>"
            "<h4>Abstract</h4>"
            f"{preview_html}"
            f"{full_html}"
            "</div>"
            f"{kw_html}"
            f"{link_row}"
            "</article>"
        )

    blocks.append("</div>")

    status = f"✅ Loaded {len(items)} paper(s) for {effective_date} (UTC)."
    if effective_date != date_iso:
        status += f" (requested {date_iso} was not available yet; auto-clamped)"
    if cached:
        status += " (cached)"
    if diagnostics and rate_headers:
        status += f" RateLimit: {rate_headers}"

    return "\n".join(blocks), status


# =========================
# Gradio app
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
                fetch_btn = gr.Button("Fetch", variant="primary")

            gr.HTML(
                "<div class='note'>"
                "<b>Important:</b> The API expects a <b>UTC date</b>. "
                "If you enter a future UTC date, the app will clamp to the latest available date."
                "</div>"
            )

            with gr.Accordion("Triage controls", open=True):
                with gr.Row():
                    query_in = gr.Textbox(label="Search (title / authors / keywords)", value="", max_lines=1)
                    sort_in = gr.Dropdown(
                        label="Sort",
                        choices=["API order", "Upvotes (desc)", "GitHub stars (desc)", "Has project page (desc)", "Title (A→Z)"],
                        value="Upvotes (desc)",
                    )

                with gr.Row():
                    only_project_in = gr.Checkbox(label="Only with project page", value=False)
                    only_github_in = gr.Checkbox(label="Only with GitHub repo", value=False)
                    show_counts_in = gr.Checkbox(label="Show artifacts counts (Models/Datasets/Spaces)", value=True)

                with gr.Row():
                    show_kw_in = gr.Checkbox(label="Show keywords", value=False)
                    expand_abs_in = gr.Checkbox(label="Auto-expand full abstracts", value=False)
                    diag_in = gr.Checkbox(label="Diagnostics", value=False)

        status_out = gr.Markdown()
        html_out = gr.HTML()

        fetch_btn.click(
            fn=render,
            inputs=[
                date_in, limit_in, query_in, sort_in,
                show_kw_in, expand_abs_in,
                only_project_in, only_github_in, show_counts_in,
                diag_in
            ],
            outputs=[html_out, status_out],
        )

        demo.load(
            fn=render,
            inputs=[
                date_in, limit_in, query_in, sort_in,
                show_kw_in, expand_abs_in,
                only_project_in, only_github_in, show_counts_in,
                diag_in
            ],
            outputs=[html_out, status_out],
        )

    return demo

demo = build_app()

if __name__ == "__main__":
    demo.queue().launch(css=CSS, theme=gr.themes.Soft(), ssr_mode=False)
