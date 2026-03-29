#!/usr/bin/env python3
"""
Fetch all publications from a Google Scholar profile page and:
  - Generate a BibTeX file (publications.bib) with full bibliometric info
    including abstracts.
  - Download open-access PDFs when available.
  - Download LaTeX source archives from arXiv when available, storing each
    paper's source in a separate folder named "<year>_<title_words>".

Usage:
    python fetch_publications.py "https://scholar.google.com/citations?user=XXXXXXXXX"
"""

from __future__ import annotations

import argparse
import difflib
import logging
import os
import re
import sys
import tarfile
import time
import types
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import requests
from scholarly import ProxyGenerator, scholarly


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    """Replace characters that are invalid in file/directory names."""
    return re.sub(r"[^\w\-]", "_", name)


def make_folder_name(pub: dict) -> str:
    """Return a human-readable folder name: ``YEAR_First_Five_Title_Words``."""
    bib = pub.get("bib", {})
    year = str(bib.get("pub_year", "unknown"))
    title = bib.get("title", "untitled")
    words = title.split()[:5]
    title_part = "_".join(words)
    return sanitize_filename(f"{year}_{title_part}")


def make_bibtex_key(pub: dict, used_keys: set) -> str:
    """Return a unique BibTeX cite-key derived from first-author + year + first title word."""
    bib = pub.get("bib", {})
    author = bib.get("author", "")
    year = str(bib.get("pub_year", "unknown"))
    title = bib.get("title", "")

    if author:
        first_author = author.split(" and ")[0].strip()
        if "," in first_author:
            last_name = first_author.split(",")[0].strip()
        else:
            parts = first_author.split()
            last_name = parts[-1] if parts else "unknown"
    else:
        last_name = "unknown"

    title_word = title.split()[0] if title else "unknown"
    base_key = sanitize_filename(f"{last_name}{year}{title_word}").lower()

    key = base_key
    counter = 1
    while key in used_keys:
        key = f"{base_key}{counter}"
        counter += 1
    used_keys.add(key)
    return key


def _escape_bibtex(value: str) -> str:
    """Escape unbalanced braces so the value is safe inside ``{...}``."""
    # Replace bare % that are not already escaped
    value = re.sub(r"(?<!\\)%", r"\\%", value)
    return value


def format_bibtex(pub: dict, key: str) -> str:
    """Return a BibTeX entry string for *pub* using cite-key *key*.

    scholarly fills ``bib['journal']`` and ``bib['conference']`` for
    AUTHOR_PUBLICATION_ENTRY publications (filled from the detail page).
    ``bib['venue']`` is the abbreviated venue from the author-page snippet.
    We prefer the explicit journal/conference fields when available.
    """
    bib = pub.get("bib", {})

    journal = bib.get("journal", "")
    conference = bib.get("conference", "")
    venue = bib.get("venue", "")

    if conference:
        entry_type = "inproceedings"
        venue_name = conference
    elif journal:
        entry_type = "article"
        venue_name = journal
    else:
        # Fall back to venue-string heuristics (PUBLICATION_SEARCH_SNIPPET)
        venue_lower = venue.lower()
        if any(kw in venue_lower for kw in ("conference", "proceedings", "workshop", "symposium")):
            entry_type = "inproceedings"
        elif "thesis" in venue_lower:
            entry_type = "phdthesis"
        elif "book" in venue_lower:
            entry_type = "book"
        else:
            entry_type = "article"
        venue_name = venue

    lines = [f"@{entry_type}{{{key},"]

    def add_field(bibtex_name: str, value: str, protect_case: bool = False) -> None:
        if not value:
            return
        value = _escape_bibtex(str(value))
        if protect_case:
            value = "{" + value + "}"
        lines.append(f"  {bibtex_name} = {{{value}}},")

    add_field("title", bib.get("title", ""), protect_case=True)
    add_field("author", bib.get("author", ""))
    add_field("year", str(bib.get("pub_year", "")))

    if entry_type == "inproceedings":
        add_field("booktitle", venue_name)
    else:
        add_field("journal", venue_name)

    add_field("volume", bib.get("volume", ""))
    add_field("number", bib.get("number", ""))
    add_field("pages", bib.get("pages", ""))
    add_field("publisher", bib.get("publisher", ""))
    add_field("abstract", bib.get("abstract", ""))

    pub_url = pub.get("pub_url", "")
    if pub_url:
        add_field("url", pub_url)

    arxiv_id = extract_arxiv_id(pub)
    if arxiv_id:
        add_field("eprint", arxiv_id)
        lines.append("  archivePrefix = {arXiv},")

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# arXiv helpers
# ---------------------------------------------------------------------------

_ARXIV_PATTERNS = [
    re.compile(r"arxiv\.org/(?:abs|pdf)/([^\s/?#]+)", re.IGNORECASE),
]


def extract_arxiv_id(pub: dict) -> str:
    """Return the arXiv identifier found in the publication URLs, or ''."""
    candidates = [
        pub.get("eprint_url", ""),
        pub.get("pub_url", ""),
    ]
    for url in candidates:
        if not url:
            continue
        for pattern in _ARXIV_PATTERNS:
            m = pattern.search(url)
            if m:
                arxiv_id = m.group(1)
                # Strip trailing .pdf suffix if present
                arxiv_id = re.sub(r"\.pdf$", "", arxiv_id, flags=re.IGNORECASE)
                return arxiv_id
    return ""


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

_HEADERS = {"User-Agent": "PublicationsFetcher/1.0 (https://github.com/jsjol/update-publications)"}


def _is_pdf_response(resp: requests.Response) -> bool:
    """Return True when *resp* looks like a PDF (by Content-Type or magic bytes)."""
    content_type = resp.headers.get("Content-Type", "").lower()
    if "pdf" in content_type:
        return True
    # Some servers use application/octet-stream or omit the type entirely.
    # Fall back to checking the PDF magic bytes.
    return resp.content[:4] == b"%PDF"


_BIORXIV_RE = re.compile(r"(biorxiv|medrxiv)\.org/content/", re.IGNORECASE)
_OPENREVIEW_RE = re.compile(r"openreview\.net/forum\?", re.IGNORECASE)
# Match PMLR paper pages ending in .html or .pdf (both occur in practice).
_PMLR_RE = re.compile(r"proceedings\.mlr\.press/v\d+/[^/]+\.(?:html|pdf)$", re.IGNORECASE)
_DIVA_RE = re.compile(r"diva-portal\.org/smash/record\.jsf", re.IGNORECASE)
_PATENTS_RE = re.compile(r"patents\.google\.com/patent/", re.IGNORECASE)
# NeurIPS / NIPS proceedings (both legacy papers.nips.cc and proceedings.neurips.cc).
_NEURIPS_RE = re.compile(
    r"(?:papers\.nips\.cc|proceedings\.neurips\.cc)/paper",
    re.IGNORECASE,
)
# JMLR paper or proceedings pages (open-access; PDF lives at same path with .pdf extension).
_JMLR_RE = re.compile(
    r"jmlr\.org/(?:papers|proceedings/papers)/v\d+/",
    re.IGNORECASE,
)
# Matches the PDF download URL embedded in Google Patents HTML pages.
_PATENT_PDF_EMBED_RE = re.compile(
    r'https://patentimages\.storage\.googleapis\.com/[^"\'<>\s]+\.pdf',
    re.IGNORECASE,
)
# Matches PDF links in DiVA portal record HTML pages.
_DIVA_PDF_LINK_RE = re.compile(
    r'href="(/smash/get/diva2:[^/]+/[^"]+\.pdf)"',
    re.IGNORECASE,
)


def _to_pdf_url(url: str, pub: dict) -> list[str]:
    """Return a list of concrete PDF URL(s) derived from *url*.

    Handles known preprint/venue patterns:
    * arXiv abstract → direct PDF URL
    * bioRxiv / medRxiv abstract → ``.full.pdf`` URL (with original as fallback)
    * OpenReview forum page → PDF page
    * PMLR paper page (.html or .pdf) → subdirectory PDF URL, simple PDF URL, HTML fallback
    * DiVA portal record page → FULLTEXT01.pdf URL (with original as fallback)
    * Google Patents page → returned as-is (PDF extracted by download_pdf scraping)
    * All other URLs are returned unchanged (may already be direct PDF links).
    """
    if not url:
        return []
    if re.search(r"arxiv\.org/abs/", url, re.IGNORECASE):
        arxiv_id = extract_arxiv_id(pub)
        return [f"https://arxiv.org/pdf/{arxiv_id}.pdf"]
    if _BIORXIV_RE.search(url):
        # PDF URL: strip trailing slash/version fragment, then append .full.pdf
        pdf_url = url.rstrip("/") + ".full.pdf"
        # Also keep the original as fallback in case the server redirects
        return [pdf_url, url]
    if _OPENREVIEW_RE.search(url):
        # Convert forum?id=X to pdf?id=X; keep original as fallback
        pdf_url = re.sub(r"forum\?", "pdf?", url, flags=re.IGNORECASE)
        return [pdf_url, url]
    if _PMLR_RE.search(url):
        # PMLR canonical PDF is at {version}/{paper_id}/{paper_id}.pdf (subdirectory).
        # Also try the flat .pdf swap and the .html page as ordered fallbacks.
        parsed = urlparse(url)
        stem = re.sub(r"\.(html|pdf)$", "", parsed.path.rsplit("/", 1)[-1], flags=re.IGNORECASE)
        base_path = parsed.path.rsplit("/", 1)[0]
        origin = f"{parsed.scheme}://{parsed.netloc}"
        subdir_pdf = f"{origin}{base_path}/{stem}/{stem}.pdf"
        simple_pdf = f"{origin}{base_path}/{stem}.pdf"
        html_url = f"{origin}{base_path}/{stem}.html"
        seen: set[str] = set()
        result: list[str] = []
        for u in [subdir_pdf, simple_pdf, html_url]:
            if u not in seen:
                seen.add(u)
                result.append(u)
        return result
    if _NEURIPS_RE.search(url):
        # NeurIPS abstract HTML → paper PDF.
        # Pattern: .../hash/HASH-Abstract[-Suffix].html → .../file/HASH-Paper[-Suffix].pdf
        pdf_url = url.replace("/hash/", "/file/")
        pdf_url = re.sub(r"-Abstract((?:-[^.]+)?)\.html$", r"-Paper\1.pdf", pdf_url,
                         flags=re.IGNORECASE)
        if pdf_url != url:
            return [pdf_url, url]
        return [url]
    if _JMLR_RE.search(url):
        # JMLR paper pages: swap .html → .pdf; keep original as fallback.
        if url.lower().endswith(".html"):
            pdf_url = re.sub(r"\.html$", ".pdf", url, flags=re.IGNORECASE)
            return [pdf_url, url]
        return [url]
    if _DIVA_RE.search(url):
        # DiVA portal: extract diva2:XXXXXX from the ?pid= query parameter and
        # construct the FULLTEXT01.pdf URL.
        # E.g. record.jsf?pid=diva2%3A1638041 -> get/diva2:1638041/FULLTEXT01.pdf
        qs = parse_qs(urlparse(url).query)
        pid_values = qs.get("pid", [])
        if pid_values:
            diva_id = unquote(pid_values[0])  # e.g. "diva2:1638041"
            base = re.match(r"(https?://[^/]+)", url, re.IGNORECASE)
            if base:
                pdf_url = f"{base.group(1)}/smash/get/{diva_id}/FULLTEXT01.pdf"
                return [pdf_url, url]
    if _PATENTS_RE.search(url):
        # Google Patents page: return as-is; download_pdf will scrape the PDF link.
        return [url]
    return [url]


def _extract_patent_pdf_url(html: str) -> str:
    """Extract the PDF download URL embedded in a Google Patents HTML page."""
    # Normalize escaped forward slashes (common in JSON-embedded HTML content).
    normalized = html.replace("\\/", "/")
    m = _PATENT_PDF_EMBED_RE.search(normalized)
    return m.group(0) if m else ""


def _extract_diva_pdf_url(html: str, record_url: str) -> str:
    """Extract a full-text PDF URL from a DiVA portal record HTML page."""
    m = _DIVA_PDF_LINK_RE.search(html)
    if m:
        base = re.match(r"(https?://[^/]+)", record_url, re.IGNORECASE)
        if base:
            return base.group(1) + m.group(1)
    return ""


def _openreview_api_pdf_url(forum_id: str) -> str:
    """Try to resolve a PDF URL via the OpenReview API for *forum_id*.

    Returns the full PDF URL string on success, or '' if unavailable.
    """
    for api_base in ("https://api2.openreview.net", "https://api.openreview.net"):
        try:
            resp = requests.get(
                f"{api_base}/notes?id={forum_id}",
                headers=_HEADERS,
                timeout=30,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            notes = data.get("notes", [])
            for note in notes:
                pdf_path = note.get("content", {}).get("pdf", "")
                if isinstance(pdf_path, str) and pdf_path:
                    if pdf_path.startswith("/"):
                        return f"https://openreview.net{pdf_path}"
                    return pdf_path
        except Exception:
            pass
    return ""


def _semantic_scholar_pdf_url(pub: dict) -> str:
    """Try to find an open-access PDF URL via the Semantic Scholar API.

    Uses DOI-based lookup (precise) when a DOI is available; otherwise
    falls back to a title search with a similarity confidence check.
    Returns the PDF URL string on success, or '' if unavailable.
    """
    bib = pub.get("bib", {})
    doi = bib.get("doi", "")
    title = bib.get("title", "")

    if doi:
        try:
            resp = requests.get(
                f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
                params={"fields": "openAccessPdf"},
                headers=_HEADERS,
                timeout=15,
            )
            if resp.status_code == 200:
                oap = resp.json().get("openAccessPdf") or {}
                if oap.get("url"):
                    return oap["url"]
        except Exception:
            pass

    if title:
        try:
            # limit=1: the top-ranked result is the most likely match; checking
            # additional results risks returning a PDF for a different paper.
            resp = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": title, "fields": "openAccessPdf,title", "limit": 1},
                headers=_HEADERS,
                timeout=15,
            )
            if resp.status_code == 200:
                title_norm = re.sub(r"\W+", " ", title).strip().lower()
                for paper in resp.json().get("data", []):
                    result_title = re.sub(r"\W+", " ", paper.get("title", "")).strip().lower()
                    # Require ≥85% character-sequence similarity to avoid returning
                    # a PDF for a different paper with overlapping keywords.
                    similarity = difflib.SequenceMatcher(None, title_norm, result_title).ratio()
                    if similarity >= 0.85:
                        oap = paper.get("openAccessPdf") or {}
                        if oap.get("url"):
                            return oap["url"]
        except Exception:
            pass

    return ""


def _pdf_url_candidates(pub: dict) -> list[str]:
    """Return a list of URLs to try in order when looking for a PDF.

    When Google Scholar marks a paper as publicly accessible (``public_access``
    is True) but ``eprint_url`` is absent, we fall back to ``pub_url``
    unconditionally — ``_is_pdf_response`` will reject non-PDF responses.
    """
    seen: set[str] = set()
    candidates: list[str] = []

    def add(url: str) -> None:
        if url and url not in seen:
            candidates.append(url)
            seen.add(url)

    for url in _to_pdf_url(pub.get("eprint_url", ""), pub):
        add(url)

    pub_url = pub.get("pub_url", "")
    if pub_url:
        # Always try pub_url when Google Scholar marks the paper as open access,
        # or when the URL already looks like a direct PDF / known preprint server.
        is_open_access = bool(pub.get("public_access"))
        is_known_pattern = (
            re.search(r"arxiv\.org/", pub_url, re.IGNORECASE)
            or _BIORXIV_RE.search(pub_url)
            or _OPENREVIEW_RE.search(pub_url)
            or _PMLR_RE.search(pub_url)
            or _NEURIPS_RE.search(pub_url)
            or _JMLR_RE.search(pub_url)
            or _DIVA_RE.search(pub_url)
            or _PATENTS_RE.search(pub_url)
            or pub_url.lower().endswith(".pdf")
        )
        if is_open_access or is_known_pattern:
            for url in _to_pdf_url(pub_url, pub):
                add(url)

    return candidates


def download_pdf(pub: dict, pub_dir: Path, force: bool = False) -> bool:
    """Download the open-access PDF into *pub_dir*/paper.pdf.  Returns True on success.

    When *force* is False (the default) and *paper.pdf* already exists inside
    *pub_dir*, the download is skipped and True is returned immediately.
    """
    pdf_path = pub_dir / "paper.pdf"
    if not force and pdf_path.exists():
        print(f"    PDF already exists, skipping → {pdf_path}")
        return True
    for pdf_url in _pdf_url_candidates(pub):
        try:
            resp = requests.get(pdf_url, headers=_HEADERS, timeout=60, allow_redirects=True)
            if resp.status_code == 200 and _is_pdf_response(resp):
                pdf_path = pub_dir / "paper.pdf"
                pdf_path.write_bytes(resp.content)
                print(f"    PDF saved → {pdf_path}")
                return True
            # For Google Patents pages the response is HTML; extract the embedded
            # PDF link (patentimages.storage.googleapis.com/…) and download it.
            if (
                resp.status_code == 200
                and _PATENTS_RE.search(pdf_url)
                and "html" in resp.headers.get("Content-Type", "").lower()
            ):
                extracted_pdf_url = _extract_patent_pdf_url(resp.text)
                if extracted_pdf_url:
                    try:
                        pdf_resp = requests.get(
                            extracted_pdf_url, headers=_HEADERS, timeout=60, allow_redirects=True
                        )
                        if pdf_resp.status_code == 200 and _is_pdf_response(pdf_resp):
                            pdf_path = pub_dir / "paper.pdf"
                            pdf_path.write_bytes(pdf_resp.content)
                            print(f"    PDF saved → {pdf_path}")
                            return True
                        print(
                            f"    Patent PDF not available at {extracted_pdf_url} "
                            f"(status {pdf_resp.status_code}, "
                            f"content-type: {pdf_resp.headers.get('Content-Type', '')})"
                        )
                    except requests.RequestException as exc:
                        print(f"    Warning: Patent PDF download failed for {extracted_pdf_url}: {exc}")
            # For DiVA record pages the response is HTML; scrape for PDF links.
            if (
                resp.status_code == 200
                and _DIVA_RE.search(pdf_url)
                and "html" in resp.headers.get("Content-Type", "").lower()
            ):
                extracted_pdf_url = _extract_diva_pdf_url(resp.text, pdf_url)
                if extracted_pdf_url:
                    try:
                        pdf_resp = requests.get(
                            extracted_pdf_url, headers=_HEADERS, timeout=60, allow_redirects=True
                        )
                        if pdf_resp.status_code == 200 and _is_pdf_response(pdf_resp):
                            pdf_path = pub_dir / "paper.pdf"
                            pdf_path.write_bytes(pdf_resp.content)
                            print(f"    PDF saved → {pdf_path}")
                            return True
                        print(
                            f"    DiVA PDF not available at {extracted_pdf_url} "
                            f"(status {pdf_resp.status_code}, "
                            f"content-type: {pdf_resp.headers.get('Content-Type', '')})"
                        )
                    except requests.RequestException as exc:
                        print(f"    Warning: DiVA PDF download failed for {extracted_pdf_url}: {exc}")
            # For OpenReview pages blocked with 403, try the OpenReview API.
            if resp.status_code == 403 and re.search(r"openreview\.net", pdf_url, re.IGNORECASE):
                qs = parse_qs(urlparse(pdf_url).query)
                forum_id = qs.get("id", [""])[0]
                if forum_id:
                    api_pdf = _openreview_api_pdf_url(forum_id)
                    if api_pdf:
                        try:
                            pdf_resp = requests.get(
                                api_pdf, headers=_HEADERS, timeout=60, allow_redirects=True
                            )
                            if pdf_resp.status_code == 200 and _is_pdf_response(pdf_resp):
                                pdf_path = pub_dir / "paper.pdf"
                                pdf_path.write_bytes(pdf_resp.content)
                                print(f"    PDF saved → {pdf_path}")
                                return True
                            print(
                                f"    OpenReview PDF not available at {api_pdf} "
                                f"(status {pdf_resp.status_code}, "
                                f"content-type: {pdf_resp.headers.get('Content-Type', '')})"
                            )
                        except requests.RequestException as exc:
                            print(f"    Warning: OpenReview PDF download failed for {api_pdf}: {exc}")
            print(f"    PDF not available at {pdf_url} (status {resp.status_code}, "
                  f"content-type: {resp.headers.get('Content-Type', '')})")
        except requests.RequestException as exc:
            print(f"    Warning: PDF download failed for {pdf_url}: {exc}")
    # Final fallback: Semantic Scholar open access PDF (tried once after all other
    # candidates are exhausted to avoid extra API calls on easy-to-download papers).
    ss_url = _semantic_scholar_pdf_url(pub)
    if ss_url:
        try:
            resp = requests.get(ss_url, headers=_HEADERS, timeout=60, allow_redirects=True)
            if resp.status_code == 200 and _is_pdf_response(resp):
                pdf_path = pub_dir / "paper.pdf"
                pdf_path.write_bytes(resp.content)
                print(f"    PDF saved (via Semantic Scholar) → {pdf_path}")
                return True
            print(
                f"    Semantic Scholar PDF not available at {ss_url} "
                f"(status {resp.status_code}, "
                f"content-type: {resp.headers.get('Content-Type', '')})"
            )
        except requests.RequestException as exc:
            print(f"    Warning: Semantic Scholar PDF download failed for {ss_url}: {exc}")
    return False


def download_arxiv_source(arxiv_id: str, pub_dir: Path, force: bool = False) -> bool:
    """Download arXiv LaTeX source into *pub_dir*.  Returns True on success.

    When *force* is False (the default) and *pub_dir* already contains files
    other than *paper.pdf* (i.e. a previous source download), the download is
    skipped and True is returned immediately.
    """
    if not arxiv_id:
        return False

    if not force:
        existing_source_files = [f for f in pub_dir.iterdir() if f.name != "paper.pdf"]
        if existing_source_files:
            print(f"    arXiv source already exists, skipping → {pub_dir}")
            return True

    source_url = f"https://arxiv.org/src/{arxiv_id}"
    try:
        resp = requests.get(source_url, headers=_HEADERS, timeout=120, allow_redirects=True)
        if resp.status_code != 200:
            print(f"    arXiv source not available (status {resp.status_code})")
            return False

        # arXiv serves either a tar.gz or a plain .tex file
        content_type = resp.headers.get("Content-Type", "")
        raw = resp.content

        # Try to open as a tar archive first
        tmp_path = pub_dir / "_source_download"
        tmp_path.write_bytes(raw)
        try:
            with tarfile.open(tmp_path) as tar:
                def _safe_members(archive):
                    """Yield members that are safe to extract (no path traversal)."""
                    for member in archive.getmembers():
                        member_path = Path(member.name)
                        # Reject absolute paths and paths containing ".."
                        if member_path.is_absolute() or ".." in member_path.parts:
                            print(f"    Skipping unsafe tar member: {member.name}")
                            continue
                        yield member
                tar.extractall(pub_dir, members=_safe_members(tar), filter="data")
            tmp_path.unlink()
            print(f"    LaTeX source extracted → {pub_dir}")
            return True
        except tarfile.TarError:
            # Not a tar archive – treat as a plain .tex file
            tex_path = pub_dir / "source.tex"
            tmp_path.rename(tex_path)
            print(f"    LaTeX source saved → {tex_path}")
            return True

    except requests.RequestException as exc:
        print(f"    Warning: arXiv source download failed: {exc}")
    return False


# ---------------------------------------------------------------------------
# Browser-based CAPTCHA fallback
# ---------------------------------------------------------------------------

def _make_browser_proxy_generator() -> ProxyGenerator:
    """Return a ProxyGenerator configured to open a *visible* browser window.

    ``scholarly``'s built-in CAPTCHA handler (``_handle_captcha2``) opens a
    Selenium browser and waits until the CAPTCHA disappears.  By default it
    uses ``--headless``, so the user never sees the page and the script hangs
    forever.  This function monkey-patches ``_get_chrome_webdriver``,
    ``_get_firefox_webdriver``, and ``_handle_captcha2`` so that:

    1. The browser window is *visible* so the user can solve the CAPTCHA.
    2. After solving, cookies are copied back into the ``httpx.Client`` session
       correctly.  scholarly 1.7.x uses ``httpx.Client`` whose ``Cookies.set()``
       only accepts ``(name, value, domain, path)`` — passing extra Selenium
       cookie fields (e.g. ``secure``) causes a ``TypeError`` that makes
       scholarly discard the session and immediately ask for another CAPTCHA.
    """
    pg = ProxyGenerator()

    def _get_chrome_webdriver(self: ProxyGenerator):
        from selenium import webdriver as _webdriver
        options = _webdriver.ChromeOptions()
        # Intentionally no --headless: the user must be able to see and solve the CAPTCHA.
        self._webdriver = _webdriver.Chrome(options=options)
        self._webdriver.get("https://scholar.google.com")
        return self._webdriver

    def _get_firefox_webdriver(self: ProxyGenerator):
        from selenium import webdriver as _webdriver
        from selenium.webdriver.firefox.options import Options as _FirefoxOptions
        options = _FirefoxOptions()
        # Intentionally no --headless: the user must be able to see and solve the CAPTCHA.
        self._webdriver = _webdriver.Firefox(options=options)
        self._webdriver.get("https://scholar.google.com")
        return self._webdriver

    def _handle_captcha2(self: ProxyGenerator, url: str):
        """Fixed version of scholarly's _handle_captcha2.

        Identical to the upstream implementation except that the cookie-copying
        loop at the end calls ``httpx.Cookies.set(name, value, domain, path)``
        explicitly instead of unpacking the full Selenium cookie dict with
        ``**cookie``.  The upstream code leaves ``secure`` (and potentially
        other browser-specific keys) in the dict, which causes
        ``TypeError: Cookies.set() got an unexpected keyword argument 'secure'``
        on httpx ≥ 0.23.  That exception makes scholarly discard the session
        and open a fresh one — triggering yet another CAPTCHA immediately.
        """
        from urllib.parse import urlparse as _urlparse
        from selenium.webdriver.support.ui import WebDriverWait as _WebDriverWait
        from selenium.common.exceptions import (
            TimeoutException as _TimeoutException,
            UnexpectedAlertPresentException as _UnexpectedAlertPresentException,
            WebDriverException as _WebDriverException,
        )
        from scholarly._proxy_generator import DOSException as _DOSException
        import time as _time

        cur_host = _urlparse(self._get_webdriver().current_url).hostname
        for cookie in self._session.cookies:
            if cur_host == cookie.domain.lstrip("."):
                self._get_webdriver().add_cookie({
                    "name": cookie.name,
                    "value": cookie.value,
                    "path": cookie.path,
                    "domain": cookie.domain,
                })
        self._get_webdriver().get(url)

        log_interval = 10
        cur = 0
        timeout = 60 * 60 * 24 * 7  # 1 week
        while cur < timeout:
            try:
                cur = cur + log_interval
                _WebDriverWait(self._get_webdriver(), log_interval).until_not(
                    lambda drv: self._webdriver_has_captcha()
                )
                break
            except _TimeoutException:
                self.logger.info(
                    f"Solving the captcha took already {cur} seconds "
                    f"(of maximum {timeout} s)."
                )
            except _UnexpectedAlertPresentException as e:
                self.logger.info(
                    f"Unexpected alert while waiting for captcha completion: {e.args}"
                )
                _time.sleep(15)
            except _DOSException as e:
                self.logger.info("Google thinks we are DOSing the captcha.")
                raise e
            except _WebDriverException as e:
                self.logger.info("Browser seems to be dysfunctional - closed by user?")
                raise e
            except Exception as e:
                self.logger.info(
                    f"Unhandled {type(e).__name__} while waiting for captcha "
                    f"completion: {e.args}"
                )
        else:
            raise _TimeoutException(
                f"Could not solve captcha in time (within {timeout} s)."
            )
        self.logger.info(f"Solved captcha in less than {cur} seconds.")

        # Copy browser cookies back into the httpx.Client session.
        # httpx.Cookies.set() only accepts (name, value, domain, path).
        # Passing the full Selenium cookie dict with **cookie crashes on 'secure'
        # and any other browser-specific keys not recognised by httpx.
        for cookie in self._get_webdriver().get_cookies():
            self._session.cookies.set(
                name=cookie["name"],
                value=cookie["value"],
                domain=cookie.get("domain", ""),
                path=cookie.get("path", "/"),
            )

        return self._session

    pg._get_chrome_webdriver = types.MethodType(_get_chrome_webdriver, pg)
    pg._get_firefox_webdriver = types.MethodType(_get_firefox_webdriver, pg)
    pg._handle_captcha2 = types.MethodType(_handle_captcha2, pg)
    return pg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch all publications from a Google Scholar profile, generate a "
            "BibTeX file with abstracts, download open-access PDFs and arXiv "
            "LaTeX sources."
        )
    )
    parser.add_argument("scholar_url", help="Google Scholar profile URL")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory in which to write publications.bib and downloaded files (default: current directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally",
    )
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF downloads")
    parser.add_argument(
        "--no-source", action="store_true", help="Skip LaTeX source downloads"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between Scholar requests to avoid rate-limiting (default: 2)",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help=(
            "Open a visible browser window so you can solve a Google Scholar CAPTCHA "
            "manually.  Use this when the tool is blocked by rate-limiting or CAPTCHA "
            "challenges.  Requires Chrome or Firefox with the matching WebDriver "
            "(chromedriver / geckodriver) on your PATH."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract the 'user' query parameter from the URL
    m = re.search(r"[?&]user=([^&]+)", args.scholar_url)
    if not m:
        parser.error(
            "Could not find 'user=…' in the URL.\n"
            "Expected format: https://scholar.google.com/citations?user=XXXXXXXXX"
        )

    author_id = m.group(1)
    print(f"Author ID: {author_id}")

    # ------------------------------------------------------------------
    # Optional: configure scholarly with a visible browser so CAPTCHAs
    # can be solved manually.  scholarly's built-in _handle_captcha2()
    # already waits for the CAPTCHA to disappear; we only need to remove
    # the --headless flag so the user can actually see and interact with
    # the browser window.
    #
    # We use ONE ProxyGenerator for both scholarly's primary (pm1) and
    # secondary (pm2) slots.  The navigator routes "citations?" URLs to
    # pm2 and everything else to pm1; with separate generators each slot
    # would require its own CAPTCHA solve.  Sharing a single generator
    # means they share the same httpx.Client and cookies, so one CAPTCHA
    # solve covers all subsequent requests.
    #
    # We also attach a StreamHandler to scholarly's INFO logger so the
    # user can see messages like "Got a captcha request." or
    # "Will retry after N seconds" instead of silent hangs.
    # ------------------------------------------------------------------
    _scholarly_logger = logging.getLogger("scholarly")
    if not _scholarly_logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        _scholarly_logger.addHandler(_handler)
    _scholarly_logger.setLevel(logging.INFO)

    if args.browser:
        print(
            "Browser mode enabled.  A browser window will open automatically if "
            "Google Scholar presents a CAPTCHA — solve it once there and the "
            "script will continue."
        )
        pg = _make_browser_proxy_generator()
        scholarly.use_proxy(pg, pg)

    # Fetch author profile
    try:
        author = scholarly.search_author_id(author_id)
        author = scholarly.fill(author, sections=["basics", "indices", "publications"])
    except Exception as exc:
        print(f"Error fetching author profile: {type(exc).__name__}: {exc}")
        if not args.browser:
            print(
                "Tip: if Google Scholar is blocking the request, re-run the same "
                "command with --browser added to open a browser window where you "
                "can solve a CAPTCHA manually."
            )
        sys.exit(1)

    name = author.get("name", "Unknown")
    publications = author.get("publications", [])
    print(f"Author: {name}")
    print(f"Publications found: {len(publications)}")
    if not publications:
        print(
            "Warning: no publications found. This may indicate that Google Scholar "
            "blocked the request. Try again later or increase --delay."
        )

    used_keys: set[str] = set()
    # Each record stores the pub dict, its cite-key, and its current BibTeX string
    # together to avoid parallel lists that could go out of sync.
    records: list[dict] = []  # keys: "pub", "key", "bibtex"

    bib_path = output_dir / "publications.bib"

    def _bib_text() -> str:
        return "\n\n".join(r["bibtex"] for r in records) + "\n"

    # ------------------------------------------------------------------
    # Phase 1: Write ALL publications with basic info to the BibTeX file.
    # This guarantees a complete bib (all references present) even if the
    # subsequent enrichment phase hangs or fails for some entries.
    # ------------------------------------------------------------------
    print("\n--- Phase 1: Building initial BibTeX from basic publication data ---")
    for idx, pub in enumerate(publications, start=1):
        raw_title = pub.get("bib", {}).get("title", "Unknown")
        print(f"  [{idx}/{len(publications)}] {raw_title[:80]}")

        key = make_bibtex_key(pub, used_keys)
        records.append({"pub": pub, "key": key, "bibtex": format_bibtex(pub, key)})

        # Write after each entry so partial results survive an interruption
        bib_path.write_text(_bib_text(), encoding="utf-8")

    print(f"\nPhase 1 complete.  BibTeX file: {bib_path}  ({len(records)} entries)")

    # ------------------------------------------------------------------
    # Phase 2: Enrich each publication with full details (abstract, URLs,
    # venue fields, …).  The BibTeX entry is updated in-place and the file
    # is rewritten after each successful enrichment so that progress is
    # preserved even if this phase is interrupted.
    # ------------------------------------------------------------------
    print("\n--- Phase 2: Enriching publications with full details (abstract, URLs, …) ---")
    for idx, record in enumerate(records, start=1):
        raw_title = record["pub"].get("bib", {}).get("title", "Unknown")
        print(f"\n[{idx}/{len(records)}] {raw_title[:80]}")

        print("  Fetching full details (may be slow if Scholar is rate-limiting)...",
              end="", flush=True)
        try:
            pub = scholarly.fill(record["pub"])
            print(" done.")
            time.sleep(args.delay)
            # Regenerate the BibTeX key now that full author info is available.
            # The initial key from Phase 1 may have been "unknown..." because the
            # author field is absent in the basic snippet returned by Scholar.
            used_keys.discard(record["key"])
            record["key"] = make_bibtex_key(pub, used_keys)
        except Exception as exc:
            print(f" failed.")
            print(f"  Warning: could not fetch full details: {exc}")
            pub = record["pub"]

        record["pub"] = pub
        record["bibtex"] = format_bibtex(pub, record["key"])

        # Rewrite the whole file so partial enrichment is not lost
        bib_path.write_text(_bib_text(), encoding="utf-8")

    print(f"\nPhase 2 complete.  BibTeX file: {bib_path}  ({len(records)} entries)")

    # ------------------------------------------------------------------
    # Phase 3: Download PDFs.
    # ------------------------------------------------------------------
    if not args.no_pdf:
        print("\n--- Phase 3: Downloading PDFs ---")
        for idx, record in enumerate(records, start=1):
            pub = record["pub"]
            raw_title = pub.get("bib", {}).get("title", "Unknown")
            candidates = _pdf_url_candidates(pub)
            if not candidates:
                continue
            print(f"\n[{idx}/{len(records)}] {raw_title[:80]}")
            pub_dir = output_dir / make_folder_name(pub)
            pub_dir.mkdir(parents=True, exist_ok=True)
            try:
                download_pdf(pub, pub_dir, force=args.force)
            except Exception as exc:
                print(f"    Warning: unexpected error during PDF download: {exc}")

    # ------------------------------------------------------------------
    # Phase 4: Download LaTeX sources from arXiv.
    # ------------------------------------------------------------------
    if not args.no_source:
        print("\n--- Phase 4: Downloading LaTeX sources ---")
        for idx, record in enumerate(records, start=1):
            pub = record["pub"]
            arxiv_id = extract_arxiv_id(pub)
            if not arxiv_id:
                continue
            raw_title = pub.get("bib", {}).get("title", "Unknown")
            print(f"\n[{idx}/{len(records)}] {raw_title[:80]}")
            pub_dir = output_dir / make_folder_name(pub)
            pub_dir.mkdir(parents=True, exist_ok=True)
            try:
                download_arxiv_source(arxiv_id, pub_dir, force=args.force)
            except Exception as exc:
                print(f"    Warning: unexpected error during arXiv source download: {exc}")
            time.sleep(args.delay)

    print(f"\nDone.  BibTeX file: {bib_path}  ({len(records)} entries)")


if __name__ == "__main__":
    main()
