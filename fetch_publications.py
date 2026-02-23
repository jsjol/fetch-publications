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
import os
import re
import sys
import tarfile
import time
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import requests
from scholarly import scholarly


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
_PMLR_RE = re.compile(r"proceedings\.mlr\.press/.*\.html$", re.IGNORECASE)
_DIVA_RE = re.compile(r"diva-portal\.org/smash/record\.jsf", re.IGNORECASE)


def _to_pdf_url(url: str, pub: dict) -> list[str]:
    """Return a list of concrete PDF URL(s) derived from *url*.

    Handles known preprint/venue patterns:
    * arXiv abstract → direct PDF URL
    * bioRxiv / medRxiv abstract → ``.full.pdf`` URL (with original as fallback)
    * OpenReview forum page → PDF page
    * PMLR HTML paper page → PDF URL
    * DiVA portal record page → FULLTEXT01.pdf URL (with original as fallback)
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
        # Convert .html to .pdf; keep original as fallback
        pdf_url = re.sub(r"\.html$", ".pdf", url, flags=re.IGNORECASE)
        return [pdf_url, url]
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
    return [url]


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
            or _DIVA_RE.search(pub_url)
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
            print(f"    PDF not available at {pdf_url} (status {resp.status_code}, "
                  f"content-type: {resp.headers.get('Content-Type', '')})")
        except requests.RequestException as exc:
            print(f"    Warning: PDF download failed for {pdf_url}: {exc}")
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

    # Fetch author profile
    try:
        author = scholarly.search_author_id(author_id)
        author = scholarly.fill(author, sections=["basics", "indices", "publications"])
    except Exception as exc:
        print(f"Error fetching author profile: {exc}", file=sys.stderr)
        sys.exit(1)

    name = author.get("name", "Unknown")
    publications = author.get("publications", [])
    print(f"Author: {name}")
    print(f"Publications found: {len(publications)}")

    bibtex_entries: list[str] = []
    used_keys: set[str] = set()

    for idx, pub in enumerate(publications, start=1):
        raw_title = pub.get("bib", {}).get("title", "Unknown")
        print(f"\n[{idx}/{len(publications)}] {raw_title[:80]}")

        # Fetch full details (abstract, URLs, …)
        print("  Fetching full details (may be slow if Scholar is rate-limiting)...",
              end="", flush=True)
        try:
            pub = scholarly.fill(pub)
            print(" done.")
            time.sleep(args.delay)
        except Exception as exc:
            print(f" failed.")
            print(f"  Warning: could not fetch full details: {exc}")

        key = make_bibtex_key(pub, used_keys)
        bibtex_entries.append(format_bibtex(pub, key))

        arxiv_id = extract_arxiv_id(pub)

        # Decide whether we need a per-paper directory
        need_dir = (not args.no_pdf and bool(_pdf_url_candidates(pub))) or (
            not args.no_source and arxiv_id
        )
        pub_dir: Path | None = None
        if need_dir:
            pub_dir = output_dir / make_folder_name(pub)
            pub_dir.mkdir(parents=True, exist_ok=True)

        if not args.no_pdf and pub_dir:
            download_pdf(pub, pub_dir, force=args.force)

        if not args.no_source and arxiv_id and pub_dir:
            download_arxiv_source(arxiv_id, pub_dir, force=args.force)
            time.sleep(args.delay)

    # Write BibTeX file
    bib_path = output_dir / "publications.bib"
    bib_path.write_text("\n\n".join(bibtex_entries) + "\n", encoding="utf-8")
    print(f"\nDone.  BibTeX file: {bib_path}  ({len(bibtex_entries)} entries)")


if __name__ == "__main__":
    main()
