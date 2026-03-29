"""
Unit tests for fetch_publications.py helper functions.
Run with:  python -m pytest tests/
"""

import tarfile
import io
from pathlib import Path
from unittest.mock import call, patch, MagicMock

import pytest

from fetch_publications import (
    sanitize_filename,
    make_folder_name,
    make_bibtex_key,
    format_bibtex,
    extract_arxiv_id,
    download_pdf,
    download_arxiv_source,
    _is_pdf_response,
    _pdf_url_candidates,
    _to_pdf_url,
    _extract_patent_pdf_url,
    _make_browser_proxy_generator,
    main,
)


# ---------------------------------------------------------------------------
# sanitize_filename
# ---------------------------------------------------------------------------

def test_sanitize_filename_removes_spaces():
    assert " " not in sanitize_filename("hello world")


def test_sanitize_filename_keeps_word_chars():
    assert sanitize_filename("abc_123-xyz") == "abc_123-xyz"


def test_sanitize_filename_replaces_special_chars():
    result = sanitize_filename("foo/bar:baz")
    assert "/" not in result
    assert ":" not in result


# ---------------------------------------------------------------------------
# make_folder_name
# ---------------------------------------------------------------------------

def test_make_folder_name_basic():
    pub = {"bib": {"pub_year": "2023", "title": "Deep Learning for Natural Language Processing"}}
    name = make_folder_name(pub)
    assert name.startswith("2023_")
    assert "Deep" in name


def test_make_folder_name_truncates_title():
    pub = {"bib": {"pub_year": "2021", "title": "A B C D E F G H"}}
    name = make_folder_name(pub)
    # Only first 5 words should be used
    parts = name.split("_")
    # year + up to 5 title words
    assert len(parts) <= 6


def test_make_folder_name_missing_fields():
    name = make_folder_name({})
    assert "unknown" in name


# ---------------------------------------------------------------------------
# make_bibtex_key
# ---------------------------------------------------------------------------

def test_make_bibtex_key_basic():
    pub = {"bib": {"author": "Smith, John", "pub_year": "2020", "title": "A Great Paper"}}
    used: set = set()
    key = make_bibtex_key(pub, used)
    assert "smith" in key
    assert "2020" in key
    assert key in used


def test_make_bibtex_key_unique():
    pub = {"bib": {"author": "Smith, John", "pub_year": "2020", "title": "A Great Paper"}}
    used: set = set()
    key1 = make_bibtex_key(pub, used)
    key2 = make_bibtex_key(pub, used)
    assert key1 != key2
    assert key1 in used
    assert key2 in used


def test_make_bibtex_key_first_last_name_format():
    pub = {"bib": {"author": "Jane Doe", "pub_year": "2019", "title": "Survey"}}
    used: set = set()
    key = make_bibtex_key(pub, used)
    assert "doe" in key


# ---------------------------------------------------------------------------
# extract_arxiv_id
# ---------------------------------------------------------------------------

def test_extract_arxiv_id_from_eprint_abs():
    pub = {"eprint_url": "https://arxiv.org/abs/2301.12345", "pub_url": ""}
    assert extract_arxiv_id(pub) == "2301.12345"


def test_extract_arxiv_id_from_eprint_pdf():
    pub = {"eprint_url": "https://arxiv.org/pdf/2301.12345.pdf", "pub_url": ""}
    assert extract_arxiv_id(pub) == "2301.12345"


def test_extract_arxiv_id_from_pub_url():
    pub = {"eprint_url": "", "pub_url": "https://arxiv.org/abs/1901.00001"}
    assert extract_arxiv_id(pub) == "1901.00001"


def test_extract_arxiv_id_missing():
    pub = {"eprint_url": "", "pub_url": "https://doi.org/10.1000/xyz"}
    assert extract_arxiv_id(pub) == ""


def test_extract_arxiv_id_no_url_keys():
    assert extract_arxiv_id({}) == ""


# ---------------------------------------------------------------------------
# format_bibtex
# ---------------------------------------------------------------------------

def test_format_bibtex_contains_key():
    pub = {"bib": {"title": "My Paper", "author": "Doe, J.", "pub_year": "2022", "abstract": "Some abstract."}}
    entry = format_bibtex(pub, "doe2022my")
    assert "@article{doe2022my," in entry


def test_format_bibtex_includes_abstract():
    pub = {"bib": {"title": "My Paper", "abstract": "This is the abstract."}}
    entry = format_bibtex(pub, "key1")
    assert "abstract" in entry
    assert "This is the abstract." in entry


def test_format_bibtex_conference_uses_inproceedings():
    """venue-string heuristic (PUBLICATION_SEARCH_SNIPPET path)."""
    pub = {"bib": {"title": "Conference Paper", "venue": "International Conference on ML"}}
    entry = format_bibtex(pub, "key1")
    assert entry.startswith("@inproceedings{")


def test_format_bibtex_article_type():
    pub = {"bib": {"title": "Journal Paper", "venue": "Nature"}}
    entry = format_bibtex(pub, "key1")
    assert entry.startswith("@article{")


def test_format_bibtex_bib_journal_field():
    """scholarly fills bib['journal'] for AUTHOR_PUBLICATION_ENTRY; use it as journal field."""
    pub = {"bib": {"title": "Journal Article", "journal": "Nature Communications"}}
    entry = format_bibtex(pub, "key1")
    assert entry.startswith("@article{")
    assert "Nature Communications" in entry
    assert "journal = {Nature Communications}" in entry


def test_format_bibtex_bib_conference_field():
    """scholarly fills bib['conference'] for AUTHOR_PUBLICATION_ENTRY; use it as booktitle."""
    pub = {"bib": {"title": "Conf Paper", "conference": "NeurIPS 2023"}}
    entry = format_bibtex(pub, "key1")
    assert entry.startswith("@inproceedings{")
    assert "NeurIPS 2023" in entry
    assert "booktitle = {NeurIPS 2023}" in entry


def test_format_bibtex_conference_field_preferred_over_venue():
    """bib['conference'] should take precedence over venue string."""
    pub = {"bib": {"title": "Paper", "conference": "ICML 2022", "venue": "ICML"}}
    entry = format_bibtex(pub, "key1")
    assert entry.startswith("@inproceedings{")
    assert "ICML 2022" in entry


def test_format_bibtex_journal_field_preferred_over_venue():
    """bib['journal'] should take precedence over venue string."""
    pub = {"bib": {"title": "Paper", "journal": "Science", "venue": "Sci"}}
    entry = format_bibtex(pub, "key1")
    assert entry.startswith("@article{")
    assert "journal = {Science}" in entry


def test_format_bibtex_arxiv_fields():
    pub = {
        "bib": {"title": "arXiv Paper"},
        "eprint_url": "https://arxiv.org/abs/2105.00001",
    }
    entry = format_bibtex(pub, "key1")
    assert "eprint" in entry
    assert "2105.00001" in entry
    assert "arXiv" in entry


def test_format_bibtex_title_case_protected():
    pub = {"bib": {"title": "Important Title"}}
    entry = format_bibtex(pub, "key1")
    # Title should be wrapped in extra braces to preserve case
    assert "{{Important Title}}" in entry


# ---------------------------------------------------------------------------
# _is_pdf_response
# ---------------------------------------------------------------------------

def test_is_pdf_response_by_content_type():
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Type": "application/pdf"}
    mock_resp.content = b"not real pdf"
    assert _is_pdf_response(mock_resp) is True


def test_is_pdf_response_by_magic_bytes():
    """Accept application/octet-stream when content starts with %PDF magic bytes."""
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Type": "application/octet-stream"}
    mock_resp.content = b"%PDF-1.4 fake content"
    assert _is_pdf_response(mock_resp) is True


def test_is_pdf_response_rejects_html():
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Type": "text/html"}
    mock_resp.content = b"<html></html>"
    assert _is_pdf_response(mock_resp) is False


# ---------------------------------------------------------------------------
# _pdf_url_candidates
# ---------------------------------------------------------------------------

def test_pdf_url_candidates_eprint_url():
    pub = {"eprint_url": "https://example.com/paper.pdf"}
    assert "https://example.com/paper.pdf" in _pdf_url_candidates(pub)


def test_pdf_url_candidates_arxiv_abs_converted():
    pub = {"eprint_url": "https://arxiv.org/abs/2301.12345"}
    cands = _pdf_url_candidates(pub)
    assert any("arxiv.org/pdf/" in c for c in cands)
    assert not any("arxiv.org/abs/" in c for c in cands)


def test_pdf_url_candidates_pub_url_pdf_extension():
    pub = {"eprint_url": "", "pub_url": "https://example.com/paper.pdf"}
    cands = _pdf_url_candidates(pub)
    assert "https://example.com/paper.pdf" in cands


def test_pdf_url_candidates_empty():
    pub = {"eprint_url": "", "pub_url": "https://example.com/page"}
    assert _pdf_url_candidates(pub) == []


# ---------------------------------------------------------------------------
# _to_pdf_url  (bioRxiv / medRxiv conversion)
# ---------------------------------------------------------------------------

def test_to_pdf_url_biorxiv_abstract():
    url = "https://www.biorxiv.org/content/10.1101/2020.01.01.123456"
    pub = {"eprint_url": url, "pub_url": ""}
    urls = _to_pdf_url(url, pub)
    assert urls[0] == url + ".full.pdf"


def test_to_pdf_url_biorxiv_with_version():
    url = "https://www.biorxiv.org/content/10.1101/2020.01.01.123456v3"
    pub = {"eprint_url": url, "pub_url": ""}
    urls = _to_pdf_url(url, pub)
    assert urls[0] == url + ".full.pdf"


def test_to_pdf_url_medrxiv():
    url = "https://www.medrxiv.org/content/10.1101/2021.06.01.12345678"
    pub = {"eprint_url": url, "pub_url": ""}
    urls = _to_pdf_url(url, pub)
    assert urls[0] == url + ".full.pdf"
    # Original URL kept as fallback
    assert url in urls


def test_pdf_url_candidates_biorxiv_eprint_url():
    url = "https://www.biorxiv.org/content/10.1101/2020.01.01.123456v2"
    pub = {"eprint_url": url, "pub_url": ""}
    cands = _pdf_url_candidates(pub)
    assert cands[0] == url + ".full.pdf"
    assert url in cands  # fallback preserved


def test_pdf_url_candidates_biorxiv_pub_url():
    url = "https://www.biorxiv.org/content/10.1101/2020.01.01.123456"
    pub = {"eprint_url": "", "pub_url": url}
    cands = _pdf_url_candidates(pub)
    assert any(".full.pdf" in c for c in cands)


def test_pdf_url_candidates_no_duplicates():
    """Same URL should not appear twice even if eprint_url == pub_url."""
    url = "https://arxiv.org/pdf/2301.00001.pdf"
    pub = {"eprint_url": url, "pub_url": url}
    cands = _pdf_url_candidates(pub)
    assert len(cands) == len(set(cands))


# ---------------------------------------------------------------------------
# _to_pdf_url  (OpenReview / PMLR conversion)
# ---------------------------------------------------------------------------

def test_to_pdf_url_openreview_forum():
    url = "https://openreview.net/forum?id=AbcDefGhIjK"
    pub = {}
    urls = _to_pdf_url(url, pub)
    assert urls[0] == "https://openreview.net/pdf?id=AbcDefGhIjK"
    assert url in urls  # fallback preserved


def test_to_pdf_url_pmlr_html():
    url = "https://proceedings.mlr.press/v235/doe24a.html"
    pub = {}
    urls = _to_pdf_url(url, pub)
    assert urls[0] == "https://proceedings.mlr.press/v235/doe24a.pdf"
    assert url in urls  # fallback preserved


def test_to_pdf_url_diva_record():
    url = "https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1638041&dswid=8847"
    pub = {}
    urls = _to_pdf_url(url, pub)
    assert urls[0] == "https://www.diva-portal.org/smash/get/diva2:1638041/FULLTEXT01.pdf"
    assert url in urls  # fallback preserved


def test_to_pdf_url_diva_record_unencoded_pid():
    """DiVA pid without URL-encoding should still work."""
    url = "https://www.diva-portal.org/smash/record.jsf?pid=diva2:9999999"
    pub = {}
    urls = _to_pdf_url(url, pub)
    assert urls[0] == "https://www.diva-portal.org/smash/get/diva2:9999999/FULLTEXT01.pdf"


def test_pdf_url_candidates_diva_pub_url():
    url = "https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1638041&dswid=8847"
    pub = {"eprint_url": "", "pub_url": url}
    cands = _pdf_url_candidates(pub)
    assert any("FULLTEXT01.pdf" in c for c in cands)


def test_download_pdf_diva_uses_fulltext_url(tmp_path):
    """DiVA record URL should be converted to FULLTEXT01.pdf URL."""
    record_url = "https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1638041&dswid=8847"
    pub = {"eprint_url": record_url, "bib": {}}

    expected_pdf_url = "https://www.diva-portal.org/smash/get/diva2:1638041/FULLTEXT01.pdf"
    captured = []

    def fake_get(url, **kwargs):
        captured.append(url)
        mock = MagicMock()
        mock.status_code = 200
        mock.headers = {"Content-Type": "application/pdf"}
        mock.content = b"%PDF-1.4 diva pdf"
        return mock

    with patch("fetch_publications.requests.get", side_effect=fake_get):
        result = download_pdf(pub, tmp_path)

    assert result is True
    assert captured[0] == expected_pdf_url
    assert (tmp_path / "paper.pdf").exists()


def test_pdf_url_candidates_openreview_pub_url():
    url = "https://openreview.net/forum?id=SomeId123"
    pub = {"eprint_url": "", "pub_url": url}
    cands = _pdf_url_candidates(pub)
    assert any("openreview.net/pdf" in c for c in cands)


def test_pdf_url_candidates_pmlr_pub_url():
    url = "https://proceedings.mlr.press/v100/smith20a.html"
    pub = {"eprint_url": "", "pub_url": url}
    cands = _pdf_url_candidates(pub)
    assert any(c.endswith(".pdf") for c in cands)


# ---------------------------------------------------------------------------
# _pdf_url_candidates  – public_access fallback
# ---------------------------------------------------------------------------

def test_pdf_url_candidates_public_access_uses_pub_url():
    """When public_access=True and eprint_url is absent, pub_url should be tried."""
    pub_url = "https://some-repo.example.com/papers/paper123"
    pub = {"eprint_url": "", "pub_url": pub_url, "public_access": True}
    cands = _pdf_url_candidates(pub)
    assert pub_url in cands


def test_pdf_url_candidates_no_public_access_generic_url_excluded():
    """Generic pub_url without public_access flag should NOT be added."""
    pub_url = "https://some-repo.example.com/papers/paper123"
    pub = {"eprint_url": "", "pub_url": pub_url, "public_access": False}
    cands = _pdf_url_candidates(pub)
    assert pub_url not in cands


def test_download_pdf_public_access_tries_pub_url(tmp_path):
    """When public_access=True, pub_url should be tried even without eprint_url."""
    pub_url = "https://some-repo.example.com/paper"
    pub = {"eprint_url": "", "pub_url": pub_url, "public_access": True, "bib": {}}

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/pdf"}
    mock_response.content = b"%PDF-1.4 open access pdf"

    captured = []

    def fake_get(url, **kwargs):
        captured.append(url)
        return mock_response

    with patch("fetch_publications.requests.get", side_effect=fake_get):
        result = download_pdf(pub, tmp_path)

    assert result is True
    assert pub_url in captured
    assert (tmp_path / "paper.pdf").exists()


def test_download_pdf_biorxiv_uses_full_pdf_url(tmp_path):
    """bioRxiv eprint_url should be converted to .full.pdf URL."""
    abstract_url = "https://www.biorxiv.org/content/10.1101/2020.01.01.123456"
    pub = {"eprint_url": abstract_url, "bib": {}}

    pdf_url = abstract_url + ".full.pdf"
    captured = []

    def fake_get(url, **kwargs):
        captured.append(url)
        mock = MagicMock()
        mock.status_code = 200
        mock.headers = {"Content-Type": "application/pdf"}
        mock.content = b"%PDF-1.4 biorxiv"
        return mock

    with patch("fetch_publications.requests.get", side_effect=fake_get):
        result = download_pdf(pub, tmp_path)

    assert result is True
    assert captured[0] == pdf_url
    assert (tmp_path / "paper.pdf").exists()


def test_download_pdf_biorxiv_falls_back_when_full_pdf_missing(tmp_path):
    """When .full.pdf returns non-PDF, fall back to the original URL."""
    abstract_url = "https://www.biorxiv.org/content/10.1101/2020.01.01.111111"
    pub = {"eprint_url": abstract_url, "bib": {}}

    def fake_get(url, **kwargs):
        mock = MagicMock()
        if url.endswith(".full.pdf"):
            # First attempt – HTML (abstract page served instead of PDF)
            mock.status_code = 200
            mock.headers = {"Content-Type": "text/html"}
            mock.content = b"<html>abstract</html>"
        else:
            # Fallback – redirect to actual PDF
            mock.status_code = 200
            mock.headers = {"Content-Type": "application/pdf"}
            mock.content = b"%PDF-1.4 actual"
        return mock

    with patch("fetch_publications.requests.get", side_effect=fake_get):
        result = download_pdf(pub, tmp_path)

    assert result is True
    assert (tmp_path / "paper.pdf").exists()


def test_download_pdf_no_candidates(tmp_path):
    pub = {"bib": {}, "eprint_url": "", "pub_url": "https://example.com/abstract"}
    result = download_pdf(pub, tmp_path)
    assert result is False
    assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# download_pdf – skip when already downloaded
# ---------------------------------------------------------------------------

def test_download_pdf_skips_existing(tmp_path):
    """download_pdf returns True immediately when paper.pdf already exists."""
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 existing")
    pub = {"eprint_url": "https://example.com/paper.pdf", "bib": {}}

    with patch("fetch_publications.requests.get") as mock_get:
        result = download_pdf(pub, tmp_path)

    assert result is True
    mock_get.assert_not_called()
    # Existing file should be unchanged
    assert pdf_path.read_bytes() == b"%PDF-1.4 existing"


def test_download_pdf_force_overwrites_existing(tmp_path):
    """download_pdf re-downloads when force=True even if paper.pdf exists."""
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 old content")
    pub = {"eprint_url": "https://example.com/paper.pdf", "bib": {}}

    new_content = b"%PDF-1.4 new content"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/pdf"}
    mock_response.content = new_content

    with patch("fetch_publications.requests.get", return_value=mock_response):
        result = download_pdf(pub, tmp_path, force=True)

    assert result is True
    assert pdf_path.read_bytes() == new_content



def test_download_pdf_success(tmp_path):
    pub = {"eprint_url": "https://example.com/paper.pdf", "bib": {}}
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/pdf"}
    mock_response.content = b"%PDF-1.4 fake pdf content"

    with patch("fetch_publications.requests.get", return_value=mock_response):
        result = download_pdf(pub, tmp_path)

    assert result is True
    assert (tmp_path / "paper.pdf").exists()


def test_download_pdf_octet_stream_with_magic_bytes(tmp_path):
    """PDF served with application/octet-stream should still be saved."""
    pub = {"eprint_url": "https://example.com/paper", "bib": {}}
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/octet-stream"}
    mock_response.content = b"%PDF-1.4 actual pdf"

    with patch("fetch_publications.requests.get", return_value=mock_response):
        result = download_pdf(pub, tmp_path)

    assert result is True
    assert (tmp_path / "paper.pdf").exists()


def test_download_pdf_wrong_content_type(tmp_path):
    pub = {"eprint_url": "https://example.com/paper", "bib": {}}
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.content = b"<html></html>"

    with patch("fetch_publications.requests.get", return_value=mock_response):
        result = download_pdf(pub, tmp_path)

    assert result is False


def test_download_pdf_converts_arxiv_abs_to_pdf_url(tmp_path):
    pub = {"eprint_url": "https://arxiv.org/abs/2105.00001", "bib": {}}
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/pdf"}
    mock_response.content = b"%PDF fake"

    captured_url = []

    def fake_get(url, **kwargs):
        captured_url.append(url)
        return mock_response

    with patch("fetch_publications.requests.get", side_effect=fake_get):
        download_pdf(pub, tmp_path)

    assert "arxiv.org/pdf/" in captured_url[0]
    assert "abs" not in captured_url[0]


def test_download_pdf_fallback_to_pub_url(tmp_path):
    """When eprint_url is absent, fall back to pub_url if it ends with .pdf."""
    pub = {"eprint_url": "", "pub_url": "https://example.com/paper.pdf", "bib": {}}
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/pdf"}
    mock_response.content = b"%PDF-1.4 fake"

    with patch("fetch_publications.requests.get", return_value=mock_response):
        result = download_pdf(pub, tmp_path)

    assert result is True
    assert (tmp_path / "paper.pdf").exists()


# ---------------------------------------------------------------------------
# download_arxiv_source (mocked)
# ---------------------------------------------------------------------------

def _make_tar_gz_bytes(files: dict) -> bytes:
    """Create an in-memory tar.gz archive from a dict of {filename: content}."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            data = content.encode() if isinstance(content, str) else content
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_download_arxiv_source_empty_id(tmp_path):
    result = download_arxiv_source("", tmp_path)
    assert result is False


def test_download_arxiv_source_tar_extraction(tmp_path):
    tar_bytes = _make_tar_gz_bytes({"main.tex": r"\documentclass{article}\begin{document}Hello\end{document}"})
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/x-tar"}
    mock_response.content = tar_bytes

    with patch("fetch_publications.requests.get", return_value=mock_response):
        result = download_arxiv_source("2105.00001", tmp_path)

    assert result is True
    assert (tmp_path / "main.tex").exists()


def test_download_arxiv_source_plain_tex(tmp_path):
    tex_content = b"\\documentclass{article}\\begin{document}Hello\\end{document}"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/x-tex"}
    mock_response.content = tex_content

    with patch("fetch_publications.requests.get", return_value=mock_response):
        result = download_arxiv_source("2105.00002", tmp_path)

    assert result is True
    assert (tmp_path / "source.tex").exists()


def test_download_arxiv_source_404(tmp_path):
    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch("fetch_publications.requests.get", return_value=mock_response):
        result = download_arxiv_source("9999.99999", tmp_path)

    assert result is False


def test_download_arxiv_source_rejects_path_traversal(tmp_path):
    """Tar members with path traversal should be skipped."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        evil_content = b"evil"
        info = tarfile.TarInfo(name="../evil.tex")
        info.size = len(evil_content)
        tar.addfile(info, io.BytesIO(evil_content))
        safe_content = b"safe"
        info2 = tarfile.TarInfo(name="safe.tex")
        info2.size = len(safe_content)
        tar.addfile(info2, io.BytesIO(safe_content))

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = buf.getvalue()

    with patch("fetch_publications.requests.get", return_value=mock_response):
        result = download_arxiv_source("2105.00003", tmp_path)

    assert result is True
    assert not (tmp_path.parent / "evil.tex").exists()
    assert (tmp_path / "safe.tex").exists()


# ---------------------------------------------------------------------------
# download_arxiv_source – skip when already downloaded
# ---------------------------------------------------------------------------

def test_download_arxiv_source_skips_existing(tmp_path):
    """download_arxiv_source returns True immediately when source files exist."""
    (tmp_path / "main.tex").write_text("existing source")
    with patch("fetch_publications.requests.get") as mock_get:
        result = download_arxiv_source("2105.00001", tmp_path)

    assert result is True
    mock_get.assert_not_called()


def test_download_arxiv_source_skips_when_source_tex_exists(tmp_path):
    """download_arxiv_source skips when source.tex from a previous run exists."""
    (tmp_path / "source.tex").write_text("\\documentclass{article}")
    with patch("fetch_publications.requests.get") as mock_get:
        result = download_arxiv_source("2105.00002", tmp_path)

    assert result is True
    mock_get.assert_not_called()


def test_download_arxiv_source_does_not_skip_when_only_pdf_present(tmp_path):
    """paper.pdf alone should not prevent a fresh source download."""
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4 existing")
    tex_content = b"\\documentclass{article}"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/x-tex"}
    mock_response.content = tex_content

    with patch("fetch_publications.requests.get", return_value=mock_response):
        result = download_arxiv_source("2105.00003", tmp_path)

    assert result is True
    assert (tmp_path / "source.tex").exists()


def test_download_arxiv_source_force_redownloads(tmp_path):
    """download_arxiv_source re-downloads when force=True even if source exists."""
    (tmp_path / "source.tex").write_text("old source")
    new_content = b"\\documentclass{article} new"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/x-tex"}
    mock_response.content = new_content

    with patch("fetch_publications.requests.get", return_value=mock_response):
        result = download_arxiv_source("2105.00004", tmp_path, force=True)

    assert result is True
    assert (tmp_path / "source.tex").read_bytes() == new_content


# ---------------------------------------------------------------------------
# Google Patents support
# ---------------------------------------------------------------------------

_SAMPLE_PATENT_HTML = """
<html><body>
<a href="https://patentimages.storage.googleapis.com/97/b7/3e/0c8d65e3a3a6da/US10046177.pdf"
   download="" title="Download PDF">Download PDF</a>
</body></html>
"""

_PATENT_PDF_URL = (
    "https://patentimages.storage.googleapis.com/97/b7/3e/0c8d65e3a3a6da/US10046177.pdf"
)


def test_extract_patent_pdf_url_found():
    """Should extract patentimages PDF URL from a Google Patents HTML page."""
    assert _extract_patent_pdf_url(_SAMPLE_PATENT_HTML) == _PATENT_PDF_URL


def test_extract_patent_pdf_url_not_found():
    """Should return empty string when no patentimages URL is present."""
    assert _extract_patent_pdf_url("<html><body>No PDF here</body></html>") == ""


def test_pdf_url_candidates_includes_patent_pub_url():
    """patents.google.com URL should be included in candidates as a known pattern."""
    patent_url = "https://patents.google.com/patent/US10046177B2/en"
    pub = {"eprint_url": "", "pub_url": patent_url}
    cands = _pdf_url_candidates(pub)
    assert patent_url in cands


def test_pdf_url_candidates_patent_eprint_url():
    """Patent URL in eprint_url should also be included in candidates."""
    patent_url = "https://patents.google.com/patent/US10046177B2/en"
    pub = {"eprint_url": patent_url, "pub_url": ""}
    cands = _pdf_url_candidates(pub)
    assert patent_url in cands


def test_to_pdf_url_patent_returns_url_unchanged():
    """_to_pdf_url should return the patent page URL as-is (scraping happens later)."""
    patent_url = "https://patents.google.com/patent/US10046177B2/en"
    urls = _to_pdf_url(patent_url, {})
    assert patent_url in urls


def test_download_pdf_patent_scrapes_and_downloads(tmp_path):
    """download_pdf should fetch the patent page, extract the PDF link, and save it."""
    patent_url = "https://patents.google.com/patent/US10046177B2/en"
    pub = {"eprint_url": "", "pub_url": patent_url, "bib": {}}

    call_log = []

    def fake_get(url, **kwargs):
        call_log.append(url)
        mock = MagicMock()
        if _PATENT_PDF_URL in url:
            mock.status_code = 200
            mock.headers = {"Content-Type": "application/pdf"}
            mock.content = b"%PDF-1.4 patent pdf"
        else:
            # Return the patent HTML page
            mock.status_code = 200
            mock.headers = {"Content-Type": "text/html"}
            mock.content = _SAMPLE_PATENT_HTML.encode()
            mock.text = _SAMPLE_PATENT_HTML
        return mock

    with patch("fetch_publications.requests.get", side_effect=fake_get):
        result = download_pdf(pub, tmp_path)

    assert result is True
    assert (tmp_path / "paper.pdf").exists()
    # First request should be the patent page, second the actual PDF
    assert any(u.startswith("https://patents.google.com/") for u in call_log)
    assert any(u.startswith("https://patentimages.storage.googleapis.com/") for u in call_log)


def test_download_pdf_patent_no_pdf_link_in_page(tmp_path):
    """When no PDF link is found in the patent page, download_pdf returns False."""
    patent_url = "https://patents.google.com/patent/US10046177B2/en"
    pub = {"eprint_url": "", "pub_url": patent_url, "bib": {}}

    def fake_get(url, **kwargs):
        mock = MagicMock()
        mock.status_code = 200
        mock.headers = {"Content-Type": "text/html"}
        mock.content = b"<html><body>No PDF link here</body></html>"
        mock.text = "<html><body>No PDF link here</body></html>"
        return mock

    with patch("fetch_publications.requests.get", side_effect=fake_get):
        result = download_pdf(pub, tmp_path)

    assert result is False


# ---------------------------------------------------------------------------
# main() – phased execution tests
# ---------------------------------------------------------------------------

def _make_fake_pub(title: str, year: str = "2023", arxiv_id: str = "") -> dict:
    """Return a minimal publication dict as returned by scholarly."""
    eprint_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
    return {
        "bib": {"title": title, "author": "Doe, J.", "pub_year": year},
        "eprint_url": eprint_url,
        "pub_url": eprint_url,
        "public_access": False,
    }


def _make_mock_scholarly(pubs: list[dict]):
    """Return a mock for `scholarly.scholarly` whose fill() returns the pub unchanged."""
    mock_scholarly = MagicMock()
    mock_scholarly.search_author_id.return_value = {"name": "Test Author", "publications": pubs}
    # fill() for the author simply returns the author dict with publications intact;
    # fill() for a publication returns the pub unchanged.
    mock_scholarly.fill.side_effect = lambda obj, **kwargs: obj
    return mock_scholarly


def test_main_bibtex_written_after_each_pub(tmp_path):
    """BibTeX file must be written after each publication, not only at the end."""
    pubs = [
        _make_fake_pub("First Paper"),
        _make_fake_pub("Second Paper"),
    ]

    bib_path = tmp_path / "publications.bib"
    write_calls: list[int] = []

    original_write_text = Path.write_text

    def tracking_write_text(self, data, **kwargs):
        if self == bib_path:
            # Count how many entries have been written so far
            write_calls.append(data.count("@article"))
        original_write_text(self, data, **kwargs)

    with (
        patch("fetch_publications.scholarly", _make_mock_scholarly(pubs)),
        patch("fetch_publications.time.sleep"),
        patch.object(Path, "write_text", tracking_write_text),
        patch("sys.argv", ["fetch_publications.py", "https://scholar.google.com/citations?user=TEST", "--output-dir", str(tmp_path), "--no-pdf", "--no-source"]),
    ):
        main()

    # The file should have been written at least twice: once per publication
    assert len(write_calls) >= 2
    # After the first write there should be 1 entry; after the second, 2
    assert write_calls[0] == 1
    assert write_calls[1] == 2


def test_main_bibtex_contains_all_entries(tmp_path):
    """BibTeX file must contain an entry for every publication."""
    pubs = [
        _make_fake_pub("Alpha Paper", year="2021"),
        _make_fake_pub("Beta Paper", year="2022"),
        _make_fake_pub("Gamma Paper", year="2023"),
    ]

    with (
        patch("fetch_publications.scholarly", _make_mock_scholarly(pubs)),
        patch("fetch_publications.time.sleep"),
        patch("sys.argv", ["fetch_publications.py", "https://scholar.google.com/citations?user=TEST", "--output-dir", str(tmp_path), "--no-pdf", "--no-source"]),
    ):
        main()

    bib_text = (tmp_path / "publications.bib").read_text(encoding="utf-8")
    assert "Alpha Paper" in bib_text
    assert "Beta Paper" in bib_text
    assert "Gamma Paper" in bib_text


def test_main_pdf_downloaded_after_all_bibtex(tmp_path):
    """All BibTeX entries must be written before any PDF download begins."""
    arxiv_pub = _make_fake_pub("arXiv Paper", arxiv_id="2301.00001")
    pubs = [arxiv_pub]

    event_log: list[str] = []

    original_write_text = Path.write_text

    def tracking_write_text(self, data, **kwargs):
        if self.name == "publications.bib":
            event_log.append("bib_write")
        original_write_text(self, data, **kwargs)

    def tracking_download_pdf(pub, pub_dir, force=False):
        event_log.append("pdf_download")
        pub_dir.mkdir(parents=True, exist_ok=True)
        (pub_dir / "paper.pdf").write_bytes(b"%PDF-1.4 fake")
        return True

    with (
        patch("fetch_publications.scholarly", _make_mock_scholarly(pubs)),
        patch("fetch_publications.time.sleep"),
        patch("fetch_publications.download_pdf", side_effect=tracking_download_pdf),
        patch("fetch_publications.download_arxiv_source", return_value=False),
        patch.object(Path, "write_text", tracking_write_text),
        patch("sys.argv", ["fetch_publications.py", "https://scholar.google.com/citations?user=TEST", "--output-dir", str(tmp_path)]),
    ):
        main()

    # BibTeX write must come before any PDF download
    bib_idx = next(i for i, e in enumerate(event_log) if e == "bib_write")
    pdf_idx = next((i for i, e in enumerate(event_log) if e == "pdf_download"), None)
    if pdf_idx is not None:
        assert bib_idx < pdf_idx


def test_main_latex_downloaded_after_pdfs(tmp_path):
    """LaTeX source downloads must happen after all PDF downloads."""
    arxiv_pub = _make_fake_pub("arXiv Paper", arxiv_id="2301.00001")
    pubs = [arxiv_pub]

    event_log: list[str] = []

    def tracking_download_pdf(pub, pub_dir, force=False):
        event_log.append("pdf_download")
        pub_dir.mkdir(parents=True, exist_ok=True)
        (pub_dir / "paper.pdf").write_bytes(b"%PDF-1.4 fake")
        return True

    def tracking_download_source(arxiv_id, pub_dir, force=False):
        event_log.append("source_download")
        return True

    with (
        patch("fetch_publications.scholarly", _make_mock_scholarly(pubs)),
        patch("fetch_publications.time.sleep"),
        patch("fetch_publications.download_pdf", side_effect=tracking_download_pdf),
        patch("fetch_publications.download_arxiv_source", side_effect=tracking_download_source),
        patch("sys.argv", ["fetch_publications.py", "https://scholar.google.com/citations?user=TEST", "--output-dir", str(tmp_path)]),
    ):
        main()

    assert "pdf_download" in event_log
    assert "source_download" in event_log
    pdf_idx = event_log.index("pdf_download")
    src_idx = event_log.index("source_download")
    assert pdf_idx < src_idx


def test_main_bibtex_key_uses_author_after_fill(tmp_path):
    """BibTeX keys must use the first author's last name after scholarly.fill() enriches
    the publication data.  The initial Phase-1 snippet often lacks the author field,
    causing keys like 'unknown2024neural'.  Phase 2 must regenerate the key once
    the full author info is available.
    """
    # Simulate a pub whose initial snippet has NO author field (as Scholar returns it)
    # and whose filled version adds the author.
    pub_snippet = {"bib": {"title": "Neural Networks Today", "pub_year": "2024"},
                   "eprint_url": "", "pub_url": "", "public_access": False}
    pub_filled = {"bib": {"title": "Neural Networks Today", "pub_year": "2024",
                          "author": "Smith, Alice"},
                  "eprint_url": "", "pub_url": "", "public_access": False}

    def fill_side_effect(obj, **kwargs):
        if "sections" in kwargs:
            return obj  # author-profile fill – pass through
        return pub_filled

    mock_scholarly = MagicMock()
    mock_scholarly.search_author_id.return_value = {
        "name": "Alice Smith",
        "publications": [pub_snippet],
    }
    mock_scholarly.fill.side_effect = fill_side_effect

    with (
        patch("fetch_publications.scholarly", mock_scholarly),
        patch("fetch_publications.time.sleep"),
        patch("sys.argv", ["fetch_publications.py", "https://scholar.google.com/citations?user=TEST",
                           "--output-dir", str(tmp_path), "--no-pdf", "--no-source"]),
    ):
        main()

    bib_text = (tmp_path / "publications.bib").read_text(encoding="utf-8")
    # Extract the BibTeX cite-key with a regex: @article{KEY, …} or @misc{KEY, …}
    import re as _re
    key_match = _re.search(r"@\w+\{([^,\s]+)", bib_text)
    assert key_match, f"No BibTeX entry found in:\n{bib_text}"
    cite_key = key_match.group(1).lower()
    assert "smith" in cite_key, (
        f"Expected 'smith' in BibTeX key '{cite_key}', got:\n{bib_text}"
    )
    assert "unknown" not in cite_key, (
        f"BibTeX key must not contain 'unknown'; got key '{cite_key}' in:\n{bib_text}"
    )


def test_main_partial_bibtex_preserved_on_scholar_failure(tmp_path):
    """If scholarly.fill raises on the second publication, the first entry must still be in the BibTeX file."""
    pub1 = _make_fake_pub("Good Paper", year="2021")
    pub2 = _make_fake_pub("Bad Paper", year="2022")

    fill_call_count = 0

    def fill_side_effect(obj, **kwargs):
        nonlocal fill_call_count
        # The author fill call (with sections kwarg) passes through unchanged
        if "sections" in kwargs:
            return obj
        fill_call_count += 1
        if fill_call_count == 2:
            raise RuntimeError("Simulated Scholar rate-limit error")
        return obj

    mock_scholarly = MagicMock()
    mock_scholarly.search_author_id.return_value = {
        "name": "Test Author",
        "publications": [pub1, pub2],
    }
    mock_scholarly.fill.side_effect = fill_side_effect

    with (
        patch("fetch_publications.scholarly", mock_scholarly),
        patch("fetch_publications.time.sleep"),
        patch("sys.argv", ["fetch_publications.py", "https://scholar.google.com/citations?user=TEST", "--output-dir", str(tmp_path), "--no-pdf", "--no-source"]),
    ):
        main()

    bib_text = (tmp_path / "publications.bib").read_text(encoding="utf-8")
    assert "Good Paper" in bib_text
    # Bad Paper entry should still be present (created with partial data)
    assert "Bad Paper" in bib_text


# ---------------------------------------------------------------------------
# _make_browser_proxy_generator / --browser flag tests
# ---------------------------------------------------------------------------

def test_make_browser_proxy_generator_chrome_not_headless():
    """The Chrome webdriver opened by _make_browser_proxy_generator must NOT use
    --headless so the user can see (and solve) the CAPTCHA.
    """
    pg = _make_browser_proxy_generator()

    chrome_options_seen: list = []
    mock_driver = MagicMock()
    mock_driver.get = MagicMock()

    def fake_chrome(options=None, **kwargs):
        chrome_options_seen.append(options)
        return mock_driver

    with patch("selenium.webdriver.Chrome", side_effect=fake_chrome):
        pg._get_chrome_webdriver()

    assert chrome_options_seen, "Chrome was never instantiated"
    args = chrome_options_seen[0].arguments if chrome_options_seen[0] else []
    assert "--headless" not in args, (
        f"Chrome must NOT be headless so the user can solve the CAPTCHA. "
        f"Got options.arguments={args}"
    )


def test_make_browser_proxy_generator_firefox_not_headless():
    """The Firefox webdriver opened by _make_browser_proxy_generator must NOT use
    --headless so the user can see (and solve) the CAPTCHA.
    """
    pg = _make_browser_proxy_generator()

    firefox_options_seen: list = []
    mock_driver = MagicMock()
    mock_driver.get = MagicMock()

    def fake_firefox(options=None, **kwargs):
        firefox_options_seen.append(options)
        return mock_driver

    with patch("selenium.webdriver.Firefox", side_effect=fake_firefox):
        pg._get_firefox_webdriver()

    assert firefox_options_seen, "Firefox was never instantiated"
    args = firefox_options_seen[0].arguments if firefox_options_seen[0] else []
    assert "--headless" not in args, (
        f"Firefox must NOT be headless so the user can solve the CAPTCHA. "
        f"Got options.arguments={args}"
    )


def test_handle_captcha2_fixed_copies_cookies_without_secure_kwarg():
    """After solving a CAPTCHA, the patched _handle_captcha2 must copy browser
    cookies into the httpx session using only (name, value, domain, path).

    scholarly 1.7.x uses httpx.Client internally.  httpx.Cookies.set() only
    accepts those four parameters; passing 'secure' (or any other
    browser-specific key) raises TypeError.  That crash causes scholarly to
    discard the session and open a fresh one, immediately triggering another
    CAPTCHA.  This test verifies that our override avoids that crash.
    """
    import types as _types
    from unittest.mock import MagicMock, call

    pg = _make_browser_proxy_generator()

    # --- fake webdriver that looks solved from the start ---
    mock_driver = MagicMock()
    mock_driver.current_url = "https://scholar.google.com/scholar"
    mock_driver.get_cookies.return_value = [
        # A typical Selenium cookie dict: includes 'secure', 'httpOnly', etc.
        {
            "name": "GSP",
            "value": "abc123",
            "domain": ".google.com",
            "path": "/",
            "secure": True,
            "httpOnly": False,
            "sameSite": "None",
            "expiry": 9999999999,
        },
        {
            "name": "NID",
            "value": "xyz",
            "domain": ".google.com",
            "path": "/",
            "secure": False,
            "httpOnly": True,
        },
    ]
    pg._webdriver = mock_driver
    pg._get_webdriver = lambda: mock_driver

    # _webdriver_has_captcha() returns False → CAPTCHA is already solved
    pg._webdriver_has_captcha = MagicMock(return_value=False)

    # --- httpx session with a trackable Cookies.set() ---
    mock_cookies = MagicMock()
    mock_session = MagicMock()
    mock_session.cookies = mock_cookies
    mock_session.cookies.__iter__ = MagicMock(return_value=iter([]))  # no existing cookies
    pg._session = mock_session

    # Run: must NOT raise TypeError
    pg._handle_captcha2("https://scholar.google.com/scholar?q=test")

    # Cookies must be set using ONLY name/value/domain/path
    expected_calls = [
        call(name="GSP", value="abc123", domain=".google.com", path="/"),
        call(name="NID", value="xyz", domain=".google.com", path="/"),
    ]
    mock_cookies.set.assert_has_calls(expected_calls, any_order=False)

    # Confirm 'secure' was NEVER passed as a keyword argument
    for actual_call in mock_cookies.set.call_args_list:
        assert "secure" not in actual_call.kwargs, (
            f"'secure' must not be passed to httpx.Cookies.set(). "
            f"Got kwargs={actual_call.kwargs!r}"
        )


def test_browser_flag_calls_scholarly_use_proxy(tmp_path):
    """When --browser is passed, scholarly.use_proxy must be called so that
    scholarly is configured with a visible-browser CAPTCHA handler.
    """
    pubs = [_make_fake_pub("Paper 1")]
    mock_scholarly = _make_mock_scholarly(pubs)

    mock_pg = MagicMock()
    mock_pg_class = MagicMock(return_value=mock_pg)

    with (
        patch("fetch_publications.scholarly", mock_scholarly),
        patch("fetch_publications._make_browser_proxy_generator", return_value=mock_pg),
        patch("fetch_publications.time.sleep"),
        patch("sys.argv", [
            "fetch_publications.py",
            "https://scholar.google.com/citations?user=TEST",
            "--output-dir", str(tmp_path),
            "--no-pdf", "--no-source",
            "--browser",
        ]),
    ):
        main()

    mock_scholarly.use_proxy.assert_called_once(), (
        "scholarly.use_proxy must be called once to configure the browser-based "
        "CAPTCHA handler when --browser is set"
    )


def test_browser_flag_uses_single_proxy_generator(tmp_path):
    """Both arguments to scholarly.use_proxy must be the SAME ProxyGenerator
    instance so that one solved CAPTCHA covers all Scholar requests (pm1 and pm2
    share the same httpx session and cookies).
    """
    pubs = [_make_fake_pub("Paper 1")]
    mock_scholarly = _make_mock_scholarly(pubs)

    mock_pg = MagicMock()

    with (
        patch("fetch_publications.scholarly", mock_scholarly),
        patch("fetch_publications._make_browser_proxy_generator", return_value=mock_pg),
        patch("fetch_publications.time.sleep"),
        patch("sys.argv", [
            "fetch_publications.py",
            "https://scholar.google.com/citations?user=TEST",
            "--output-dir", str(tmp_path),
            "--no-pdf", "--no-source",
            "--browser",
        ]),
    ):
        main()

    args, _ = mock_scholarly.use_proxy.call_args
    assert len(args) == 2, "use_proxy must be called with two positional arguments"
    assert args[0] is args[1], (
        "Both arguments to scholarly.use_proxy must be the SAME ProxyGenerator "
        "instance so that one solved CAPTCHA covers pm1 and pm2. "
        f"Got args[0]={args[0]!r}, args[1]={args[1]!r}"
    )


def test_scholarly_logging_enabled_always(tmp_path, capsys):
    """The scholarly logger must have at least one handler attached so that
    internal messages (CAPTCHA requests, retry waits, …) reach the user
    regardless of whether --browser is set.
    """
    import logging

    pubs = [_make_fake_pub("Paper 1")]
    mock_scholarly = _make_mock_scholarly(pubs)

    with (
        patch("fetch_publications.scholarly", mock_scholarly),
        patch("fetch_publications.time.sleep"),
        patch("sys.argv", [
            "fetch_publications.py",
            "https://scholar.google.com/citations?user=TEST",
            "--output-dir", str(tmp_path),
            "--no-pdf", "--no-source",
        ]),
    ):
        main()

    scholarly_logger = logging.getLogger("scholarly")
    assert scholarly_logger.handlers, (
        "The 'scholarly' logger must have at least one handler so its INFO-level "
        "messages (CAPTCHA requests, retry waits, …) are visible to the user."
    )
    assert scholarly_logger.level <= logging.INFO, (
        f"The 'scholarly' logger level must be INFO or lower. "
        f"Got {logging.getLevelName(scholarly_logger.level)}"
    )


def test_error_without_browser_flag_suggests_browser(tmp_path, capsys):
    """When the author-profile fetch fails and --browser was not set, the error
    output must suggest running again with --browser so the user knows about the
    CAPTCHA-solving fallback.
    """
    mock_scholarly = MagicMock()
    mock_scholarly.search_author_id.side_effect = RuntimeError("MaxTriesExceeded")

    with (
        patch("fetch_publications.scholarly", mock_scholarly),
        patch("sys.argv", [
            "fetch_publications.py",
            "https://scholar.google.com/citations?user=TEST",
            "--output-dir", str(tmp_path),
            "--no-pdf", "--no-source",
        ]),
        pytest.raises(SystemExit),
    ):
        main()

    captured = capsys.readouterr()
    assert "--browser" in captured.out, (
        "The error message should suggest --browser as a fallback when the initial "
        f"fetch fails. Got stdout={captured.out!r}"
    )


def test_main_author_search_error_visible_on_stdout(tmp_path, capsys):
    """When scholarly.search_author_id() raises, the error must appear on stdout.

    Previously the error was only sent to stderr, causing it to be invisible
    when the user does not capture stderr — the script appeared to hang after
    printing 'Author ID: …' with no further output.
    """
    mock_scholarly = MagicMock()
    mock_scholarly.search_author_id.side_effect = RuntimeError("Scholar rate limit hit")

    with (
        patch("fetch_publications.scholarly", mock_scholarly),
        patch("sys.argv", ["fetch_publications.py", "https://scholar.google.com/citations?user=TEST", "--output-dir", str(tmp_path), "--no-pdf", "--no-source"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "Scholar rate limit hit" in captured.out, (
        "Error message must be on stdout so the user can see it. "
        f"Got stdout={captured.out!r}, stderr={captured.err!r}"
    )


def test_main_author_fill_error_visible_on_stdout(tmp_path, capsys):
    """When scholarly.fill() raises during the author-profile fetch, the error
    must appear on stdout (not only on stderr).
    """
    mock_scholarly = MagicMock()
    mock_scholarly.search_author_id.return_value = {"name": "Test Author", "publications": []}
    mock_scholarly.fill.side_effect = RuntimeError("MaxTriesExceeded: Google Scholar blocked")

    with (
        patch("fetch_publications.scholarly", mock_scholarly),
        patch("sys.argv", ["fetch_publications.py", "https://scholar.google.com/citations?user=TEST", "--output-dir", str(tmp_path), "--no-pdf", "--no-source"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "MaxTriesExceeded" in captured.out, (
        "Error message must be on stdout so the user can see it. "
        f"Got stdout={captured.out!r}, stderr={captured.err!r}"
    )


def test_main_empty_publications_warns_user(tmp_path, capsys):
    """When scholarly returns zero publications (e.g. because the profile is
    blocked), the script should print a clear warning instead of silently
    producing an empty bib file.
    """
    mock_scholarly = MagicMock()
    mock_scholarly.search_author_id.return_value = {"name": "Test Author", "publications": []}
    mock_scholarly.fill.side_effect = lambda obj, **kwargs: obj

    with (
        patch("fetch_publications.scholarly", mock_scholarly),
        patch("fetch_publications.time.sleep"),
        patch("sys.argv", ["fetch_publications.py", "https://scholar.google.com/citations?user=TEST", "--output-dir", str(tmp_path), "--no-pdf", "--no-source"]),
    ):
        main()

    captured = capsys.readouterr()
    assert "Publications found: 0" in captured.out, (
        "Should print 'Publications found: 0' so user notices the issue. "
        f"Got stdout={captured.out!r}"
    )
    # A warning about zero publications should also appear
    assert "Warning: no publications found" in captured.out, (
        "Expected a 'Warning: no publications found' message. "
        f"Got stdout={captured.out!r}"
    )


def test_main_all_pubs_in_bib_before_scholarly_fill(tmp_path):
    """All publications must be written to the bib file before scholarly.fill()
    is called for any individual publication.

    This guarantees that the bib is complete (priority 1 from issue #8) even
    if scholarly.fill() hangs or fails for some entries.
    """
    pubs = [_make_fake_pub(f"Paper {i}") for i in range(1, 6)]  # 5 publications
    bib_path = tmp_path / "publications.bib"
    event_log: list[str] = []

    original_write_text = Path.write_text

    def tracking_write_text(self, data, **kwargs):
        if self == bib_path:
            event_log.append(f"bib_write:{data.count('@article')}")
        original_write_text(self, data, **kwargs)

    fill_pub_calls = [0]

    def fill_side_effect(obj, **kwargs):
        if "sections" in kwargs:
            return obj  # author fill – pass through
        fill_pub_calls[0] += 1
        event_log.append(f"fill_pub:{fill_pub_calls[0]}")
        return obj

    mock_scholarly = MagicMock()
    mock_scholarly.search_author_id.return_value = {"name": "Test", "publications": pubs}
    mock_scholarly.fill.side_effect = fill_side_effect

    with (
        patch("fetch_publications.scholarly", mock_scholarly),
        patch("fetch_publications.time.sleep"),
        patch.object(Path, "write_text", tracking_write_text),
        patch(
            "sys.argv",
            [
                "fetch_publications.py",
                "https://scholar.google.com/citations?user=TEST",
                "--output-dir",
                str(tmp_path),
                "--no-pdf",
                "--no-source",
            ],
        ),
    ):
        main()

    # The bib file must contain all 5 entries BEFORE any fill() call for individual pubs
    all_written_idx = next(
        (i for i, e in enumerate(event_log) if e == "bib_write:5"), None
    )
    first_fill_idx = next(
        (i for i, e in enumerate(event_log) if e.startswith("fill_pub")), None
    )

    assert all_written_idx is not None, "bib file was never written with all 5 entries"
    if first_fill_idx is not None:
        assert all_written_idx < first_fill_idx, (
            "Expected all 5 publications to be written to bib before scholarly.fill() "
            "is called for any individual publication. "
            f"First 'all-written' bib event was at index {all_written_idx}, "
            f"but first fill() call was at index {first_fill_idx}. "
            "Event log: " + str(event_log)
        )


# ---------------------------------------------------------------------------
# Phase 3/4 robustness: loop must not stop on unexpected exceptions
# ---------------------------------------------------------------------------

def test_phase3_continues_after_pdf_download_exception(tmp_path):
    """Phase 3 must continue processing remaining publications even when
    download_pdf raises an unexpected exception (e.g. OSError) for one entry.

    Without a try/except in the Phase 3 loop, the first uncaught exception
    stops the loop and all older publications (processed later in reverse-
    chronological order) are silently skipped.
    """
    pubs = [
        _make_fake_pub("New Paper", year="2023", arxiv_id="2301.00001"),
        _make_fake_pub("Middle Paper", year="2021", arxiv_id="2101.00001"),
        _make_fake_pub("Old Paper", year="2019", arxiv_id="1901.00001"),
    ]
    mock_scholarly = _make_mock_scholarly(pubs)

    attempted_titles: list[str] = []

    def fake_download_pdf(pub, pub_dir, force=False):
        title = pub.get("bib", {}).get("title", "")
        attempted_titles.append(title)
        if title == "Middle Paper":
            raise OSError("Simulated disk-full error")
        return True

    with (
        patch("fetch_publications.scholarly", mock_scholarly),
        patch("fetch_publications.time.sleep"),
        patch("fetch_publications.download_pdf", side_effect=fake_download_pdf),
        patch("fetch_publications.download_arxiv_source", return_value=False),
        patch("sys.argv", [
            "fetch_publications.py",
            "https://scholar.google.com/citations?user=TEST",
            "--output-dir", str(tmp_path),
            "--no-source",
        ]),
    ):
        main()

    assert "New Paper" in attempted_titles, "New Paper should have been attempted"
    assert "Middle Paper" in attempted_titles, "Middle Paper should have been attempted"
    assert "Old Paper" in attempted_titles, (
        "Old Paper must still be attempted even though Middle Paper raised an "
        "exception.  The Phase 3 loop must not abort on unexpected errors."
    )


def test_phase4_continues_after_source_download_exception(tmp_path):
    """Phase 4 must continue processing remaining publications even when
    download_arxiv_source raises an unexpected exception for one entry.

    Without a try/except in the Phase 4 loop, the first uncaught exception
    stops the loop and older publications lose their LaTeX source downloads.
    """
    pubs = [
        _make_fake_pub("New Paper", year="2023", arxiv_id="2301.00001"),
        _make_fake_pub("Middle Paper", year="2021", arxiv_id="2101.00001"),
        _make_fake_pub("Old Paper", year="2019", arxiv_id="1901.00001"),
    ]
    mock_scholarly = _make_mock_scholarly(pubs)

    attempted_ids: list[str] = []

    def fake_download_source(arxiv_id, pub_dir, force=False):
        attempted_ids.append(arxiv_id)
        if arxiv_id == "2101.00001":
            raise OSError("Simulated disk-full error")
        return True

    with (
        patch("fetch_publications.scholarly", mock_scholarly),
        patch("fetch_publications.time.sleep"),
        patch("fetch_publications.download_pdf", return_value=False),
        patch("fetch_publications.download_arxiv_source", side_effect=fake_download_source),
        patch("sys.argv", [
            "fetch_publications.py",
            "https://scholar.google.com/citations?user=TEST",
            "--output-dir", str(tmp_path),
            "--no-pdf",
        ]),
    ):
        main()

    assert "2301.00001" in attempted_ids, "New Paper's arXiv source should have been attempted"
    assert "2101.00001" in attempted_ids, "Middle Paper's arXiv source should have been attempted"
    assert "1901.00001" in attempted_ids, (
        "Old Paper's arXiv source must still be attempted even though Middle Paper "
        "raised an exception.  The Phase 4 loop must not abort on unexpected errors."
    )
