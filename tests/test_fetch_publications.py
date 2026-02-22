"""
Unit tests for fetch_publications.py helper functions.
Run with:  python -m pytest tests/
"""

import tarfile
import io
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from fetch_publications import (
    sanitize_filename,
    make_folder_name,
    make_bibtex_key,
    format_bibtex,
    extract_arxiv_id,
    download_pdf,
    download_arxiv_source,
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
    pub = {"bib": {"title": "Conference Paper", "venue": "International Conference on ML"}}
    entry = format_bibtex(pub, "key1")
    assert entry.startswith("@inproceedings{")


def test_format_bibtex_article_type():
    pub = {"bib": {"title": "Journal Paper", "venue": "Nature"}}
    entry = format_bibtex(pub, "key1")
    assert entry.startswith("@article{")


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
# download_pdf (mocked)
# ---------------------------------------------------------------------------

def test_download_pdf_no_eprint_url(tmp_path):
    pub = {"bib": {}, "eprint_url": ""}
    result = download_pdf(pub, tmp_path)
    assert result is False
    assert not any(tmp_path.iterdir())


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
