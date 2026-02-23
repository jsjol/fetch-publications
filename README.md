# Fetch publications from Google Scholar

A Python script that reads a Google Scholar profile URL, fetches all
publications, and:

* generates a **`publications.bib`** file containing full bibliometric
  information plus the abstract of every paper;
* downloads every available **open-access PDF**;
* downloads the **LaTeX source** from arXiv (or a compatible repository)
  for each paper that has one, and stores it in a dedicated folder.

## Requirements

Python 3.10+ and the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python fetch_publications.py "https://scholar.google.com/citations?user=XXXXXXXXX"
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir DIR` | `.` | Directory where `publications.bib` and downloaded files are written |
| `--no-pdf` | – | Skip open-access PDF downloads |
| `--no-source` | – | Skip LaTeX source downloads |
| `--delay SECONDS` | `2.0` | Pause between Scholar requests (reduce if your IP is rate-limited) |

### Example

```bash
python fetch_publications.py \
    "https://scholar.google.com/citations?user=AbCdEfGhIjK" \
    --output-dir ./my_publications \
    --delay 3
```

This will create:

```
my_publications/
├── publications.bib          # all entries with abstracts
├── 2023_Deep_Learning_for/
│   ├── paper.pdf             # open-access PDF (if available)
│   └── main.tex              # arXiv LaTeX source (if available)
├── 2021_Efficient_Neural/
│   └── paper.pdf
└── …
```

## Notes

* Google Scholar aggressively rate-limits automated access.  If you
  encounter errors, try increasing `--delay` or use a proxy (see the
  [`scholarly` documentation](https://scholarly.readthedocs.io/en/stable/ProxyGenerator.html)).
* Only publications with an arXiv (or similar) link will have their LaTeX
  source downloaded.
* BibTeX entries use the `abstract` field, which is supported by most
  modern LaTeX bibliography backends (BibLaTeX, natbib, etc.).
