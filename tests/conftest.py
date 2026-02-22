"""
pytest configuration: add the repository root to sys.path so that
fetch_publications.py can be imported without modification.
"""

import sys
from pathlib import Path

# Repository root is the parent of the tests/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))
