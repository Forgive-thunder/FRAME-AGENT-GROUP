"""Data-loading module for the FRAME pipeline.

Reads the prepared test PDFs from an absolute path (see ``PAPERS_DIR``) and
hands each paper to the Extractor agent as a plain dict.

The loader is deliberately defensive: if the directory is missing, or if
``pypdf`` is not installed, it degrades gracefully so ``main.py`` keeps
running end-to-end.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

# Absolute path to the source data (as requested).
PAPERS_DIR: Path = Path(r"D:\FRAME_Agent_Project\input_papers")


def list_papers(directory: Path | str | None = None) -> List[Path]:
    """Return the sorted list of ``*.pdf`` files under ``directory``.

    Returns an empty list (rather than raising) when the directory does not
    exist, so the scaffold stays runnable before the data is in place.
    """
    directory = Path(directory) if directory is not None else PAPERS_DIR
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.pdf"))


def load_paper(path: Path | str) -> Dict[str, str]:
    """Extract the text of a single PDF.

    Returns a dict of the shape the Extractor agent expects::

        {
            "paper_id": "<filename stem>",
            "path": "<absolute path>",
            "text": "<raw text>",
        }

    If ``pypdf`` is unavailable, a dummy text block is returned instead so the
    remaining stages still exercise the full data flow.
    """
    path = Path(path)
    text: str

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except ImportError:
        text = f"[DUMMY TEXT — 'pypdf' not installed] would read: {path.name}"
    except Exception as exc:  # noqa: BLE001 — keep the scaffold running on bad PDFs
        text = f"[READ ERROR] {path.name}: {exc}"

    return {
        "paper_id": path.stem,
        "filename": path.name,
        "path": str(path),
        "text": text,
    }


def split_paragraphs(text: str, min_length: int = 40) -> List[str]:
    """Split raw paper text into paragraph chunks (blank-line separated).

    Very short chunks (headers, page numbers, etc.) are dropped so the game
    loop only runs on substantive paragraphs.
    """
    chunks = re.split(r"\n\s*\n", text or "")
    paragraphs = [chunk.strip() for chunk in chunks]
    return [p for p in paragraphs if len(p) >= min_length]
