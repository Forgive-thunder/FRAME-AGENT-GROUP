"""Central configuration for the FRAME pipeline.

Keep runtime knobs here so ``main.py`` and the agents stay declarative.
"""
from __future__ import annotations

import os

# --- Source data ---------------------------------------------------------
# Absolute path to the prepared test PDFs (as requested).
PAPERS_DIR: str = r"D:\FRAME_Agent_Project\input_papers"

# --- LLM (OpenAI-compatible) ---------------------------------------------
# Read from the environment: API_KEY, BASE_URL, MODEL.
LLM_MODEL: str = os.getenv("MODEL", "gpt-4o-mini")
LLM_API_KEY: str | None = os.getenv("API_KEY")
LLM_BASE_URL: str | None = os.getenv("BASE_URL")

# --- Vector store --------------------------------------------------------
FAISS_DIMENSION: int = 768
FAISS_INDEX_PATH: str | None = None

# --- Stage 1 Extractor <-> Checker game loop -----------------------------
MAX_REFINE_ROUNDS: int = 3     # max correction rounds per paragraph (n=3)
SCORE_THRESHOLD: float = 8.0   # stop early once a Checker score reaches this

# --- Small-scale test mode -----------------------------------------------
MAX_PAPERS: int = None               # read at most this many PDFs from input_papers
MAX_PARAGRAPHS_PER_PAPER: int = None  # process at most this many paragraphs per paper

# --- Stage 2 Generator <-> Evaluator game loop ---------------------------
MAX_STAGE2_ROUNDS: int = 3
STAGE2_SCORE_THRESHOLD: float = 8.0

# --- Persistence ---------------------------------------------------------
OUTPUT_DIR: str = "output_dataset"
STAGE1_RESULTS_FILE: str = "stage1_results.json"
STAGE2_RESULTS_FILE: str = "stage2_results.json"
STAGE3_RESULTS_FILE: str = "stage3_final_report.json"
