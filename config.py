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

# --- Vector store / long-term memory ------------------------------------
# Dimension is auto-detected from the embedding model; kept for reference.
FAISS_DIMENSION: int = 768
FAISS_INDEX_PATH: str = "data/reflector_memory.index"
FAISS_METADATA_PATH: str = "data/reflector_memory_metadata.json"

# Embedding model used to vectorize memory chunks (sentence-transformers).
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Stage 1 Extractor <-> Checker game loop -----------------------------
MAX_REFINE_ROUNDS: int = 3     # max correction rounds per paragraph (n=3)
SCORE_THRESHOLD: float = 8.0   # stop early once a Checker score reaches this

# --- Small-scale test mode -----------------------------------------------
MAX_PAPERS: int = None               # read at most this many PDFs from input_papers
MAX_PARAGRAPHS_PER_PAPER: int = None  # process at most this many paragraphs per paper

# --- Stage 2 Generator <-> Evaluator game loop ---------------------------
MAX_STAGE2_ROUNDS: int = 3
STAGE2_SCORE_THRESHOLD: float = 8.0

# --- Stage 3 Filter (topic feasibility / long-term memory) ---------------
FILTER_TOP_K: int = 3
FILTER_MIN_SCORE: float = 0.0   # cosine-similarity threshold for "relevant"
ENABLE_MEMORY: bool = True      # build/update the FAISS index during the pipeline

# --- Persistence ---------------------------------------------------------
OUTPUT_DIR: str = "output_dataset"
STAGE1_RESULTS_FILE: str = "stage1_results.json"
STAGE2_RESULTS_FILE: str = "stage2_results.json"
STAGE3_RESULTS_FILE: str = "stage3_final_report.json"
