"""Persistence helpers for the FRAME pipeline.

Stage 1 writes its finalized records to ``output_dataset/stage1_results.json``;
Stage 2 reads those records and writes its generated instances to
``output_dataset/stage2_results.json``; Stage 3 reads both and writes the
final report to ``output_dataset/stage3_final_report.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import config


def _resolve(path: Optional[Union[str, Path]], default: Path) -> Path:
    return Path(path) if path is not None else default


def _write_json(data: Any, path: Path) -> Path:
    """Write ``data`` to ``path`` as JSON, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return path


def _read_list(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON list from disk, raising a clear error if absent/malformed."""
    if not path.exists():
        raise FileNotFoundError(
            f"File not found at {path}. Run the earlier stage first (python main.py)."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list at {path}, got {type(data).__name__}."
        )
    return data


def save_stage1_results(
    dataset: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Write the Stage 1 record list to JSON (auto-creates the directory)."""
    return _write_json(
        dataset,
        _resolve(output_path, Path(config.OUTPUT_DIR) / config.STAGE1_RESULTS_FILE),
    )


def load_stage1_results(
    input_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Read the Stage 1 record list from disk (used as Stage 2/3 input)."""
    return _read_list(
        _resolve(input_path, Path(config.OUTPUT_DIR) / config.STAGE1_RESULTS_FILE)
    )


def save_stage2_results(
    instances: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Write the winning Stage 2 instances to JSON (auto-creates the directory)."""
    return _write_json(
        instances,
        _resolve(output_path, Path(config.OUTPUT_DIR) / config.STAGE2_RESULTS_FILE),
    )


def load_stage2_results(
    input_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Read the Stage 2 instances from disk (used as Stage 3 input)."""
    return _read_list(
        _resolve(input_path, Path(config.OUTPUT_DIR) / config.STAGE2_RESULTS_FILE)
    )


def save_stage3_final_report(
    final_answer: str,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Write the final report text as ``{"final_answer": ...}`` to JSON."""
    return _write_json(
        {"final_answer": final_answer},
        _resolve(output_path, Path(config.OUTPUT_DIR) / config.STAGE3_RESULTS_FILE),
    )
