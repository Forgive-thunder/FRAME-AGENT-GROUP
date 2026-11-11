"""FRAME — Feedback-Refined Agent Methodology.

Orchestrates the three paper stages:

  1. Dataset Construction — Extractor <-> Checker game loop per paragraph,
     persisted to ``output_dataset/stage1_results.json``.
  2. Training            — Generator -> Evaluator game loop with Reflector
     feedback, persisted to ``output_dataset/stage2_results.json``.
  3. Inference           — Integrator (medical insight report).

Run from the project root::

    python main.py

Stages 1, 2, and 3 call a real LLM via the OpenAI-compatible client in
``llm.client``.  Set the environment variables ``API_KEY`` and ``BASE_URL``
(and optionally ``MODEL``) before running.  The pipeline runs in small-scale
test mode (see ``config.MAX_PAPERS`` / ``config.MAX_PARAGRAPHS_PER_PAPER``).
"""
from __future__ import annotations

import sys

# Reconfigure stdout to UTF-8 so Windows terminals render Chinese output correctly.
sys.stdout.reconfigure(encoding="utf-8")

from typing import Any, Dict, List

import config
from data.loader import PAPERS_DIR, list_papers, load_paper, split_paragraphs
from data.storage import (
    load_stage1_results,
    load_stage2_results,
    save_stage1_results,
    save_stage2_results,
    save_stage3_final_report,
)
from memory.faiss_manager import MemoryManager
from agents import (
    ExtractorAgent,
    CheckerAgent,
    GeneratorAgent,
    EvaluatorAgent,
    ReflectorAgent,
    IntegratorAgent,
)


# ---------------------------------------------------------------------------
# Stage 1 — Dataset Construction
# ---------------------------------------------------------------------------
def run_game_loop(
    extractor: ExtractorAgent,
    checker: CheckerAgent,
    source_file: str,
    paragraph: str,
) -> Dict[str, Any]:
    """Alternate Extractor -> Checker for one paragraph (max ``n`` rounds).

    Each round's ``{score, reason}`` is fed back to the Extractor on the next
    round until the score reaches the threshold or the budget is exhausted.
    The highest-scored conclusion wins.

    Returns the finalized record::

        {
            "source_file": str,
            "original_paragraph": str,
            "final_conclusion": str,
            "final_score": float,
            "refinement_rounds": int,
        }
    """
    history: List[Dict[str, Any]] = []
    best_conclusion: str = ""
    best_score: float = -1.0

    for round_no in range(1, config.MAX_REFINE_ROUNDS + 1):
        print(f"  round {round_no}/{config.MAX_REFINE_ROUNDS}")

        extraction = extractor.run(
            {"paragraph": paragraph, "history": list(history)}
        )
        conclusion = extraction["conclusion"]

        check = checker.run({"paragraph": paragraph, "conclusion": conclusion})
        score = float(check["score"])
        reason = check["reason"]
        print(f"    score={score:.1f}  reason={reason[:80]}")

        if score > best_score:
            best_score = score
            best_conclusion = conclusion

        history.append({"score": score, "reason": reason})

        if score >= config.SCORE_THRESHOLD:
            print(f"    threshold reached ({score:.1f} >= {config.SCORE_THRESHOLD}).")
            break

    return {
        "source_file": source_file,
        "original_paragraph": paragraph,
        "final_conclusion": best_conclusion,
        "final_score": best_score,
        "refinement_rounds": len(history),
    }


def stage1_dataset_construction(papers: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Run the Extractor <-> Checker game loop over every paragraph."""
    print("\n=== Stage 1: Dataset Construction ===")
    extractor = ExtractorAgent()
    checker = CheckerAgent()

    dataset: List[Dict[str, Any]] = []
    for paper in papers:
        paragraphs = split_paragraphs(paper["text"])
        if config.MAX_PARAGRAPHS_PER_PAPER:
            paragraphs = paragraphs[: config.MAX_PARAGRAPHS_PER_PAPER]

        print(f"\n-- paper: {paper['filename']} ({len(paragraphs)} paragraphs) --")
        for idx, para in enumerate(paragraphs, start=1):
            print(f"  [para {idx}/{len(paragraphs)}]")
            dataset.append(run_game_loop(extractor, checker, paper["filename"], para))

    print(f"\nStage 1 complete: {len(dataset)} conclusions kept.")
    return dataset


# ---------------------------------------------------------------------------
# Stage 2 — Training
# ---------------------------------------------------------------------------
def stage2_training() -> List[Dict[str, Any]]:
    """Generator -> Evaluator game loop with Reflector feedback.

    Reads the Stage 1 conclusions from ``stage1_results.json`` and returns the
    winning instances (persisted to ``stage2_results.json``).
    """
    print("\n=== Stage 2: Training ===")
    stage1_records = load_stage1_results()
    conclusions = [rec["final_conclusion"] for rec in stage1_records]
    print(f"Loaded {len(conclusions)} conclusions from Stage 1.")
    if not conclusions:
        print("WARNING: no conclusions found; Generator has no input.")

    generator = GeneratorAgent()
    evaluator = EvaluatorAgent()
    reflector = ReflectorAgent()

    feedback_history: List[str] = []
    reflections: List[Dict[str, Any]] = []
    best_instances: List[Dict[str, Any]] = []
    best_score: float = -1.0
    final_verdict: str = "fail"

    for round_no in range(1, config.MAX_STAGE2_ROUNDS + 1):
        print(f"\n  round {round_no}/{config.MAX_STAGE2_ROUNDS}")

        generated = generator.run(
            {
                "final_conclusions": conclusions,
                "feedback_history": list(feedback_history),
            }
        )
        instances = generated["instances"]

        eval_report = evaluator.run({"instances": instances})
        score = float(eval_report["score"])
        verdict = eval_report["verdict"]
        reason = eval_report["reason"]
        print(f"    score={score:.1f}  verdict={verdict}  reason={reason[:80]}")

        if score > best_score:
            best_score = score
            best_instances = instances
            final_verdict = verdict

        if verdict == "pass" and score >= config.STAGE2_SCORE_THRESHOLD:
            print("    accepted (pass and score >= threshold).")
            break

        # Fail or low score -> reflect and retry (unless out of budget).
        if round_no < config.MAX_STAGE2_ROUNDS:
            reflection = reflector.run({"instances": instances, "reason": reason})
            feedback_history.append(reflection["feedback"])
            reflections.append(
                {
                    "round": round_no,
                    "reason": reason,
                    "feedback": reflection["feedback"],
                }
            )
            print("    reflected; feeding feedback back to Generator.")

    stage2_data = {"instances": best_instances, "reflections": reflections}
    out_path = save_stage2_results(stage2_data)
    print(f"\nStage 2 complete: {len(best_instances)} instances saved to {out_path}")
    print(f"Best score={best_score:.1f}  verdict={final_verdict}")

    # Reflector output is now persisted -> update the FAISS long-term memory.
    if config.ENABLE_MEMORY:
        try:
            memory = MemoryManager()
            n_chunks = memory.build_or_update_index()
            print(f"FAISS memory updated: {n_chunks} chunks indexed.")
        except Exception as exc:  # noqa: BLE001 — memory is additive, don't kill the run
            print(f"WARNING: FAISS memory update failed: {exc}")

    return best_instances


# ---------------------------------------------------------------------------
# Stage 3 — Inference
# ---------------------------------------------------------------------------
def stage3_inference() -> Dict[str, Any]:
    """Integrator reads Stage 1 + Stage 2 results and produces the report."""
    print("\n=== Stage 3: Inference ===")
    stage1_records = load_stage1_results()
    stage2_instances = load_stage2_results()["instances"]
    conclusions = [rec["final_conclusion"] for rec in stage1_records]
    print(
        f"Loaded {len(conclusions)} conclusions and {len(stage2_instances)} instances."
    )

    integrator = IntegratorAgent()
    final = integrator.run(
        {"final_conclusions": conclusions, "instances": stage2_instances}
    )

    final_answer = final["final_answer"]
    print("\n" + "=" * 60)
    print("Comprehensive Medical Insight Report")
    print("=" * 60)
    print(final_answer)
    print("=" * 60)

    out_path = save_stage3_final_report(final_answer)
    print(f"\nStage 3 report saved to: {out_path}")
    return final


def main() -> None:
    print("FRAME scaffold starting ...")
    print(f"Source data path: {PAPERS_DIR}")
    print(f"LLM model: {config.LLM_MODEL}  base_url: {config.LLM_BASE_URL}")
    print(
        f"Test mode: first {config.MAX_PAPERS} paper(s), "
        f"{config.MAX_PARAGRAPHS_PER_PAPER} paragraph(s) each"
    )

    papers = list_papers()[: config.MAX_PAPERS]
    if not papers:
        print(f"WARNING: no PDFs found under {PAPERS_DIR}. Stage 1 will be empty.")

    loaded = [load_paper(p) for p in papers]

    # Stage 1
    dataset = stage1_dataset_construction(loaded)
    out1 = save_stage1_results(dataset)
    print(f"\nStage 1 results saved to: {out1}")

    # Stage 2 (reads Stage 1 results back from disk)
    stage2_training()

    # Stage 3 (reads Stage 1 & Stage 2 results back from disk)
    stage3_inference()

    print("\n=== Pipeline finished ===")


if __name__ == "__main__":
    main()
