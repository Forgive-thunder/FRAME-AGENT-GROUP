# FRAME — Feedback-Refined Agent Methodology

Python skeleton reproducing the FRAME multi-agent pipeline. Stages 1 and 2
call a real LLM via an OpenAI-compatible client; Stage 3 is still a stub.

## Layout

```
FRAME-AGENT-GROUP/
├── main.py              # orchestrates the 3 stages, wires the 7 agents
├── config.py            # paths + pipeline knobs (rounds, thresholds, LLM)
├── data/loader.py       # reads PDFs from D:\FRAME_Agent_Project\input_papers
├── data/storage.py      # persists/loads stage1_results.json & stage2_results.json
├── llm/client.py        # OpenAI-compatible LLM client (real, JSON mode)
├── memory/faiss_store.py# FAISS placeholder (add_memory / search_memory stubs)
└── agents/
    ├── base.py          # shared BaseAgent interface
    ├── extractor.py     # Stage 1 — real prompt (Paper Analyst)
    ├── checker.py       # Stage 1 — real prompt (Paper Evaluator)
    ├── reflector.py     # Stage 2 — real prompt (failure analysis / feedback)
    ├── filter.py        # Stage 1 — unused in current loop (kept for parity)
    ├── generator.py     # Stage 2 — real prompt (task-instance generator)
    ├── evaluator.py     # Stage 2 — real prompt (3-dimension evaluator)
    └── integrator.py    # Stage 3 — real prompt (medical insight report)
```

## Run

Export your credentials, then run from the project root:

```bash
export API_KEY="sk-..."
export BASE_URL="https://api.example.com/v1"   # optional; omit for OpenAI
export MODEL="gpt-4o-mini"                     # optional, defaults to gpt-4o-mini

python main.py
```

The pipeline runs in small-scale test mode: at most `MAX_PAPERS` (2) PDFs and
`MAX_PARAGRAPHS_PER_PAPER` (2) paragraphs each.

## Stage 1 — Dataset Construction

For each paragraph, `Extractor -> Checker` alternate (up to `MAX_REFINE_ROUNDS`).
The finalized records are saved to `output_dataset/stage1_results.json` with the
schema `{source_file, original_paragraph, final_conclusion, final_score, refinement_rounds}`.

## Stage 2 — Training

`Generator -> Evaluator`, with `Reflector` feedback:

1. **Generator** reads the Stage 1 `final_conclusion` set and produces
   `{"instances": [{"instruction", "input", "output"}, ...]}`.
2. **Evaluator** scores on difficulty / diversity / accuracy:
   `{"score": 8.5, "reason": "...", "verdict": "pass"}`.
3. If `verdict == "fail"` or `score < STAGE2_SCORE_THRESHOLD` (8.0), the
   **Reflector** emits `{"feedback": "..."}`, which is fed back to the
   Generator.  Up to `MAX_STAGE2_ROUNDS` (3) rounds; the highest-scored
   instances are saved to `output_dataset/stage2_results.json`.

## Stage 3 — Inference

The Integrator (a medical research analyst) reads `stage1_results.json` and
`stage2_results.json`, produces a comprehensive medical insight report, prints
it, and saves it to `output_dataset/stage3_final_report.json`.
