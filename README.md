# FRAME — Feedback-Refined Agent Methodology

Python skeleton reproducing the FRAME multi-agent pipeline. All three stages
call a real LLM via an OpenAI-compatible client; a FAISS long-term memory feeds
the Stage 3 topic-feasibility Filter.

## Layout

```
FRAME-AGENT-GROUP/
├── main.py              # orchestrates the 3 stages, wires the 7 agents
├── test_memory.py       # demo: FAISS retrieval + Filter topic check
├── config.py            # paths + pipeline knobs (rounds, thresholds, LLM)
├── data/loader.py       # reads PDFs from D:\FRAME_Agent_Project\input_papers
├── data/storage.py      # persists/loads stage1/2/3 result files
├── llm/client.py        # OpenAI-compatible LLM client (real, JSON mode)
├── memory/faiss_manager.py# FAISS long-term memory (build/query + embedding)
├── memory/faiss_store.py# FAISS low-level placeholder (stubs)
└── agents/
    ├── base.py          # shared BaseAgent interface
    ├── extractor.py     # Stage 1 — real prompt (Paper Analyst)
    ├── checker.py       # Stage 1 — real prompt (Paper Evaluator)
    ├── reflector.py     # Stage 2 — real prompt (failure analysis / feedback)
    ├── filter.py        # Stage 3 — topic feasibility (FAISS-backed)
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

pip install -r requirements.txt                # incl. faiss-cpu + sentence-transformers

python main.py
```

`MAX_PAPERS` / `MAX_PARAGRAPHS_PER_PAPER` in `config.py` control the test-mode
size (set to `None` for a full run).

## Stage 1 — Dataset Construction

For each paragraph, `Extractor -> Checker` alternate (up to `MAX_REFINE_ROUNDS`).
Finalized records are saved to `output_dataset/stage1_results.json`.

## Stage 2 — Training

`Generator -> Evaluator`, with `Reflector` feedback. Up to `MAX_STAGE2_ROUNDS`
rounds; the winning instances **and** the Reflector's reflections are saved to
`output_dataset/stage2_results.json` (as `{instances, reflections}`). After
saving, the FAISS long-term memory is rebuilt automatically.

## Stage 3 — Inference

The Integrator reads `stage1_results.json` and `stage2_results.json`, produces a
comprehensive medical insight report, prints it, and saves it to
`output_dataset/stage3_final_report.json`.

## Long-term memory (FAISS)

`memory/faiss_manager.py` vectorizes the Stage 2 reflections + instances into a
FAISS index (`data/reflector_memory.index`) using a sentence-transformers
embedder. The `Filter` agent queries this memory to judge whether a proposed
research topic has historical support. Run `python test_memory.py` for a quick
end-to-end demo.
