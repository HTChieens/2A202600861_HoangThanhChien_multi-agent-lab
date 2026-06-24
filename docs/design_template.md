# Design Template

## Problem

Build a research assistant that accepts a long-form question, gathers or plans sources, analyzes the evidence, writes a final answer, and records trace/benchmark data for comparison with a single-agent baseline.

## Why multi-agent?

A single-agent baseline is simpler, but it mixes search planning, evidence analysis, writing, and review in one prompt. The multi-agent design separates responsibilities so each step is easier to trace, debug, retry, and evaluate.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Route the next worker and stop safely | Shared `ResearchState` | Route in `route_history` | Unknown route or max-iteration stop |
| Researcher | Collect sources and produce research notes | Query, audience, max sources | `sources`, `research_notes` | Search/LLM fallback records errors |
| Analyst | Extract claims, implications, caveats | Research notes, sources | `analysis_notes` | Validation error if notes are missing |
| Writer | Synthesize final answer | Research and analysis notes | `final_answer` | Fallback answer from state if LLM fails |
| Critic | Review answer quality and citation risks | Final answer, sources | Critic `AgentResult` | Validation error if answer is missing |

## Shared state

`ResearchState` contains:

- `request`: original `ResearchQuery`.
- `iteration` and `route_history`: routing trace and loop guard.
- `sources`: search results or local fallback source notes.
- `research_notes`, `analysis_notes`, `final_answer`: handoff artifacts between agents.
- `agent_results`: structured record of each agent output.
- `trace`: JSON-serializable events/spans for observability.
- `errors`: recoverable failures and fallback notes.

## Routing policy

The default route is deterministic and inspectable:

```text
Supervisor
  -> Researcher when research_notes are missing
  -> Analyst when analysis_notes are missing
  -> Writer when final_answer is missing
  -> Critic once after final_answer exists
  -> done
```

The supervisor enforces `MAX_ITERATIONS`; the workflow also has a `max_iterations + 2` hard stop.

## Guardrails

- Max iterations: `Settings.max_iterations`, default 6.
- Timeout: `Settings.timeout_seconds`, used by LLM/search clients.
- Retry: `LLMClient` retries provider execution errors with exponential backoff.
- Fallback: search falls back to local sources; worker agents fall back to state-derived text when LLM calls fail.
- Validation: Pydantic schemas validate config/state inputs; agents raise `ValidationError` for missing required handoff fields.

## Benchmark plan

Run:

```bash
python -m multi_agent_research_lab.cli benchmark \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

Metrics:

- Latency: wall-clock time per runner.
- Cost: summed provider-reported cost when available.
- Quality: heuristic score based on completed artifacts, answer length, sources, and errors.
- Citation coverage: ratio of non-fallback sources.
- Failure signal: count of recoverable errors in `ResearchState.errors`.

Expected outcome: baseline is faster and simpler; multi-agent is more traceable and reviewable, especially when source collection and critique matter.
