# agent-perf

An open-source profiler for the inference workloads coding agents generate.

agent-perf runs Claude Code and Codex on Harbor task suites and records every model request the harness issues: input, cached and output tokens, time to first token, inter-token latency, throughput, and which subagent or tool call produced the request.

## Analysis

- **Workload shape**: the traces are analyzed in Python with pandas to characterize workload shape across harnesses: how context grows over long-horizon tool use, how parallel subagents fan requests out, and how much of each prompt is a reused prefix.
- **Harness architecture**: parallel subagents are switched on or off and the same Harbor tasks are rerun, measuring how that decision shifts inference load against task pass rate.

## Getting started

Requires Python 3.11+. Running experiments also requires [Harbor](https://harborframework.com) (`uv tool install harbor` or `pip install harbor`), Docker, and API credentials for the harness under test.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Record a harness by hand: start the proxy, then point the harness at it.

```bash
agent-perf proxy --provider anthropic --port 8787 --out traces/requests.jsonl
ANTHROPIC_BASE_URL=http://127.0.0.1:8787 CLAUDE_CODE_GATEWAY_HINT_HEADERS=1 claude

agent-perf proxy --provider openai --port 8788 --out traces/requests.jsonl
OPENAI_BASE_URL=http://127.0.0.1:8788/v1 codex
```

Run the subagents on/off experiment through Harbor, then analyze:

```bash
cp examples/subagents.toml my-experiment.toml   # set harness, model, dataset
agent-perf run my-experiment.toml --dry-run     # print the harbor commands
agent-perf run my-experiment.toml
agent-perf summarize traces/ --jobs jobs/subagents-subagents_on jobs/subagents-subagents_off
agent-perf plot traces/ --out plots/
```

## Design

- **Capture** (`agent_perf.proxy`): an async reverse proxy in front of the Anthropic Messages API (Claude Code, via `ANTHROPIC_BASE_URL`) or the OpenAI Responses API (Codex, via `OPENAI_BASE_URL`). Requests and responses are forwarded unchanged and relayed chunk by chunk; the stream parsers in `agent_perf.streams` timestamp output deltas as they pass through and read the final usage, including cached tokens.
- **Trace schema** (`agent_perf.schema`): one JSONL record per model request with normalized token counts, per-delta arrival offsets, origin attribution and a prefix fingerprint (cumulative hashes of system prompt, tools and input items, so prefix reuse can be measured without storing prompts).
- **Attribution** (`agent_perf.attribution`): session, subagent and request class come from the headers each harness sends (Claude Code documents `x-claude-code-session-id`, `x-claude-code-agent-id` and opt-in hint headers); the tool calls that triggered a request come from the tool results in its trailing turn.
- **Orchestration** (`agent_perf.harbor`): runs `harbor run` once per variant (`subagents_on`, `subagents_off`) with the harness pointed at the proxy, and loads per-trial rewards from Harbor's job directory.
- **Analysis** (`agent_perf.analysis`, `agent_perf.plotting`): pandas functions for context growth, fan-out and concurrency, prefix reuse, latency (TTFT, ITL, throughput) and the variant comparison of inference load against pass rate.

## Status

Under active development. No measured results are published yet.
