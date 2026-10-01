# agent-perf

An open-source profiler for the inference workloads coding agents generate.

agent-perf runs Claude Code and Codex on Harbor task suites and records every model request the harness issues: input, cached and output tokens, time to first token, inter-token latency, throughput, and which subagent or tool call produced the request.

## Analysis

- **Workload shape**: the traces are analyzed in Python with pandas to characterize workload shape across harnesses: how context grows over long-horizon tool use, how parallel subagents fan requests out, and how much of each prompt is a reused prefix.
- **Harness architecture**: parallel subagents are switched on or off and the same Harbor tasks are rerun, measuring how that decision shifts inference load against task pass rate.

## Status

Under active development. No measured results are published yet.
