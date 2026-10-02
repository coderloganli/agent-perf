"""Command-line interface: ``agent-perf {proxy,run,summarize,plot}``."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
from pathlib import Path

import pandas as pd

from agent_perf.proxy import DEFAULT_UPSTREAMS, ProxyConfig, serve


def _cmd_proxy(args: argparse.Namespace) -> int:
    cfg = ProxyConfig(
        provider=args.provider,
        upstream=args.upstream or DEFAULT_UPSTREAMS[args.provider],
        trace_path=Path(args.out),
        default_run=args.run_id,
    )

    async def _main() -> None:
        runner = await serve(cfg, args.host, args.port)
        try:
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_main())
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from agent_perf.harbor import (
        VARIANTS,
        ExperimentConfig,
        format_command,
        harbor_command,
        run_experiment,
    )

    cfg = ExperimentConfig.from_toml(args.config)
    variants = [args.variant] if args.variant else list(VARIANTS)
    if args.dry_run:
        for v in variants:
            print(format_command(harbor_command(cfg, v)))
        return 0
    codes = asyncio.run(run_experiment(cfg, variants))
    for v, code in codes.items():
        print(f"{v}: harbor exit code {code}")
    return max(codes.values(), default=0)


def _cmd_summarize(args: argparse.Namespace) -> int:
    from agent_perf.analysis import (
        compare_variants,
        context_growth_summary,
        fanout,
        latency_summary,
        load_traces,
    )
    from agent_perf.harbor import load_harbor_results

    df = load_traces(args.traces)
    if df.empty:
        print("no records found")
        return 1
    with pd.option_context("display.width", 160, "display.max_columns", 30):
        print("== latency ==")
        print(latency_summary(df))
        print("\n== fan-out ==")
        print(fanout(df))
        print("\n== context growth ==")
        print(context_growth_summary(df))
        if args.jobs:
            results = pd.concat([load_harbor_results(j) for j in args.jobs], ignore_index=True)
            print("\n== variants ==")
            print(compare_variants(df, results))
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    from agent_perf.analysis import load_traces
    from agent_perf.plotting import save_all

    df = load_traces(args.traces)
    if df.empty:
        print("no records found")
        return 1
    for path in save_all(df, args.out):
        print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent-perf")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("proxy", help="run the recording proxy")
    sp.add_argument("--provider", choices=sorted(DEFAULT_UPSTREAMS), required=True)
    sp.add_argument("--upstream", help="upstream base URL (default: provider API)")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8787)
    sp.add_argument("--out", default="traces/requests.jsonl", help="JSONL trace file")
    sp.add_argument("--run-id", default="default", help="run id when no /r/<id>/ prefix")
    sp.set_defaults(func=_cmd_proxy)

    sr = sub.add_parser("run", help="run the subagents on/off experiment through Harbor")
    sr.add_argument("config", help="experiment TOML file")
    sr.add_argument("--variant", choices=["subagents_on", "subagents_off"])
    sr.add_argument("--dry-run", action="store_true", help="print harbor commands only")
    sr.set_defaults(func=_cmd_run)

    ss = sub.add_parser("summarize", help="print workload summaries")
    ss.add_argument("traces", nargs="+", help="trace files or directories")
    ss.add_argument("--jobs", nargs="*", help="Harbor job directories for pass rates")
    ss.set_defaults(func=_cmd_summarize)

    spl = sub.add_parser("plot", help="write workload plots")
    spl.add_argument("traces", nargs="+")
    spl.add_argument("--out", default="plots")
    spl.set_defaults(func=_cmd_plot)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
