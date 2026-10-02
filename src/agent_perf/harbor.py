"""Harbor run orchestration for the parallel-subagents experiment.

Harbor (https://harborframework.com, github.com/laude-institute/harbor) runs
an agent on a task suite in sandboxes: ``harbor run -d <org/dataset@ref>``
(or ``-p <local path>``) ``-a <agent> -m <model>``, with ``-n`` concurrent
trials, ``-k`` attempts, ``-o`` jobs directory, ``--job-name``, repeatable
``--ak key=value`` agent kwargs (JSON-parsed) and ``--ae KEY=VALUE`` agent
environment variables.

This module runs the same task suite twice, once per variant, with the
harness pointed at the recording proxy:

* ``subagents_on``: the harness default.
* ``subagents_off``:
  - Claude Code: ``--ak disallowed_tools=Agent``. Harbor's claude-code agent
    maps ``disallowed_tools`` to ``--disallowedTools``; the Claude Code
    subagent docs name ``Agent`` as the tool that spawns subagents.
  - Codex: ``--ak config=<json>`` with ``agents.enabled = false`` and
    ``features.multi_agent = false`` (both documented in the Codex config
    reference). Harbor uses ``config`` as the base ``config.toml``.

Each variant's requests are tagged with run id ``<name>-<variant>`` through
the proxy's ``/r/<run_id>/`` path prefix.

Assumptions / TODOs:

* Harbor runs agents inside containers. The base URL handed to the harness
  uses ``container_host`` (default ``host.docker.internal``), which must
  resolve to the machine running the proxy from inside the sandbox. This
  holds for Docker Desktop; on Linux it may need ``--add-host`` or a bridge
  IP. Cloud sandbox environments cannot reach a local proxy at all.
* The proxy forwards the harness's own credential headers upstream, so the
  normal ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` must still be available
  to Harbor, which forwards them to the agent.
* Trial-to-session join: Harbor's trial ``agent/trajectory.json`` (ATIF)
  carries a ``session_id``. It is assumed to equal the harness session id
  seen in request headers; this is unverified.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from agent_perf.proxy import DEFAULT_UPSTREAMS, ProxyConfig, serve

VARIANTS = ("subagents_on", "subagents_off")

HARNESSES: dict[str, dict[str, Any]] = {
    "claude-code": {
        "provider": "anthropic",
        "base_url_env": "ANTHROPIC_BASE_URL",
        "base_url_suffix": "",
        # Gateway hint headers (request class, agent type, prompt id, tool
        # durations) are off by default for a custom base URL.
        "extra_env": {"CLAUDE_CODE_GATEWAY_HINT_HEADERS": "1"},
        "subagents_off_kwargs": {"disallowed_tools": "Agent"},
    },
    "codex": {
        "provider": "openai",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url_suffix": "/v1",
        "extra_env": {},
        "subagents_off_kwargs": {
            "config": {"agents": {"enabled": False}, "features": {"multi_agent": False}}
        },
    },
}


@dataclass
class ExperimentConfig:
    name: str
    harness: str
    model: str
    dataset: str | None = None
    path: str | None = None
    n_concurrent: int = 1
    n_attempts: int = 1
    jobs_dir: str = "jobs"
    traces_dir: str = "traces"
    extra_harbor_args: list[str] = field(default_factory=list)
    proxy_bind: str = "0.0.0.0"
    proxy_port: int = 8787
    container_host: str = "host.docker.internal"
    upstream: str | None = None

    def __post_init__(self) -> None:
        if self.harness not in HARNESSES:
            raise ValueError(f"harness must be one of {sorted(HARNESSES)}, got {self.harness!r}")
        if (self.dataset is None) == (self.path is None):
            raise ValueError("set exactly one of 'dataset' or 'path'")

    @property
    def provider(self) -> str:
        return HARNESSES[self.harness]["provider"]

    @classmethod
    def from_toml(cls, path: str | Path) -> ExperimentConfig:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        exp = dict(data.get("experiment", {}))
        proxy = data.get("proxy", {})
        for key in ("bind", "port"):
            if key in proxy:
                exp[f"proxy_{key}"] = proxy[key]
        for key in ("container_host", "upstream"):
            if key in proxy:
                exp[key] = proxy[key]
        return cls(**exp)

    def run_id(self, variant: str) -> str:
        return f"{self.name}-{variant}"


def _kwarg(key: str, value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
    return f"{key}={text}"


def harbor_command(cfg: ExperimentConfig, variant: str) -> list[str]:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    spec = HARNESSES[cfg.harness]
    base_url = (
        f"http://{cfg.container_host}:{cfg.proxy_port}/r/{cfg.run_id(variant)}"
        + spec["base_url_suffix"]
    )
    cmd = ["harbor", "run"]
    cmd += ["-d", cfg.dataset] if cfg.dataset else ["-p", cfg.path]
    cmd += ["-a", cfg.harness, "-m", cfg.model]
    cmd += ["-n", str(cfg.n_concurrent), "-k", str(cfg.n_attempts)]
    cmd += ["-o", cfg.jobs_dir, "--job-name", cfg.run_id(variant)]
    cmd += ["--ae", f"{spec['base_url_env']}={base_url}"]
    for k, v in spec["extra_env"].items():
        cmd += ["--ae", f"{k}={v}"]
    if variant == "subagents_off":
        for k, v in spec["subagents_off_kwargs"].items():
            cmd += ["--ak", _kwarg(k, v)]
    cmd += list(cfg.extra_harbor_args)
    return cmd


def trace_path(cfg: ExperimentConfig) -> Path:
    return Path(cfg.traces_dir) / f"{cfg.name}.jsonl"


async def _run_variant(cfg: ExperimentConfig, variant: str) -> int:
    proc = await asyncio.create_subprocess_exec(*harbor_command(cfg, variant))
    return await proc.wait()


async def run_experiment(cfg: ExperimentConfig, variants=VARIANTS) -> dict[str, int]:
    """Start the proxy, run each variant sequentially, return Harbor exit codes."""
    proxy_cfg = ProxyConfig(
        provider=cfg.provider,
        upstream=cfg.upstream or DEFAULT_UPSTREAMS[cfg.provider],
        trace_path=trace_path(cfg),
        default_run=cfg.name,
    )
    runner = await serve(proxy_cfg, cfg.proxy_bind, cfg.proxy_port)
    codes: dict[str, int] = {}
    try:
        for variant in variants:
            codes[variant] = await _run_variant(cfg, variant)
    finally:
        await runner.cleanup()
    return codes


def format_command(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd)


# --- results ---------------------------------------------------------------


def _reward(result: dict[str, Any]) -> float | None:
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    if not rewards:
        return None
    # TODO: tasks may define several reward keys; "reward" is used when present,
    # otherwise the first key. Revisit per task suite.
    value = rewards.get("reward", next(iter(rewards.values())))
    return float(value) if isinstance(value, (int, float)) else None


def _trajectory_session_id(trial_dir: Path) -> str | None:
    traj = trial_dir / "agent" / "trajectory.json"
    if not traj.is_file():
        return None
    try:
        return json.loads(traj.read_text(encoding="utf-8")).get("session_id")
    except (json.JSONDecodeError, OSError, AttributeError):
        return None


def load_harbor_results(
    job_dir: str | Path, run_id: str | None = None, pass_threshold: float = 1.0
) -> pd.DataFrame:
    """One row per trial from ``<jobs_dir>/<job_name>/<trial>/result.json``."""
    job_dir = Path(job_dir)
    rows = []
    for result_file in sorted(job_dir.glob("*/result.json")):
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "trial_name" not in data:
            continue
        reward = _reward(data)
        exc = data.get("exception_info") or {}
        rows.append(
            {
                "run_id": run_id or job_dir.name,
                "task_name": data.get("task_name"),
                "trial_name": data.get("trial_name"),
                "reward": reward,
                "passed": reward is not None and reward >= pass_threshold,
                "exception_type": exc.get("exception_type"),
                "session_id": _trajectory_session_id(result_file.parent),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "run_id",
            "task_name",
            "trial_name",
            "reward",
            "passed",
            "exception_type",
            "session_id",
        ],
    )
