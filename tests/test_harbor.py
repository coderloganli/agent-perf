from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_perf.cli import main
from agent_perf.harbor import ExperimentConfig, harbor_command, load_harbor_results

EXAMPLE = Path(__file__).parents[1] / "examples" / "subagents.toml"


def _cfg(**kw) -> ExperimentConfig:
    base = dict(name="exp", harness="claude-code", model="MODEL", dataset="org/ds@1")
    base.update(kw)
    return ExperimentConfig(**base)


def _pairs(cmd: list[str], flag: str) -> list[str]:
    return [cmd[i + 1] for i, a in enumerate(cmd) if a == flag]


def test_claude_code_commands():
    on = harbor_command(_cfg(), "subagents_on")
    off = harbor_command(_cfg(), "subagents_off")
    assert on[:4] == ["harbor", "run", "-d", "org/ds@1"]
    assert _pairs(on, "-a") == ["claude-code"]
    assert _pairs(on, "--job-name") == ["exp-subagents_on"]
    assert "ANTHROPIC_BASE_URL=http://host.docker.internal:8787/r/exp-subagents_on" in _pairs(
        on, "--ae"
    )
    assert "CLAUDE_CODE_GATEWAY_HINT_HEADERS=1" in _pairs(on, "--ae")
    assert _pairs(on, "--ak") == []
    assert _pairs(off, "--ak") == ["disallowed_tools=Agent"]


def test_codex_commands():
    cfg = _cfg(harness="codex", dataset=None, path="tasks/")
    on = harbor_command(cfg, "subagents_on")
    off = harbor_command(cfg, "subagents_off")
    assert _pairs(on, "-p") == ["tasks/"]
    assert "OPENAI_BASE_URL=http://host.docker.internal:8787/r/exp-subagents_on/v1" in _pairs(
        on, "--ae"
    )
    [kwarg] = _pairs(off, "--ak")
    key, _, value = kwarg.partition("=")
    assert key == "config"
    assert json.loads(value) == {"agents": {"enabled": False}, "features": {"multi_agent": False}}


def test_config_validation():
    with pytest.raises(ValueError):
        _cfg(harness="other")
    with pytest.raises(ValueError):
        _cfg(path="x")  # both dataset and path
    with pytest.raises(ValueError):
        harbor_command(_cfg(), "some_other_variant")


def test_example_config_loads():
    cfg = ExperimentConfig.from_toml(EXAMPLE)
    assert cfg.harness == "claude-code" and cfg.proxy_port == 8787


def test_cli_dry_run(capsys):
    assert main(["run", str(EXAMPLE), "--dry-run"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert "subagents_off" in lines[1] and "disallowed_tools=Agent" in lines[1]


def _write_trial(job: Path, name: str, rewards, exc=None, session=None):
    d = job / name
    d.mkdir(parents=True)
    result = {"trial_name": name, "task_name": name.split("__")[0], "verifier_result": None}
    if rewards is not None:
        result["verifier_result"] = {"rewards": rewards}
    if exc:
        result["exception_info"] = {"exception_type": exc}
    (d / "result.json").write_text(json.dumps(result), encoding="utf-8")
    if session:
        (d / "agent").mkdir()
        (d / "agent" / "trajectory.json").write_text(json.dumps({"session_id": session}))


def test_load_harbor_results(tmp_path):
    job = tmp_path / "exp-subagents_on"
    job.mkdir()
    (job / "result.json").write_text(json.dumps({"id": "job-level"}))  # job summary, skipped
    _write_trial(job, "task-a__1", {"reward": 1.0}, session="s-a")
    _write_trial(job, "task-b__1", {"reward": 0.0})
    _write_trial(job, "task-c__1", None, exc="AgentTimeoutError")
    df = load_harbor_results(job).set_index("trial_name")
    assert len(df) == 3
    assert df.loc["task-a__1", "passed"] and df.loc["task-a__1", "session_id"] == "s-a"
    assert not df.loc["task-b__1", "passed"]
    assert df.loc["task-c__1", "exception_type"] == "AgentTimeoutError"
    assert set(df["run_id"]) == {"exp-subagents_on"}
