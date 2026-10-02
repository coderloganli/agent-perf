from __future__ import annotations

from agent_perf.attribution import (
    analyze_body,
    attribution_from_headers,
    parse_prev_tool_durations,
    prefix_fingerprint,
)


def test_claude_code_headers_with_hints():
    a = attribution_from_headers(
        {
            "X-Claude-Code-Session-Id": "s1",
            "x-claude-code-agent-id": "a1",
            "x-claude-code-parent-agent-id": "a0",
            "x-claude-code-request-class": "subagent",
            "x-claude-code-agent-type": "Explore",
            "x-claude-code-prompt-id": "p1",
            "x-claude-code-prev-tool-durations": "Bash=742;Read=9;Bash=8;my%3Btool=3;bad",
        }
    )
    assert (a.session_id, a.agent_id, a.parent_agent_id) == ("s1", "a1", "a0")
    assert (a.request_class, a.agent_type, a.prompt_id) == ("subagent", "Explore", "p1")
    assert a.prev_tool_durations_ms == {"Bash": 750, "Read": 9, "my;tool": 3}


def test_claude_code_headers_without_hints_infer_class_from_agent_id():
    assert attribution_from_headers({"x-claude-code-session-id": "s"}).request_class == "main"
    sub = attribution_from_headers({"x-claude-code-session-id": "s", "x-claude-code-agent-id": "a"})
    assert sub.request_class == "subagent"


def test_codex_headers():
    main = attribution_from_headers({"session-id": "s", "thread-id": "t0"})
    assert (main.session_id, main.agent_id, main.request_class) == ("s", None, "main")
    spawn = attribution_from_headers(
        {
            "session-id": "s",
            "thread-id": "t1",
            "x-codex-parent-thread-id": "t0",
            "x-openai-subagent": "collab_spawn",
        }
    )
    assert (spawn.agent_id, spawn.parent_agent_id, spawn.request_class) == ("t1", "t0", "subagent")
    assert attribution_from_headers({"x-openai-subagent": "compact"}).request_class == "compaction"


def test_unknown_client_yields_empty_attribution():
    assert attribution_from_headers({"user-agent": "curl"}).session_id is None


def test_parse_prev_tool_durations_empty():
    assert parse_prev_tool_durations(None) == {}


def test_anthropic_body_triggering_tools_and_prefix():
    body = {
        "model": "m",
        "stream": True,
        "system": [{"type": "text", "text": "sys"}],
        "tools": [{"name": "Bash"}],
        "messages": [
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}},
                    {"type": "tool_use", "id": "t2", "name": "Read", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                    {"type": "tool_result", "tool_use_id": "t2", "content": "ok"},
                ],
            },
        ],
    }
    shape = analyze_body("anthropic", body)
    assert shape.stream and shape.model == "m"
    assert shape.n_input_items == 3
    assert shape.triggering_tools == ["Bash", "Read"]
    assert len(shape.prefix_hashes) == len(shape.prefix_chars) == 4
    assert shape.prefix_chars == sorted(shape.prefix_chars)


def test_anthropic_body_plain_user_turn_has_no_trigger():
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    assert analyze_body("anthropic", body).triggering_tools == []


def test_responses_body_triggering_tools():
    body = {
        "model": "m",
        "instructions": "be brief",
        "input": [
            {"type": "message", "role": "user", "content": "go"},
            {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
            {"type": "custom_tool_call", "call_id": "c2", "name": "apply_patch", "input": ""},
            {"type": "function_call_output", "call_id": "c1", "output": "x"},
            {"type": "custom_tool_call_output", "call_id": "c2", "output": "y"},
        ],
    }
    shape = analyze_body("openai", body)
    assert shape.triggering_tools == ["shell", "apply_patch"]
    assert shape.n_input_items == 5


def test_responses_body_string_input():
    shape = analyze_body("openai", {"model": "m", "input": "hello"})
    assert shape.n_input_items == 1 and shape.triggering_tools == []


def test_prefix_fingerprint_shared_prefix_matches():
    head = {"system": "s"}
    a, _ = prefix_fingerprint(head, ["u1", "a1"])
    b, _ = prefix_fingerprint(head, ["u1", "a1", "u2"])
    c, _ = prefix_fingerprint(head, ["u1-edited", "a1"])
    assert b[: len(a)] == a
    assert c[0] == a[0] and c[1] != a[1]
