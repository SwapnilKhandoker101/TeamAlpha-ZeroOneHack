"""Tests for the JSON-fence stripping in the Featherless wrapper.

Small instruct models wrap JSON in ```json fences and sometimes add a line of
prose; the tag-picker depends on this being parsed reliably."""

import json

from gas_agent.llm import strip_json_fence


def test_strips_json_fence():
    raw = '```json\n{"ok": true}\n```'
    assert json.loads(strip_json_fence(raw)) == {"ok": True}


def test_strips_bare_fence():
    raw = '```\n{"a": 1}\n```'
    assert json.loads(strip_json_fence(raw)) == {"a": 1}


def test_passes_through_unfenced_json():
    raw = '{"a": 1}'
    assert json.loads(strip_json_fence(raw)) == {"a": 1}


def test_handles_unterminated_fence():
    raw = '```json\n{"a": 1}'
    assert json.loads(strip_json_fence(raw)) == {"a": 1}
