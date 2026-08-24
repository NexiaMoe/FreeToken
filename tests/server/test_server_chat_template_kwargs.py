"""``--chat-template-kwargs``: server-wide template defaults under each request's keys.

The seam exists for harnesses that cannot send a thinking knob themselves. Hermes, for
instance, is a plain OpenAI client and FreeToken's launcher only patches base_url/api_key/
model/context_length into its config -- so on a checkpoint whose template defaults thinking
OFF (Laguna), nothing the harness does can turn reasoning on.

What matters is precedence: the default must never override a request that speaks about
thinking, in either direction, and it must not swallow a request's ``reasoning_effort``.
"""

from __future__ import annotations

import pytest

from freetoken.server.api_models import ChatCompletionRequest
from freetoken.server.model_meta import merge_server_ctk
from freetoken.server.openai_api import chat_request_to_genspec

THINKING_ON = {"enable_thinking": True}
MESSAGES = [{"role": "user", "content": "hi"}]


def _ctk(server_ctk: dict | None = None, **request_fields) -> dict:
    """The chat_template_kwargs a request ends up rendering with."""
    req = ChatCompletionRequest(model="m", messages=MESSAGES, **request_fields)
    return chat_request_to_genspec(req, {}, server_ctk).chat_template_kwargs


# ------------------------------------------------------------------ merge primitive


def test_no_server_default_is_a_passthrough():
    assert merge_server_ctk(None, {"a": 1}) == {"a": 1}
    assert merge_server_ctk({}, {"a": 1}) == {"a": 1}
    assert merge_server_ctk(None, None) == {}


def test_request_keys_win_over_server_defaults():
    assert merge_server_ctk({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}


def test_merge_does_not_mutate_either_input():
    defaults, ctk = {"a": 1}, {"b": 2}
    merge_server_ctk(defaults, ctk)
    assert defaults == {"a": 1} and ctk == {"b": 2}


# ------------------------------------------------------------- end-to-end precedence


def test_default_applies_when_the_request_says_nothing():
    """The whole point: a harness that sends no knob still gets thinking."""
    assert _ctk(THINKING_ON) == {"enable_thinking": True}


def test_without_the_flag_nothing_is_injected():
    assert _ctk(None) == {}


def test_an_explicit_request_opt_out_beats_a_thinking_on_server():
    assert _ctk(THINKING_ON, chat_template_kwargs={"enable_thinking": False}) == {
        "enable_thinking": False
    }


@pytest.mark.parametrize("effort", ["none", "off"])
def test_reasoning_effort_none_beats_a_thinking_on_server(effort):
    """Disabling must survive the merge, or a benchmark could not measure the
    non-thinking arm against a server started with the default on."""
    assert _ctk(THINKING_ON, reasoning_effort=effort)["enable_thinking"] is False


def test_thinking_disabled_beats_a_thinking_on_server():
    assert _ctk(THINKING_ON, thinking={"type": "disabled"})["enable_thinking"] is False


def test_a_server_default_does_not_swallow_the_requests_effort():
    """``effort_toggle_kwargs`` short-circuits when a thinking key is already present, so
    seeding the defaults BEFORE the effort mapping would drop ``reasoning_effort`` on the
    floor for graded templates. The merge must come after."""
    ctk = _ctk(THINKING_ON, reasoning_effort="high")
    assert ctk["reasoning_effort"] == "high"
    assert ctk["enable_thinking"] is True


def test_unrelated_server_defaults_ride_along_with_a_thinking_request():
    ctk = _ctk({"custom_flag": "x"}, reasoning_effort="high")
    assert ctk["custom_flag"] == "x"
    assert ctk["enable_thinking"] is True


# ------------------------------------------------------------------------- CLI parsing


def _parsed(argv: list[str]):
    from unittest.mock import patch

    from freetoken.server.args import parse_args

    class _Config:
        def to_dict(self):
            return {"architectures": ["LlamaForCausalLM"], "torch_dtype": "bfloat16"}

    with patch("freetoken.utils.cached_load_hf_config", lambda _p: _Config()):
        args, _ = parse_args(["--model", "/models/anon", *argv])
    return args


def test_flag_parses_a_json_object():
    args = _parsed(["--chat-template-kwargs", '{"enable_thinking": true}'])
    assert args.chat_template_kwargs == {"enable_thinking": True}


def test_flag_defaults_to_empty():
    assert _parsed([]).chat_template_kwargs == {}


@pytest.mark.parametrize("bad", ["not json", "[1,2]", '"a string"', "42"])
def test_flag_rejects_anything_but_a_json_object(bad):
    with pytest.raises(SystemExit):
        _parsed(["--chat-template-kwargs", bad])
