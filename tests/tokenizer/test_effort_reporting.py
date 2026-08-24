"""What the effort probe *reports* versus what it actually serves.

``EffortProfile.supported`` is "the probe observed no rejection", which for a template
that ignores ``reasoning_effort`` entirely is the whole scale. ``effective_efforts`` is
the vocabulary requests are really served with, and it is empty for such a template. The
startup log used to print the former, so a checkpoint would advertise seven gears and
then report every one of them as unsupported on each request -- these tests pin the two
apart, and pin that dropping a grade never disturbs the thinking toggle.
"""

from __future__ import annotations

import pytest

from freetoken.tokenizer.effort import (
    KNOWN_REASONING_EFFORTS,
    EffortProfile,
    effective_efforts,
    probe_effort_profile,
    quantize_effort,
)

MESSAGES_TAIL = "<assistant>"


def _render_ignoring_effort(kwargs, tools):
    """A template that reads ``enable_thinking`` and nothing else -- Laguna's and
    GLM-4.7-Flash's shape. Undeclared Jinja variables are simply ignored."""
    return "<user>hi</user>" + ("<think>" if kwargs.get("enable_thinking") else "</think>")


def _render_grading_effort(kwargs, tools):
    """A template that interpolates the effort verbatim without validating it. The
    no-effort baseline renders a sentinel that is NOT a gear name, so no gear is
    detected as the template default."""
    return f"<user>hi</user>[effort={kwargs.get('reasoning_effort', '-')}]"


def _render_validating_effort(kwargs, tools):
    """A template that rejects everything outside its own ladder."""
    effort = kwargs.get("reasoning_effort")
    if effort is not None and effort not in ("low", "high"):
        raise ValueError(f"unsupported reasoning_effort {effort!r}")
    return f"<user>hi</user>[effort={effort}]"


def test_a_template_that_ignores_the_knob_reports_no_effective_vocabulary():
    profile = probe_effort_profile(_render_ignoring_effort)
    # Nothing was rejected, so the raw set is everything...
    assert profile.supported == frozenset(KNOWN_REASONING_EFFORTS)
    # ...but nothing was consumed either, so nothing is actually served.
    assert profile.consumes_effort is False
    assert effective_efforts(profile) == frozenset()


def test_the_raw_supported_set_is_not_what_gets_served():
    """The exact confusion the startup log had: supported is non-empty while the
    effective vocabulary is empty."""
    profile = probe_effort_profile(_render_ignoring_effort)
    assert profile.supported and not effective_efforts(profile)


@pytest.mark.parametrize("effort", ["high", "medium", "max", "xhigh"])
def test_every_effort_quantizes_to_nothing_when_the_template_ignores_it(effort):
    profile = probe_effort_profile(_render_ignoring_effort)
    assert quantize_effort(effort, profile) is None  # "send nothing"


def test_a_grading_template_does_serve_a_vocabulary():
    profile = probe_effort_profile(_render_grading_effort)
    assert profile.consumes_effort is True
    # Accepted everything without validating -> capped to the OpenAI triple.
    assert effective_efforts(profile) == frozenset({"low", "medium", "high"})
    assert quantize_effort("high", profile) == "high"


def test_a_validating_template_reports_its_real_ladder():
    profile = probe_effort_profile(_render_validating_effort)
    assert profile.validates is True
    assert effective_efforts(profile) == frozenset({"low", "high"})
    # "medium" (0.7) lands on "high" (0.9): 0.2 away, beyond the quantize distance.
    assert quantize_effort("medium", profile) is None
    assert quantize_effort("low", profile) == "low"


def test_dropping_the_grade_leaves_the_thinking_toggle_intact():
    """The reassurance behind the log wording: a harness sending reasoning_effort=high
    still gets thinking, because effort_toggle_kwargs sets enable_thinking and only the
    ungradeable ``reasoning_effort`` key is stripped."""
    from freetoken.server.model_meta import effort_toggle_kwargs

    ctk = effort_toggle_kwargs("high", None)
    assert ctk["enable_thinking"] is True
    assert ctk["reasoning_effort"] == "high"

    profile = probe_effort_profile(_render_ignoring_effort)
    sanitized = dict(ctk)
    if quantize_effort(sanitized["reasoning_effort"], profile) is None:
        del sanitized["reasoning_effort"]
    assert sanitized == {"enable_thinking": True, "thinking_mode": "enabled"}
    # ...and that is what actually opens the think block.
    assert _render_ignoring_effort(sanitized, None).endswith("<think>")


def test_an_empty_profile_advertises_nothing():
    profile = EffortProfile(supported=frozenset(), default=None, consumes_effort=False)
    assert effective_efforts(profile) == frozenset()
    assert quantize_effort("high", profile) is None
