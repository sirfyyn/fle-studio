"""The chip row above the notes: planview.summary builds it from a plan's public form."""

from __future__ import annotations

from typing import Any

from fle_studio import planview

FLE_MTP = {
    "engine": "fle",
    "profile": "auto",
    "args": ["-m", "/m.gguf", "-c", "32768", "--jinja", "-md", "/h.gguf", "--spec-type", "draft-mtp"],
    "fallback": "",
}
HAVE = {"mtp": {"supported": True, "head": "/h.gguf", "vocab": "/v.txt"}}
LACK = {"mtp": {"supported": True}}


def chips(plan: dict[str, Any], **kw: Any) -> dict[str, tuple[str, str]]:
    return {c["label"]: (c["value"], c["tone"]) for c in planview.summary(plan, **kw)}


def test_the_chips_have_label_value_and_a_known_tone() -> None:
    for chip in planview.summary(FLE_MTP, model=HAVE):
        assert set(chip) == {"label", "value", "tone"}
        assert chip["tone"] in ("ok", "warn", "info")


def test_an_fle_plan_with_mtp() -> None:
    got = chips(FLE_MTP, model=HAVE)
    assert got["Engine"] == ("FLE", "info")
    assert got["Speculation"] == ("mtp", "ok")
    assert got["MTP files"] == ("present", "ok")
    assert got["Context"] == ("32,768 tokens", "info")
    assert got["Per slot"] == ("32,768", "info") and got["Slots"] == ("1", "info")
    assert got["Fallback"] == ("no", "ok")


def test_missing_mtp_files_warn_only_when_mtp_would_be_used() -> None:
    plan = {**FLE_MTP, "args": ["-m", "/m.gguf", "-c", "8192"]}
    assert chips(plan, model=LACK)["MTP files"] == ("missing", "warn")
    assert chips(plan, model=LACK, spec_requested="none")["MTP files"][1] == "info"
    assert chips(plan, model=LACK, mtp_on=False)["MTP files"][1] == "info"
    assert "MTP files" not in chips(plan, model={"mtp": {"supported": False}})
    assert "MTP files" not in chips(plan, model=None)


def test_speculation_is_what_the_command_does_not_what_was_asked() -> None:
    ngram = {**FLE_MTP, "args": ["-c", "4096", "--spec-type", "ngram-mod"]}
    assert chips(ngram, model=None, spec_requested="ngram-mod")["Speculation"] == ("ngram-mod", "ok")
    off = {**FLE_MTP, "args": ["-c", "4096", "--spec-type=none"]}
    assert chips(off, model=None)["Speculation"] == ("none", "info")
    # asked for ngram, but the plan carries no --spec-type (an engine that cannot): a warning, not a quiet "none"
    unsupported = {**FLE_MTP, "args": ["-c", "4096"]}
    assert chips(unsupported, model=None, spec_requested="ngram-mod")["Speculation"] == ("none (not as asked)", "warn")
    # a profile's own command decides: no complaint
    profile = {**unsupported, "profile": "mine"}
    assert chips(profile, model=None, spec_requested="ngram-mod")["Speculation"] == ("none", "info")


def test_slots_share_a_unified_cache() -> None:
    plan = {**FLE_MTP, "args": ["-c", "32768", "--parallel", "4", "--kv-unified"]}
    got = chips(plan, model=None)
    assert got["Slots"][0] == "4" and got["Context"][0] == "32,768 tokens"
    assert got["Per slot"][0] == "up to 32,768 (shared)"


def test_slots_with_separate_caches_split_the_context() -> None:
    plan = {**FLE_MTP, "args": ["-c=32768", "-np", "4"]}
    got = chips(plan, model=None)
    assert got["Per slot"][0] == "8,192" and got["Slots"][0] == "4"


def test_a_command_without_a_context_has_no_context_chips() -> None:
    got = chips({**FLE_MTP, "args": ["-m", "/m.gguf"]}, model=None)
    assert "Context" not in got and "Slots" not in got and "Per slot" not in got


def test_an_upstream_fallback_is_a_warning() -> None:
    plan = {
        "engine": "upstream",
        "profile": "auto",
        "args": ["-c", "4096"],
        "fallback": "The FLE plan could not be built.",
    }
    got = chips(plan, model=None)
    assert got["Engine"] == ("Upstream llama.cpp", "warn")
    assert got["Fallback"] == ("yes", "warn")
    assert chips({**plan, "fallback": ""}, model=None)["Engine"] == ("Upstream llama.cpp", "info")


def test_an_own_build_is_named_by_what_its_command_shows() -> None:
    assert chips({"engine": "custom", "args": []}, model=None)["Engine"][0] == "llama.cpp (own build)"
    fle_flags = {"engine": "custom", "args": ["--moe-expert-cache-inserts", "1"]}
    assert chips(fle_flags, model=None)["Engine"][0] == "FLE (own build)"
    fle_env = {"engine": "custom", "args": [], "env": {"FLE_MOE_CACHE_BUDGET_MIB": "1024"}}
    assert chips(fle_env, model=None)["Engine"][0] == "FLE (own build)"
