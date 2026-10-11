"""Every part of the commands FLE Studio builds is explained in a sentence."""

from __future__ import annotations

import copy
from pathlib import Path

from fle_studio import flaghelp
from fle_studio.config import DEFAULTS
from fle_studio.launch import REFERENCE_ARGS, REFERENCE_ENV, auto_plan
from fle_studio.system import GiB

MODEL = {
    "id": "m",
    "name": "M",
    "path": "/models/m.gguf",
    "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
    "mtp": {"supported": True, "head": "/models/head.gguf", "vocab": "/models/vocab.txt"},
    "info": {
        "context_length": 32768,
        "gpu_bytes": 2 * GiB,
        "expert_bytes": 20 * GiB,
        "kv_bytes_per_token_q8": 1024,
        "moe": True,
    },
}
GPU = {"vendor": "nvidia", "index": 0, "name": "RTX", "memory_total": 16 * GiB, "memory_used": 0}
SYSINFO = {"cpu": {"physical": 8, "logical": 16}, "memory": {"total": 64 * GiB}, "gpus": [GPU]}


def test_the_reference_command_is_fully_explained() -> None:
    entries = flaghelp.explain(REFERENCE_ARGS, REFERENCE_ENV)
    unknown = [e["name"] for e in entries if not e["known"]]
    assert unknown == []
    by_name = {e["name"]: e for e in entries}
    assert by_name["-c"]["value"] == "262144"
    assert by_name["--jinja"]["value"] == ""  # a switch has no value
    assert by_name["-cmoed"]["value"] == ""
    assert by_name["-t"]["value"] == "16"
    assert by_name["FLE_MOE_CACHE_BUDGET_MIB"]["kind"] == "env"
    assert "expert cache" in by_name["FLE_MOE_CACHE_BUDGET_MIB"]["text"]
    assert all(e["text"] for e in entries)


def test_the_automatic_commands_are_fully_explained(tmp_path: Path) -> None:
    for kind in ("fle", "upstream"):
        engine = {"kind": kind, "path": "/engines/llama-server", "backend": "cuda"}
        plan = auto_plan(
            model=MODEL, engine=engine, settings=copy.deepcopy(DEFAULTS), sysinfo=SYSINFO, port=8080, data_dir=tmp_path
        )
        entries = flaghelp.explain(plan.args, plan.env)
        assert [e["name"] for e in entries if not e["known"]] == []
        assert any(e["name"] == "-m" for e in entries)


def test_aliases_and_inline_values() -> None:
    entries = flaghelp.explain(["--threads=6", "--ctx-size", "4096", "-ncmoe", "10", "--unknown-flag", "x"])
    assert [(e["name"], e["value"], e["known"]) for e in entries] == [
        ("--threads", "6", True),
        ("--ctx-size", "4096", True),
        ("-ncmoe", "10", True),
        ("--unknown-flag", "x", False),
    ]
    assert flaghelp.lookup("--threads")[0] == "-t"


def test_negative_numbers_are_values_not_flags() -> None:
    entries = flaghelp.explain(["--cache-ram", "-1", "-t", "4"])
    assert [(e["name"], e["value"]) for e in entries] == [("--cache-ram", "-1"), ("-t", "4")]


def test_the_tooltip_table_covers_all_aliases() -> None:
    table = flaghelp.table()
    assert table["flags"]["--threads"]["canonical"] == "-t"
    assert table["flags"]["-t"]["value"] is True
    assert table["flags"]["--jinja"]["value"] is False
    assert "FLE_MOE_CACHE_BUDGET_MIB" in table["env"]
