"""The shared speculation contract (tests/spec_cases.json) run against the real plan logic.

The same file lives in the helper's repository; each side runs it against its own implementation. A rule changes in
both copies and both implementations together.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from fle_studio.config import DEFAULTS
from fle_studio.launch import PlanError, auto_plan
from fle_studio.system import GiB

CASES = json.loads((Path(__file__).parent / "spec_cases.json").read_text(encoding="utf-8"))["cases"]

MODEL = {
    "id": "m",
    "name": "M",
    "path": "/models/m.gguf",
    "info": {
        "context_length": 32768,
        "gpu_bytes": 2 * GiB,
        "expert_bytes": 20 * GiB,
        "kv_bytes_per_token_q8": 1024,
        "moe": True,
    },
}
SIDECARS = {"supported": True, "head": "/models/m-mtp.gguf", "vocab": "/models/m-vocab.bin"}
GPU = {"vendor": "nvidia", "index": 0, "name": "RTX", "memory_total": 16 * GiB, "memory_used": 0}


def run_case(case: dict[str, Any], tmp_path: Path) -> Any:
    settings = copy.deepcopy(DEFAULTS)
    settings["launch"]["spec"] = case["spec"]
    settings["launch"]["mtp"] = case["mtp_enabled"]
    model = copy.deepcopy(MODEL)
    model["mtp"] = dict(SIDECARS) if case["sidecars"] else {"supported": True, "head": "", "vocab": ""}
    engine = {"kind": "fle", "path": "/engines/llama-server", "backend": "cuda", "spec_type": case["engine_spec_type"]}
    sysinfo = {"cpu": {"physical": 8, "logical": 16}, "memory": {"total": 64 * GiB}, "gpus": [GPU]}
    return auto_plan(model=model, engine=engine, settings=settings, sysinfo=sysinfo, port=8080, data_dir=tmp_path)


def effective(args: list[str]) -> str:
    if "--spec-type" not in args:
        return "none"
    kind = args[args.index("--spec-type") + 1]
    return "mtp" if kind == "draft-mtp" else kind


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_contract_case(case: dict[str, Any], tmp_path: Path) -> None:
    expect = case["expect"]
    if "error" in expect:
        with pytest.raises(PlanError) as err:
            run_case(case, tmp_path)
        assert expect["error"] in str(err.value).lower()
    else:
        assert effective(run_case(case, tmp_path).args) == expect["effective"]


def test_contract_covers_every_mode() -> None:
    assert {c["spec"] for c in CASES} >= {"auto", "none", "mtp", "ngram-mod", "ngram-simple"}
