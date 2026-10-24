"""Speculation mode of the automatic FLE command, and the effective-context note.

``spec`` picks how the engine drafts ahead: auto (MTP only when its head and vocabulary file are there), none, mtp
(an error when the sidecars are missing), or a self-speculation mode that needs no extra file.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from fle_studio.config import DEFAULTS
from fle_studio.launch import PlanError, auto_plan
from fle_studio.qol import effective_context
from fle_studio.system import GiB

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
MTP = {"supported": True, "head": "/models/m-mtp.gguf", "vocab": "/models/m-vocab.bin"}
GPU = {"vendor": "nvidia", "index": 0, "name": "RTX", "memory_total": 16 * GiB, "memory_used": 0}


def plan(tmp_path: Path, *, spec: str | None = None, mtp: dict[str, Any] | None = None, parallel: int = 1) -> Any:
    settings = copy.deepcopy(DEFAULTS)
    if spec is not None:
        settings["launch"]["spec"] = spec
    settings["launch"]["parallel"] = parallel
    model = copy.deepcopy(MODEL)
    if mtp is not None:
        model["mtp"] = mtp
    sysinfo = {"cpu": {"physical": 8, "logical": 16}, "memory": {"total": 64 * GiB}, "gpus": [GPU]}
    engine = {"kind": "fle", "path": "/engines/llama-server", "backend": "cuda"}
    return auto_plan(model=model, engine=engine, settings=settings, sysinfo=sysinfo, port=8080, data_dir=tmp_path)


def test_auto_uses_mtp_when_the_sidecars_exist(tmp_path: Path) -> None:
    p = plan(tmp_path, mtp=MTP)
    assert "draft-mtp" in p.args
    assert p.env["FLE_MTP_VOCAB"] == MTP["vocab"]


def test_auto_explains_missing_sidecars_and_stays_off(tmp_path: Path) -> None:
    p = plan(tmp_path, mtp={"supported": True, "head": "", "vocab": ""})
    assert "--spec-type" not in p.args
    assert any("MTP is off" in n for n in p.notes)


def test_none_turns_speculation_off(tmp_path: Path) -> None:
    p = plan(tmp_path, spec="none", mtp=MTP)
    assert "--spec-type" not in p.args
    assert "-md" not in p.args
    assert any("Speculation is off" in n for n in p.notes)


def test_mtp_without_sidecars_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(PlanError):
        plan(tmp_path, spec="mtp", mtp={"supported": True, "head": "", "vocab": ""})


@pytest.mark.parametrize("mode", ["ngram-mod", "ngram-simple"])
def test_ngram_modes_need_no_sidecars(tmp_path: Path, mode: str) -> None:
    p = plan(tmp_path, spec=mode)
    assert p.args[p.args.index("--spec-type") + 1] == mode
    assert "-md" not in p.args


def test_unknown_mode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PlanError):
        plan(tmp_path, spec="eagle")


def test_parallel_slots_share_one_context(tmp_path: Path) -> None:
    p = plan(tmp_path, parallel=4)
    assert "--kv-unified" in p.args
    assert any("shared by all 4 slots" in n for n in p.notes)


def test_effective_context_separate_kv_splits_per_slot() -> None:
    e = effective_context(32768, 4, False)
    assert e["per_slot"] == 8192
    assert "separate KV" in e["note"]


def test_effective_context_single_slot_has_the_whole_context() -> None:
    assert effective_context(32768, 1, False)["per_slot"] == 32768
