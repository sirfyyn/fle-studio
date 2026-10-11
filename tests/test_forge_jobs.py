from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from forge_synth import make_imatrix, make_model

from fle_studio.forge import ggufio, jobs, tools
from fle_studio.forge import recipe as recipes

FAKES = Path(__file__).parent / "fake_tools"
LAYERS = 4


@pytest.fixture
def calls(tmp_path: Path, monkeypatch) -> Path:
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_TOOLS_LOG", str(log))
    monkeypatch.delenv("FAKE_DELAY", raising=False)
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    return log


def calls_of(log: Path) -> list[dict]:
    return [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []


@pytest.fixture
def files(tmp_path: Path) -> dict[str, str]:
    (tmp_path / "corpus.txt").write_text("some calibration text\n" * 50, encoding="utf-8")
    (tmp_path / "eval.txt").write_text("some evaluation text\n" * 50, encoding="utf-8")
    types = [("Q4_0",) * 3] * LAYERS
    return {
        "full": str(make_model(tmp_path / "full.gguf", seed=1, layers=LAYERS, types=[("Q8_0",) * 3] * LAYERS)),
        "low": str(make_model(tmp_path / "low.gguf", seed=2, layers=LAYERS, types=types)),
        "corpus": str(tmp_path / "corpus.txt"),
        "eval": str(tmp_path / "eval.txt"),
    }


def make_jobs(tmp_path: Path, found: dict | None = None) -> jobs.ForgeJobs:
    found = tools.find([("settings", FAKES)], use_path=False) if found is None else found
    return jobs.ForgeJobs(tmp_path / "data", lambda: tmp_path / "out", lambda: found)


def wait(manager: jobs.ForgeJobs, ident: str, states=("done", "failed", "cancelled"), timeout: float = 60) -> dict:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        j = manager.get(ident)
        if j["state"] in states and not j["running"]:
            return j
        time.sleep(0.05)
    raise AssertionError(
        f"job still {manager.get(ident)['state']}: "
        + "\n".join(x["text"] for x in manager.log_since(ident, 0)["lines"][-15:])
    )


def quantize_recipe(files: dict[str, str], **extra) -> dict:
    r = {
        "name": "qtest",
        "mode": "quantize",
        "sources": {"S": {"path": files["full"], "role": "base"}},
        "quantize": {"source": "S", "type": "q4_0", "rules": [{"pattern": "ffn_down_exps", "type": "q8_0"}]},
        "imatrix": {"corpus": files["corpus"], "chunks": 3, "context": 128},
        "verify": {"enabled": True, "reference": "S", "text": files["eval"], "chunks": 3, "context": 128},
    }
    for key, value in extra.items():
        r[key] = {**r.get(key, {}), **value} if isinstance(value, dict) else value
    return r


def hybrid_recipe(files: dict[str, str], **extra) -> dict:
    r = {
        "name": "htest",
        "mode": "hybrid",
        "sources": {"B": {"path": files["full"], "role": "base"}, "L": {"path": files["low"], "role": "donor"}},
        "hybrid": {"base": "B", "layers": {"1": "L", "3": "L"}},
        "verify": {"enabled": False},
    }
    for key, value in extra.items():
        r[key] = {**r.get(key, {}), **value} if isinstance(value, dict) else value
    return r


def test_the_steps_follow_the_recipe():
    q = recipes.normalize({"mode": "quantize", "quantize": {"source": "S"}})
    assert jobs.plan_steps(q) == ["imatrix", "build", "kld_base", "kld", "finish"]
    assert jobs.needed_tools(q) == ["llama-imatrix", "llama-quantize", "llama-perplexity"]
    q = recipes.normalize(
        {"mode": "quantize", "quantize": {"source": "S", "imatrix": False}, "verify": {"base_logits": "/x/b.kld"}}
    )
    assert jobs.plan_steps(q) == ["build", "kld", "finish"]
    h = recipes.normalize({"mode": "hybrid", "verify": {"enabled": False}, "reap": {"enabled": True, "keep": 4}})
    assert jobs.plan_steps(h) == ["build", "reap_table", "reap_prune", "finish"] and jobs.needed_tools(h) == []
    h["reap"]["table"] = "/x/t.tsv"
    assert jobs.plan_steps(h) == ["build", "reap_prune", "finish"]


def test_check_names_what_is_missing(tmp_path: Path, files):
    m = make_jobs(tmp_path, {})
    info = m.check(quantize_recipe(files))
    assert not info["ok"] and info["missing_tools"] == ["llama-imatrix", "llama-quantize", "llama-perplexity"]
    with pytest.raises(ValueError, match="llama-quantize is missing"):
        m.create(quantize_recipe(files))
    ok = make_jobs(tmp_path).check(quantize_recipe(files))
    assert ok["ok"] and [s["id"] for s in ok["steps"]] == ["imatrix", "build", "kld_base", "kld", "finish"]
    bad = make_jobs(tmp_path).check(quantize_recipe(files, sources={"S": {"path": str(tmp_path / "gone.gguf")}}))
    assert not bad["ok"] and any("no file at" in p for p in bad["problems"])


def test_a_quantize_job_runs_all_steps_and_writes_the_result_files(tmp_path: Path, files, calls):
    m = make_jobs(tmp_path)
    started = m.create(quantize_recipe(files))
    done = wait(m, started["id"])
    assert done["state"] == "done", done["error"]
    assert [s["state"] for s in done["steps"]] == ["done"] * 5
    final = Path(done["result"]["model"])
    assert final.parent == tmp_path / "out" and final.name == "qtest.gguf" and final.is_file()
    assert not final.with_name("qtest.gguf.part").exists()
    assert done["result"]["sha256"] and len(done["result"]["sha256"]) == 64
    card = done["result"]["card"]
    assert card["complete"] and card["kld"] == pytest.approx(0.0123) and card["top1"] == pytest.approx(0.95123)
    shared = json.loads(final.with_name("qtest.recipe.json").read_text(encoding="utf-8"))
    assert shared["output"]["sha256"] == done["result"]["sha256"] and shared["sources"]["S"]["path"] == ""
    manifest = json.loads(final.with_name("qtest.forge.json").read_text(encoding="utf-8"))
    assert manifest["card"]["kld"] == card["kld"] and manifest["tools"]["llama-quantize"].startswith("9999")

    seen = {c["tool"]: c["argv"] for c in calls_of(calls) if c["argv"][:1] != ["--version"]}
    q = seen["quantize"]
    assert q[q.index("--imatrix") + 1].endswith("imatrix.gguf") and "--tensor-type" in q
    assert q[q.index("--tensor-type") + 1] == "ffn_down_exps=q8_0" and q[-1] == "q4_0"
    compare = [c["argv"] for c in calls_of(calls) if c["tool"] == "perplexity" and "--kl-divergence" in c["argv"]]
    assert len(compare) == 1
    base_run = [c["argv"] for c in calls_of(calls) if c["tool"] == "perplexity" and "--kl-divergence" not in c["argv"]]
    assert len(base_run) == 1 and "--kl-divergence-base" in base_run[0]

    text = "\n".join(x["text"] for x in m.log_since(started["id"], 0)["lines"])
    assert "== Importance matrix" in text and "$ " in text and "Done" in text


def test_a_hybrid_job_builds_without_any_tool(tmp_path: Path, files, calls):
    m = make_jobs(tmp_path, {})
    done = wait(m, m.create(hybrid_recipe(files))["id"])
    assert done["state"] == "done", done["error"]
    out = ggufio.open_model(done["result"]["model"])
    assert out.number("expert_count") == 8
    assert calls_of(calls) == []


def test_a_hybrid_that_takes_nothing_from_a_donor_fails_with_a_clear_message(tmp_path: Path, files):
    m = make_jobs(tmp_path, {})
    done = wait(m, m.create(hybrid_recipe(files, hybrid={"layers": {}}))["id"])
    assert done["state"] == "failed" and "copy of the base" in done["error"]
    assert done["steps"][0]["state"] == "failed"


def test_reap_runs_after_the_build_with_the_table_of_the_tool(tmp_path: Path, files, calls):
    m = make_jobs(tmp_path)
    r = hybrid_recipe(
        files,
        reap={"enabled": True, "keep": 4, "score": "freq", "chunks": 2},
        imatrix={"corpus": files["corpus"], "context": 128},
    )
    done = wait(m, m.create(r)["id"])
    assert done["state"] == "done", done["error"]
    assert [s["id"] for s in done["steps"]] == ["build", "reap_table", "reap_prune", "finish"]
    pruned = ggufio.open_model(done["result"]["model"])
    assert pruned.number("expert_count") == 4 and pruned.number("expert_used_count") == 2
    assert Path(done["result"]["model"] + ".reap.json").is_file()
    assert not list((tmp_path / "data" / "forge" / "jobs").glob("*/work/built.gguf"))
    assert any(c["tool"] == "fle-reap" for c in calls_of(calls))


def test_reap_without_the_reap_tool_derives_the_table_from_an_importance_matrix(tmp_path: Path, files):
    found = {n: t for n, t in tools.find([("settings", FAKES)], use_path=False).items() if n != "llama-fle-reap"}
    m = make_jobs(tmp_path, found)
    r = hybrid_recipe(files, reap={"enabled": True, "keep": 5}, imatrix={"file": str(tmp_path / "im.gguf")})
    first = wait(m, m.create(r)["id"])
    assert first["state"] == "failed" and "importance matrix" in first["error"]  # the file does not exist yet
    make_imatrix(tmp_path / "im.gguf", LAYERS, 8, {layer: [1, 5, 2, 9, 3, 8, 4, 7] for layer in range(LAYERS)})
    done = wait(m, m.resume(first["id"])["id"])
    assert done["state"] == "done", done["error"]
    assert ggufio.open_model(done["result"]["model"]).number("expert_count") == 5
    assert "deriving routing counts" in "\n".join(x["text"] for x in m.log_since(first["id"], 0)["lines"])


def test_cancel_stops_the_tool_and_resume_finishes_the_job(tmp_path: Path, files, calls, monkeypatch):
    monkeypatch.setenv("FAKE_DELAY", "0.4")
    m = make_jobs(tmp_path)
    r = quantize_recipe(files, verify={"enabled": False}, quantize={"imatrix": False})
    ident = m.create(r)["id"]
    end = time.monotonic() + 30
    while time.monotonic() < end and not any("quantizing" in x["text"] for x in m.log_since(ident, 0)["lines"]):
        time.sleep(0.05)
    t0 = time.monotonic()
    m.cancel(ident)
    done = wait(m, ident)
    assert done["state"] == "cancelled" and time.monotonic() - t0 < 20
    assert done["steps"][0]["state"] == "cancelled"
    assert not (tmp_path / "out" / "qtest.gguf").exists()
    with pytest.raises(RuntimeError, match="cannot be resumed"):
        m.resume(wait(m, m.create(hybrid_recipe(files))["id"])["id"])  # a finished job cannot be resumed
    monkeypatch.setenv("FAKE_DELAY", "0")
    again = wait(m, m.resume(ident)["id"])
    assert again["state"] == "done" and Path(again["result"]["model"]).is_file()


def test_a_failing_tool_fails_the_step_and_resume_repeats_only_that_step(tmp_path: Path, files, calls, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "quantize")
    m = make_jobs(tmp_path)
    ident = m.create(quantize_recipe(files, verify={"enabled": False}))["id"]
    failed = wait(m, ident)
    assert failed["state"] == "failed" and "code 3" in failed["error"]
    assert [s["state"] for s in failed["steps"]] == ["done", "failed", "pending"]
    monkeypatch.delenv("FAKE_FAIL")
    before = len([c for c in calls_of(calls) if c["tool"] == "imatrix"])
    done = wait(m, m.resume(ident)["id"])
    assert done["state"] == "done" and len([c for c in calls_of(calls) if c["tool"] == "imatrix"]) == before


def test_only_one_job_runs_at_a_time(tmp_path: Path, files, calls, monkeypatch):
    monkeypatch.setenv("FAKE_DELAY", "0.3")
    m = make_jobs(tmp_path)
    first = m.create(quantize_recipe(files, verify={"enabled": False}, quantize={"imatrix": False}))["id"]
    with pytest.raises(RuntimeError, match="Another Forge job"):
        m.create(hybrid_recipe(files))
    with pytest.raises(RuntimeError, match="Cancel the job"):
        m.delete(first)
    m.cancel(first)
    wait(m, first)
    m.delete(first)
    assert m.list() == []


def test_a_job_that_was_running_when_studio_closed_comes_back_interrupted(tmp_path: Path, files, calls):
    m = make_jobs(tmp_path)
    ident = m.create(quantize_recipe(files, verify={"enabled": False}), start=False)["id"]
    assert m.get(ident)["state"] == "queued"
    again = make_jobs(tmp_path)  # a new Studio process reads the same folder
    assert again.get(ident)["state"] == "interrupted" and again.list()[0]["id"] == ident
    assert "recipe" not in again.list()[0]
    assert any("Job " in x["text"] for x in again.log_since(ident, 0)["lines"])
    done = wait(again, again.resume(ident)["id"])
    assert done["state"] == "done"


def test_log_since_returns_only_new_lines(tmp_path: Path, files, calls):
    m = make_jobs(tmp_path)
    ident = wait(m, m.create(quantize_recipe(files, verify={"enabled": False}))["id"])["id"]
    everything = m.log_since(ident, 0)
    assert everything["lines"] and everything["next"] == everything["lines"][-1]["seq"]
    middle = everything["lines"][len(everything["lines"]) // 2]["seq"]
    rest = m.log_since(ident, middle)
    assert [x["seq"] for x in rest["lines"]] == list(range(middle + 1, everything["next"] + 1))
    assert m.log_since(ident, everything["next"])["lines"] == []
    with pytest.raises(KeyError):
        m.log_since("nope", 0)


def test_deleting_a_job_can_keep_or_remove_the_model(tmp_path: Path, files, calls):
    m = make_jobs(tmp_path)
    a = wait(m, m.create(quantize_recipe(files, verify={"enabled": False}))["id"])
    m.delete(a["id"])
    assert Path(a["result"]["model"]).is_file() and m.list() == []
    b = wait(m, m.create(quantize_recipe(files, verify={"enabled": False}))["id"])
    assert Path(b["result"]["model"]).name == "qtest-2.gguf"  # the name of the first result is taken
    m.delete(b["id"], keep_model=False)
    assert not Path(b["result"]["model"]).exists()
