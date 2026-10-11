"""Quality measurements: the KL divergence report, the knowledge probe and the jobs that run them."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
from helpers import FAKE_SERVER, wait_for

from fle_studio import quality

# A report in the shape llama.cpp prints (tools/perplexity, kl_divergence()).
REPORT = """
chunk             PPL       ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p
   1        5.4321        0.01234        0.056789      4.567 %    92.345 %

====== Perplexity statistics ======
Mean PPL(Q)                   :   5.432100 ±   0.123400
Mean PPL(base)                :   5.300000 ±   0.120000
Cor(ln(PPL(Q)), ln(PPL(base))):  98.76%
Mean ln(PPL(Q)/PPL(base))     :   0.024600 ±   0.003200
Mean PPL(Q)/PPL(base)         :   1.024905 ±   0.003300
Mean PPL(Q)-PPL(base)         :   0.132100 ±   0.017000

====== KL divergence statistics ======
Mean    KLD:   0.051234 ±   0.001234
Maximum KLD:   4.250000
99.9%   KLD:   2.100000
99.0%   KLD:   0.900000
95.0%   KLD:   0.250000
90.0%   KLD:   0.120000
Median  KLD:   0.010000
10.0%   KLD:   0.000500
 5.0%   KLD:   0.000200
 1.0%   KLD:   0.000030
 0.1%   KLD:   0.000002
Minimum KLD:  -0.000010

====== Token probability statistics ======
Mean    Δp: -0.512 ± 0.034 %
Maximum Δp: 62.100%
99.9%   Δp: 40.000%
Median  Δp: -0.001%
Minimum Δp: -70.000%

RMS Δp    :  6.789 ± 0.123 %
Same top p: 91.234 ± 0.321 %
"""


def test_parse_kld_reads_the_report() -> None:
    r = quality.parse_kld(REPORT)
    assert r["mean_kld"] == 0.051234 and r["mean_kld_unc"] == 0.001234
    assert r["max_kld"] == 4.25 and r["median_kld"] == 0.01 and r["kld_p99"] == 0.9 and r["kld_p95"] == 0.25
    assert r["kld_p999"] == 2.1 and r["kld_p90"] == 0.12 and r["kld_p10"] == 0.0005
    assert r["kld_p5"] == 0.0002 and r["kld_p1"] == 0.00003 and r["kld_p01"] == 0.000002
    assert r["min_kld"] == -0.00001
    assert r["ppl_q"] == 5.4321 and r["ppl_base"] == 5.3 and r["ppl_ratio"] == 1.024905
    assert r["ppl_correlation"] == 98.76
    assert r["same_top_p"] == 91.234 and r["same_top_p_unc"] == 0.321 and r["rms_dp"] == 6.789
    assert r["mean_dp"] == -0.512


def test_parse_kld_takes_the_ascii_plus_minus_and_missing_blocks() -> None:
    r = quality.parse_kld("Mean    KLD:   0.010000 +/-   0.000100\nMaximum KLD:   1.5\n")
    assert r == {"mean_kld": 0.01, "mean_kld_unc": 0.0001, "max_kld": 1.5}


def test_parse_kld_refuses_an_unfinished_run() -> None:
    with pytest.raises(ValueError, match="did not finish"):
        quality.parse_kld("perplexity: calculating perplexity over 4 chunks\n[1]5.1,[2]5.2,")


def test_answers_are_matched_by_whole_words_without_accents_or_case() -> None:
    assert quality.is_correct("Canberra.", ["Canberra"])
    assert quality.is_correct("The capital is CANBERRA, I think", ["canberra"])
    assert quality.is_correct("Zürich", ["Zurich"])
    assert quality.is_correct("Es ist Rom.", ["Rom"])
    assert not quality.is_correct("Rome was not built in a day", ["Rom"])  # a part of a word does not count
    assert not quality.is_correct("1989 was a year", ["19"])
    assert quality.is_correct("It boils at 100 °C", ["100"])
    assert not quality.is_correct("", ["x"])
    assert not quality.is_correct("anything", [""])


def test_wilson_interval() -> None:
    lo, hi = quality.wilson(48, 48)
    assert 0.92 < lo < 0.94 and hi == pytest.approx(1.0)
    lo, hi = quality.wilson(24, 48)
    assert 0.36 < lo < 0.37 and 0.63 < hi < 0.64
    assert quality.wilson(0, 0) == (0.0, 1.0)


def test_the_built_in_questions() -> None:
    meta, questions = quality.load_questions()
    assert meta["license"] == "CC0-1.0"
    assert len(questions) == 48 and meta["languages"] == ["de", "en"]
    assert all(q["q"] and q["a"] for q in questions)
    en = quality.load_questions(lang="en")[1]
    assert len(en) == 24
    assert len(quality.load_questions(limit=10)[1]) == 10
    ids = {}
    for q in questions:
        ids.setdefault(q["id"], set()).add(q["lang"])
    assert all(langs == {"en", "de"} for langs in ids.values())  # every fact in both languages


def test_a_missing_or_unknown_set_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Import the MKQA"):
        quality.load_questions("mkqa", tmp_path)
    with pytest.raises(ValueError, match="No question file"):
        quality.load_questions(str(tmp_path / "none.json"))
    with pytest.raises(ValueError, match="no questions in language fr"):
        quality.load_questions(lang="fr")


class QaServer:
    """An engine that answers the built-in questions: right for the first `right` of them, wrong for the rest."""

    def __init__(self, right: int) -> None:
        answers = {q["q"]: q["a"][0] for q in quality.load_questions()[1]}
        order = {q["q"]: i for i, q in enumerate(quality.load_questions()[1])}
        self.asked: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.asked.append(body)
                question = body["messages"][-1]["content"]
                text = answers[question] if order[question] < right else "I do not know."
                data = json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_the_probe_scores_the_answers() -> None:
    qa = QaServer(right=36)
    try:
        steps: list[str] = []
        result = quality.run_probe(qa.base, {}, steps.append)
    finally:
        qa.close()
    assert result["n"] == 48 and result["correct"] == 36 and result["accuracy"] == 0.75
    assert result["ci_lo"] < 0.75 < result["ci_hi"]
    assert result["set"]["license"] == "CC0-1.0"
    assert steps[0] == "Knowledge probe 1/48" and len(steps) == 48
    assert sum(1 for x in result["items"] if x["ok"]) == 36
    assert result["items"][0]["expected"] == "Canberra"
    asked = qa.asked[0]
    assert asked["temperature"] == 0 and asked["stream"] is False
    assert asked["chat_template_kwargs"] == {"enable_thinking": False}
    assert set(result["by_lang"]) == {"de", "en"} and result["by_lang"]["en"]["n"] == 24


def test_the_probe_can_be_cancelled_between_questions() -> None:
    qa = QaServer(right=48)

    def stop_after_three(text: str) -> None:
        if text.endswith("4/48"):
            raise InterruptedError

    try:
        with pytest.raises(InterruptedError):
            quality.run_probe(qa.base, {}, stop_after_three)
    finally:
        qa.close()
    assert len(qa.asked) == 3


def test_two_probes_are_compared_by_their_intervals() -> None:
    qa = QaServer(right=48)
    qb = QaServer(right=12)
    try:
        good, bad = quality.run_probe(qa.base, {}), quality.run_probe(qb.base, {})
    finally:
        qa.close()
        qb.close()
    assert quality.compare_probes(good, bad)["verdict"] == "worse"
    assert quality.compare_probes(bad, good)["verdict"] == "better"
    assert quality.compare_probes(good, good)["verdict"] == "within noise"


def test_import_mkqa_keeps_short_answers_only(tmp_path: Path) -> None:
    def row(i: int, kind: str, text: str, aliases: list[str]) -> dict[str, Any]:
        return {
            "row": {
                "example_id": str(i),
                "queries": {"en": f"question {i}?", "de": f"Frage {i}?"},
                "answers": {
                    "en": [{"type": kind, "text": text, "aliases": aliases}],
                    "de": [{"type": kind, "text": text + " de", "aliases": []}],
                },
            }
        }

    rows = [
        row(1, "entity", "Dave Edmunds", ["Edmunds"]),
        row(2, "long_answer", "a long passage", []),
        row(3, "unanswerable", "", []),
        row(4, "date", "1989", []),
        row(5, "binary", "yes", []),
    ]
    calls: list[str] = []

    def fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        return {"rows": rows if len(calls) == 1 else []}

    info = quality.import_mkqa(tmp_path, "en", 10, fetch=fetch)
    assert info["count"] == 2 and "CC BY-SA 3.0" in info["license"]
    assert "apple%2Fmkqa" in calls[0]
    meta, questions = quality.load_questions("mkqa", tmp_path)
    assert [q["q"] for q in questions] == ["question 1?", "question 4?"]
    assert questions[0]["a"] == ["Dave Edmunds", "Edmunds"]
    assert "CC BY-SA" in meta["license"]
    with pytest.raises(ValueError, match="language code"):
        quality.import_mkqa(tmp_path, "../x", fetch=fetch)


# ---- the KL divergence jobs, against the stand-in llama-perplexity


@pytest.fixture
def models(tmp_path: Path) -> dict[str, dict[str, str]]:
    out = {}
    for name in ("Q8_0", "IQ3_M", "IQ2_M"):
        p = tmp_path / "models" / f"stub-{name}.gguf"
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"GGUF" + b"\0" * 64)
        out[name] = {"name": f"stub-{name}", "path": str(p)}
    return out


def finished(q: quality.Quality, timeout: float = 60) -> dict[str, Any]:
    return wait_for(lambda: (s := q.status()) and s["state"] != "running" and s, timeout, "the job to finish")


def test_a_kl_run_measures_the_candidate_against_the_reference(tmp_path: Path, models: dict[str, Any]) -> None:
    q = quality.Quality(tmp_path / "data")
    q.start_kld(reference=models["Q8_0"], candidate=models["IQ3_M"], engine_path=str(FAKE_SERVER))
    job = finished(q)
    assert job["state"] == "done", job.get("error")
    result = job["result"]
    assert 0 < result["mean_kld"] < 0.3 and result["same_top_p"] > 50
    assert len(job["commands"]) == 2 and "--kl-divergence" not in job["commands"][0].split()
    assert "--kl-divergence" in job["commands"][1].split()
    assert job["text"]["name"] == "kld_text.txt"
    # a worse quantization drifts further; the reference run is not repeated
    q.start_kld(reference=models["Q8_0"], candidate=models["IQ2_M"], engine_path=str(FAKE_SERVER))
    worse = finished(q)
    assert worse["state"] == "done" and len(worse["commands"]) == 1
    assert worse["result"]["mean_kld"] > result["mean_kld"]
    assert [r["id"] for r in q.results()] == [worse["id"], job["id"]]
    by_model = q.latest_kld_by_model()
    assert by_model["stub-IQ2_M"]["mean_kld"] == worse["result"]["mean_kld"]
    assert by_model["stub-IQ3_M"]["reference"] == "stub-Q8_0"
    q.delete(worse["id"])
    assert [r["id"] for r in q.results()] == [job["id"]]


def test_a_failing_reference_run_is_reported_and_not_cached(
    tmp_path: Path, models: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    q = quality.Quality(tmp_path / "data")
    monkeypatch.setenv("FAKE_PPL_FAIL", "1")
    q.start_kld(reference=models["Q8_0"], candidate=models["IQ3_M"], engine_path=str(FAKE_SERVER))
    job = finished(q)
    assert job["state"] == "failed" and "reference run failed" in job["error"]
    monkeypatch.delenv("FAKE_PPL_FAIL")
    q.start_kld(reference=models["Q8_0"], candidate=models["IQ3_M"], engine_path=str(FAKE_SERVER))
    assert finished(q)["state"] == "done"


def test_a_kl_run_can_be_cancelled(tmp_path: Path, models: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    q = quality.Quality(tmp_path / "data")
    monkeypatch.setenv("FAKE_PPL_CHUNK_SECONDS", "5")
    q.start_kld(reference=models["Q8_0"], candidate=models["IQ3_M"], engine_path=str(FAKE_SERVER))
    wait_for(lambda: (q.status() or {}).get("commands"), 10, "the first tool to start")
    t0 = time.monotonic()
    q.cancel()
    job = finished(q, 15)
    assert job["state"] == "cancelled" and time.monotonic() - t0 < 10


def test_kl_start_refusals(tmp_path: Path, models: dict[str, Any]) -> None:
    q = quality.Quality(tmp_path / "data")
    with pytest.raises(RuntimeError, match="no llama-perplexity"):
        q.start_kld(reference=models["Q8_0"], candidate=models["IQ3_M"], engine_path=str(tmp_path / "llama-server"))
    with pytest.raises(ValueError, match="two different models"):
        q.start_kld(reference=models["Q8_0"], candidate=models["Q8_0"], engine_path=str(FAKE_SERVER))
    with pytest.raises(ValueError, match="No text file"):
        q.start_kld(
            reference=models["Q8_0"],
            candidate=models["IQ3_M"],
            engine_path=str(FAKE_SERVER),
            text=str(tmp_path / "nope.txt"),
        )


def test_a_standalone_probe_job_is_stored(tmp_path: Path) -> None:
    qa = QaServer(right=40)
    q = quality.Quality(tmp_path / "data")
    try:
        q.start_probe(base=qa.base, headers={}, model="stub")
        job = finished(q)
    finally:
        qa.close()
    assert job["state"] == "done" and job["result"]["correct"] == 40
    assert q.results()[0]["kind"] == "probe"
