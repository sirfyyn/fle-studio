"""Quality next to speed: how far a quantization drifts from its reference, and whether it still knows things.

Two measurements, both on this machine:

* **KL divergence** against a reference model (usually the BF16 or a Q8 file of the same model), computed by
  ``llama-perplexity``: a first run with the reference saves its output probabilities for a short free text
  (``--kl-divergence-base``), a second run with the candidate compares against them (``--kl-divergence``). The mean KLD
  is the number the model cards of the FLE files quote; lower is better, 0 is identical. ``parse_kld`` reads the
  tool's report. This module stands on its own: the Forge page of this project reads the same report with its own
  parser, the two are to be merged later.
* **A knowledge probe**: a few dozen short factual questions sent to the running engine, answers checked against a
  list of accepted words. The built-in set is 24 facts in English and German written for this project (CC0). A harder
  set can be imported from MKQA on request (``import_mkqa``); that data is not part of this repository because its
  license (CC BY-SA 3.0 on the project page, CC BY 3.0 on the Hugging Face card) asks for attribution and share-alike.

Results are stored as JSON files in the data folder and shown on the Experiments page next to the speed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import threading
import time
import unicodedata
import urllib.request
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

from . import engine_tools, net, paths

KLD_CTX = 256  # tokens per chunk of the KL run; the built-in text has about 1,400 tokens, so about 5 chunks
KLD_TEXT = "kld_text.txt"
BUILTIN = "builtin"
MKQA_URL = "https://datasets-server.huggingface.co/rows"
MKQA_LICENSE = "CC BY-SA 3.0 (Apple ml-mkqa; the Hugging Face card says CC BY 3.0). Attribution and share-alike apply."
SYSTEM_PROMPT = "Answer with the shortest correct answer only: a name, a number or a few words. No explanation."


# ------------------------------------------------------------------------------------------------ KL divergence

_NUM = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
_PM = r"\s*(?:±|\+/-|\+-)\s*" + _NUM  # "±" in the tool's output, ASCII in some builds


def _first(text: str, pattern: str) -> tuple[float, float | None] | None:
    m = re.search(pattern, text)
    if not m:
        return None
    return float(m.group(1)), (float(m.group(2)) if m.lastindex and m.lastindex >= 2 and m.group(2) else None)


def parse_kld(text: str) -> dict[str, Any]:
    """Read the report of ``llama-perplexity --kl-divergence`` into numbers.

    Understands the "Perplexity statistics", "KL divergence statistics" and "Token probability statistics" blocks of
    llama.cpp (tools/perplexity). Keys that the build does not print are missing. Raises ValueError when there is no
    KL divergence in the text at all, which means the run did not finish."""
    out: dict[str, Any] = {}
    mean = _first(text, r"Mean\s+KLD:\s*" + _NUM + r"(?:" + _PM + r")?")
    if mean is None:
        raise ValueError("The output has no KL divergence statistics; the run did not finish.")
    out["mean_kld"], out["mean_kld_unc"] = mean
    for key, label in (
        ("max_kld", r"Maximum\s+KLD:"),
        ("kld_p999", r"99\.9%\s+KLD:"),
        ("kld_p99", r"99\.0%\s+KLD:"),
        ("kld_p95", r"95\.0%\s+KLD:"),
        ("kld_p90", r"90\.0%\s+KLD:"),
        ("median_kld", r"Median\s+KLD:"),
        ("kld_p10", r"10\.0%\s+KLD:"),
        ("kld_p5", r"5\.0%\s+KLD:"),
        ("kld_p1", r"1\.0%\s+KLD:"),
        ("kld_p01", r"0\.1%\s+KLD:"),
        ("min_kld", r"Minimum\s+KLD:"),
    ):
        v = _first(text, r"(?<![\d.])" + label + r"\s*" + _NUM)  # "5.0%" must not match inside "95.0%"
        if v:
            out[key] = v[0]
    for key, label in (("ppl_q", r"Mean\s+PPL\(Q\)\s*:"), ("ppl_base", r"Mean\s+PPL\(base\)\s*:")):
        v = _first(text, label + r"\s*" + _NUM + r"(?:" + _PM + r")?")
        if v:
            out[key], out[key + "_unc"] = v
    v = _first(text, r"Mean\s+PPL\(Q\)/PPL\(base\)\s*:\s*" + _NUM + r"(?:" + _PM + r")?")
    if v:
        out["ppl_ratio"], out["ppl_ratio_unc"] = v
    v = _first(text, r"Cor\(ln\(PPL\(Q\)\),\s*ln\(PPL\(base\)\)\):\s*" + _NUM + r"\s*%")
    if v:
        out["ppl_correlation"] = v[0]
    v = _first(text, r"Same\s+top\s+p:\s*" + _NUM + r"(?:" + _PM + r")?\s*%")
    if v:
        out["same_top_p"], out["same_top_p_unc"] = v
    v = _first(text, r"RMS\s+Δp\s*:\s*" + _NUM + r"(?:" + _PM + r")?\s*%")
    if v:
        out["rms_dp"], out["rms_dp_unc"] = v
    v = _first(text, r"Mean\s+Δp:\s*" + _NUM + r"(?:" + _PM + r")?\s*%")
    if v:
        out["mean_dp"], out["mean_dp_unc"] = v
    return out


def kld_text_path(data_dir: Path, custom: str = "") -> Path:
    """The text the KL run reads: the built-in one (copied into the data folder, so the tool gets a plain file) or a
    file of the user's own."""
    if custom.strip():
        p = Path(custom).expanduser()
        if not p.is_file():
            raise ValueError(f"No text file at {p}")
        return p
    dest = paths.ensure(data_dir / "quality") / KLD_TEXT
    data = (resources.files("fle_studio") / "quality_data" / KLD_TEXT).read_bytes()
    if not dest.exists() or dest.read_bytes() != data:
        dest.write_bytes(data)
    return dest


def base_file(data_dir: Path, reference: Path, text: Path, ctx: int, chunks: int) -> Path:
    """Where the reference run keeps its output. The name holds everything that changes it, so a run is reused only
    when model, text, chunk size and chunk count are the same."""
    st = reference.stat()
    digest = hashlib.sha256(text.read_bytes()).hexdigest()
    key = f"{reference.resolve()}|{st.st_size}|{int(st.st_mtime)}|{digest}|{ctx}|{chunks}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return paths.ensure(data_dir / "quality" / "base") / f"{reference.stem[:40]}-{digest}.kld"


def kld_args(
    model: str,
    text: Path,
    base: Path,
    ctx: int,
    chunks: int,
    *,
    reference: bool,
    threads: int = 0,
    gpu_layers: int | None = None,
) -> list[str]:
    args = ["-m", model, "-f", str(text), "--kl-divergence-base", str(base), "-c", str(ctx)]
    if not reference:
        args.append("--kl-divergence")
    if chunks > 0:
        args += ["--chunks", str(chunks)]
    if threads > 0:
        args += ["-t", str(threads)]
    if gpu_layers is not None:
        args += ["-ngl", str(gpu_layers)]
    return args


# ------------------------------------------------------------------------------------------------ knowledge probe


def normalize(text: str) -> str:
    """Lower case, accents and punctuation gone, one space between words: ``Zürich!`` and ``zurich`` are the same."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


def is_correct(answer: str, accepted: list[str]) -> bool:
    """True when the answer contains one of the accepted answers as whole words."""
    hay = f" {normalize(answer)} "
    return any(f" {n} " in hay for n in (normalize(a) for a in accepted) if n)


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % interval of a share (Wilson score): honest for small n and for shares near 0 or 1."""
    if n <= 0:
        return 0.0, 1.0
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load_questions(
    source: str = BUILTIN, data_dir: Path | None = None, lang: str = "", limit: int = 0
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(description, questions) of a question set.

    ``source``: "builtin", "mkqa" (the set imported to the data folder), or the path of a JSON file in the same
    format as quality_data/knowledge_cc0.json: {name, license, source, questions: [{id, lang, q, a: [...]}]}."""
    if source == BUILTIN:
        raw = json.loads((resources.files("fle_studio") / "quality_data" / "knowledge_cc0.json").read_text("utf-8"))
    elif source == "mkqa":
        if data_dir is None:
            raise ValueError("Import the MKQA questions first")
        file = data_dir / "quality" / "knowledge_mkqa.json"
        if not file.is_file():
            raise ValueError("Import the MKQA questions first")
        raw = json.loads(file.read_text("utf-8"))
    else:
        file = Path(source).expanduser()
        if not file.is_file():
            raise ValueError(f"No question file at {file}")
        raw = json.loads(file.read_text("utf-8"))
    questions = [q for q in raw.get("questions", []) if q.get("q") and q.get("a")]
    if lang:
        questions = [q for q in questions if q.get("lang") == lang]
    if limit > 0:
        questions = questions[:limit]
    if not questions:
        raise ValueError("The question set has no questions" + (f" in language {lang}" if lang else ""))
    meta = {k: raw.get(k, "") for k in ("name", "license", "source", "note")}
    meta["languages"] = sorted({q.get("lang", "") for q in questions})
    return meta, questions


def ask(base: str, headers: dict[str, str], question: str, seed: int = 0, timeout: float = 120) -> str:
    body = {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
        "max_tokens": 64,
        "temperature": 0,
        "seed": seed,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")


def run_probe(
    base: str,
    headers: dict[str, str],
    progress: Callable[[str], None] | None = None,
    *,
    source: str = BUILTIN,
    data_dir: Path | None = None,
    lang: str = "",
    limit: int = 0,
) -> dict[str, Any]:
    """Ask the running engine the questions one by one and score the answers.

    ``progress(text)`` is called before every question (and may raise InterruptedError to stop). The result keeps the
    share of correct answers with its 95 % interval, the share per language, and every question with the answer
    given, so a surprising number can be looked at."""
    meta, questions = load_questions(source, data_dir, lang, limit)
    items: list[dict[str, Any]] = []
    t0 = time.time()
    for i, q in enumerate(questions):
        if progress:
            progress(f"Knowledge probe {i + 1}/{len(questions)}")
        answer = ask(base, headers, q["q"], seed=i)
        items.append(
            {
                "id": q.get("id", ""),
                "lang": q.get("lang", ""),
                "q": q["q"],
                "expected": q["a"][0],
                "answer": answer.strip()[:200],
                "ok": is_correct(answer, q["a"]),
            }
        )
    correct = sum(1 for x in items if x["ok"])
    lo, hi = wilson(correct, len(items))
    by_lang = {}
    for code in meta["languages"]:
        sub = [x for x in items if x["lang"] == code]
        by_lang[code] = {
            "n": len(sub),
            "correct": sum(1 for x in sub if x["ok"]),
            "accuracy": round(sum(1 for x in sub if x["ok"]) / len(sub), 4),
        }
    return {
        "kind": "probe",
        "set": meta,
        "n": len(items),
        "correct": correct,
        "accuracy": round(correct / len(items), 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
        "by_lang": by_lang,
        "items": items,
        "seconds": round(time.time() - t0, 1),
    }


def compare_probes(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Is B's share of correct answers different from A's? Two Wilson intervals that overlap are "within noise"; this
    is conservative on purpose (a probe of 48 questions is a smoke test, not a benchmark)."""
    lo_a, hi_a = a["ci_lo"], a["ci_hi"]
    lo_b, hi_b = b["ci_lo"], b["ci_hi"]
    if lo_b > hi_a:
        verdict = "better"
    elif hi_b < lo_a:
        verdict = "worse"
    else:
        verdict = "within noise"
    return {"diff": round(b["accuracy"] - a["accuracy"], 4), "verdict": verdict}


def import_mkqa(
    data_dir: Path, lang: str = "en", count: int = 48, offset: int = 0, fetch: Callable[[str], Any] | None = None
) -> dict[str, Any]:
    """Download ``count`` questions of MKQA in ``lang`` (via the Hugging Face dataset viewer) into the data folder.

    Only questions with a short answer (an entity, a date or a number) are kept. The data stays in the user's data
    folder; it is not shipped with FLE Studio (see MKQA_LICENSE)."""
    if not re.fullmatch(r"[a-z]{2}(?:_[a-z]{2})?", lang):
        raise ValueError("lang must be a language code like en or de")
    get = fetch or (lambda url: net.get_json(url, timeout=60)[0])
    kept: list[dict[str, Any]] = []
    scanned = 0
    while len(kept) < count and scanned < 600:
        url = f"{MKQA_URL}?dataset=apple%2Fmkqa&config=mkqa&split=train&offset={offset + scanned}&length=100"
        rows = get(url).get("rows") or []
        if not rows:
            break
        scanned += len(rows)
        for entry in rows:
            row = entry.get("row") or {}
            queries, answers = row.get("queries") or {}, (row.get("answers") or {}).get(lang) or []
            text = queries.get(lang)
            accepted: list[str] = []
            for ans in answers:
                if ans.get("type") in ("entity", "date", "number", "number_with_unit", "short_phrase"):
                    accepted += [str(ans.get("text") or ""), *[str(x) for x in ans.get("aliases") or []]]
            accepted = [a for a in dict.fromkeys(accepted) if a.strip()]
            if text and accepted:
                kept.append({"id": str(row.get("example_id", "")), "lang": lang, "q": text, "a": accepted})
                if len(kept) >= count:
                    break
    if not kept:
        raise ValueError("The dataset viewer returned no usable MKQA questions")
    doc = {
        "name": f"MKQA ({lang}, {len(kept)} questions)",
        "license": MKQA_LICENSE,
        "source": "https://huggingface.co/datasets/apple/mkqa (Longpre, Lu, Daiber: MKQA, 2020)",
        "note": "Imported on request; not part of FLE Studio.",
        "questions": kept,
    }
    file = paths.ensure(data_dir / "quality") / "knowledge_mkqa.json"
    file.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"count": len(kept), "lang": lang, "path": str(file), "license": MKQA_LICENSE}


# ------------------------------------------------------------------------------------------------ the jobs


class Quality:
    """Runs KL divergence jobs and standalone probes in a thread, keeps their results in the data folder."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.dir = paths.ensure(data_dir / "quality")
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self.current: dict[str, Any] | None = None

    # ---- results

    def results(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.dir.glob("run-*.json"), reverse=True)[:200]:
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return out

    def delete(self, ident: str) -> None:
        for p in self.dir.glob(f"run-*-{ident}.json"):
            p.unlink()

    def status(self) -> dict[str, Any] | None:
        with self._lock:
            return json.loads(json.dumps(self.current)) if self.current else None

    def cancel(self) -> None:
        self._cancel.set()

    def _save(self, record: dict[str, Any]) -> None:
        name = time.strftime("run-%Y%m%d-%H%M%S", time.localtime(record["started"])) + f"-{record['id']}.json"
        (self.dir / name).write_text(json.dumps(record, indent=1), encoding="utf-8")

    def latest_kld_by_model(self) -> dict[str, dict[str, Any]]:
        """The newest finished KL run per candidate model name, for the column next to the speed."""
        out: dict[str, dict[str, Any]] = {}
        for r in reversed(self.results()):  # oldest first, so the newest wins
            if r.get("kind") == "kld" and r.get("state") == "done" and r.get("result"):
                out[r["candidate"]["name"]] = {
                    "id": r["id"],
                    "mean_kld": r["result"]["mean_kld"],
                    "reference": r["reference"]["name"],
                    "finished": r.get("finished"),
                }
        return out

    # ---- KL divergence

    def start_kld(
        self,
        *,
        reference: dict[str, Any],
        candidate: dict[str, Any],
        engine_path: str,
        text: str = "",
        ctx: int = KLD_CTX,
        chunks: int = 0,
        threads: int = 0,
        gpu_layers: int | None = None,
    ) -> dict[str, Any]:
        """Start the two llama-perplexity runs. ``reference`` and ``candidate`` are library models ({name, path})."""
        binary = engine_tools.find_tool(engine_path, "llama-perplexity")
        if binary is None:
            raise RuntimeError(
                "This engine has no llama-perplexity next to llama-server. Install a llama.cpp "
                "release (Engine page) or put llama-perplexity beside your own build."
            )
        if reference["path"] == candidate["path"]:
            raise ValueError("Pick two different models: the reference and the one to measure")
        text_file = kld_text_path(self.data_dir, text)
        ctx = max(64, min(8192, int(ctx)))
        chunks = max(0, min(1000, int(chunks)))
        with self._lock:
            if self.current and self.current["state"] == "running":
                raise RuntimeError("A quality measurement is already running")
            self._cancel.clear()
            self.current = {
                "id": secrets.token_hex(4),
                "kind": "kld",
                "state": "running",
                "started": time.time(),
                "step": "Starting",
                "reference": {"name": reference["name"], "path": reference["path"]},
                "candidate": {"name": candidate["name"], "path": candidate["path"]},
                "text": {"name": text_file.name, "chars": text_file.stat().st_size},
                "ctx": ctx,
                "chunks": chunks,
                "progress": 0.0,
                "commands": [],
            }
            cur = self.current
        threading.Thread(target=self._run_kld, args=(cur, binary, text_file, threads, gpu_layers), daemon=True).start()
        return dict(cur)

    def _run_kld(self, cur: dict[str, Any], binary: Path, text: Path, threads: int, gpu_layers: int | None) -> None:
        try:
            ref_path, cand_path = Path(cur["reference"]["path"]), Path(cur["candidate"]["path"])
            base = base_file(self.data_dir, ref_path, text, cur["ctx"], cur["chunks"])
            steps = [("candidate", cand_path, False)]
            if not base.is_file():
                steps.insert(0, ("reference", ref_path, True))
            for i, (what, model, is_reference) in enumerate(steps):
                args = kld_args(
                    str(model),
                    text,
                    base,
                    cur["ctx"],
                    cur["chunks"],
                    reference=is_reference,
                    threads=threads,
                    gpu_layers=gpu_layers,
                )
                with self._lock:
                    cur["step"] = f"Step {i + 1} of {len(steps)}: {what} model " + model.name
                    cur["commands"].append(" ".join([binary.name, *args]))
                    cur["progress"] = i / len(steps)

                def on_text(piece: str) -> None:
                    m = re.findall(r"\[(\d+)\]", piece)
                    if m:
                        with self._lock:
                            cur["chunk"] = int(m[-1])

                rc, output = engine_tools.run_tool(binary, args, cancel=self._cancel, on_text=on_text)
                if is_reference:
                    if rc != 0 or not base.is_file():
                        base.unlink(missing_ok=True)
                        raise RuntimeError(f"The reference run failed (exit code {rc}): {_tail(output)}")
                    continue
                try:
                    result = parse_kld(output)
                except ValueError as exc:
                    raise RuntimeError(f"{exc} Exit code {rc}: {_tail(output)}") from None
                with self._lock:
                    cur["result"] = result
            with self._lock:
                cur["state"], cur["progress"] = "done", 1.0
        except InterruptedError:
            with self._lock:
                cur["state"] = "cancelled"
        except Exception as exc:  # report, never crash the thread silently
            with self._lock:
                cur["state"], cur["error"] = "failed", str(exc)
        with self._lock:
            cur["finished"], cur["step"] = time.time(), ""
            record = json.loads(json.dumps(cur))
        self._save(record)

    # ---- standalone probe

    def start_probe(
        self,
        *,
        base: str,
        headers: dict[str, str],
        model: str,
        source: str = BUILTIN,
        lang: str = "",
        limit: int = 0,
        label: str = "",
    ) -> dict[str, Any]:
        load_questions(source, self.data_dir, lang, limit)  # fail early for a bad set
        with self._lock:
            if self.current and self.current["state"] == "running":
                raise RuntimeError("A quality measurement is already running")
            self._cancel.clear()
            self.current = {
                "id": secrets.token_hex(4),
                "kind": "probe",
                "state": "running",
                "started": time.time(),
                "step": "Starting",
                "candidate": {"name": model, "path": ""},
                "label": label[:80],
            }
            cur = self.current
        threading.Thread(target=self._run_probe, args=(cur, base, headers, source, lang, limit), daemon=True).start()
        return dict(cur)

    def _run_probe(
        self, cur: dict[str, Any], base: str, headers: dict[str, str], source: str, lang: str, limit: int
    ) -> None:
        def say(text: str) -> None:
            if self._cancel.is_set():
                raise InterruptedError
            with self._lock:
                cur["step"] = text

        try:
            result = run_probe(base, headers, say, source=source, data_dir=self.data_dir, lang=lang, limit=limit)
            with self._lock:
                cur["result"], cur["state"] = result, "done"
        except InterruptedError:
            with self._lock:
                cur["state"] = "cancelled"
        except Exception as exc:
            with self._lock:
                cur["state"], cur["error"] = "failed", str(exc)
        with self._lock:
            cur["finished"], cur["step"] = time.time(), ""
            record = json.loads(json.dumps(cur))
        self._save(record)


def _tail(output: str, lines: int = 3) -> str:
    rows = [x.strip() for x in output.strip().splitlines() if x.strip()]
    return " | ".join(rows[-lines:])[:400] or "no output"
