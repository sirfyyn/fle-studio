"""The steps of a Forge job. Each takes the job context and returns the files it made (a dict of name -> value)."""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from fle_studio import __version__

from . import build, ggufio, reap, verify
from . import recipe as recipes
from .jobs import Ctx, StepError

PROGRESS = re.compile(r"^\[\s*(\d+)\s*/\s*(\d+)\s*\]")
CHUNKS_TOTAL = re.compile(r"computing over (\d+) chunks", re.I)
CHUNK_DONE = re.compile(r"\[(\d+)\]-?\d")
ETA = re.compile(r"ETA\s+([\d.]+)\s*(minutes|seconds|hours)", re.I)


def _path(ctx: Ctx, sid: str) -> str:
    return str(Path(ctx.recipe["sources"][sid]["path"]).expanduser())


def _need_file(path: str, what: str) -> str:
    if not path or not Path(path).expanduser().is_file():
        raise StepError(f"{what}: no file at {path!r}" if path else f"{what} is not set")
    return str(Path(path).expanduser())


def _free_check(folder: Path, need: int, what: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder).free
    if free < need + (1 << 30):
        raise StepError(f"{what} needs about {need / 1e9:.1f} GB in {folder}, but only {free / 1e9:.1f} GB are free")


def _threads(n: int) -> list[str]:
    return ["-t", str(n)] if n > 0 else []


def _model_path(ctx: Ctx) -> str:
    return _need_file(str(ctx.artifacts.get("model", "")), "the model built by the previous step")


# ------------------------------------------------------------------------------------------------ importance matrix


def run_imatrix(ctx: Ctx) -> dict[str, Any]:
    r = ctx.recipe
    im = r["imatrix"]
    tool = ctx.tool("llama-imatrix")
    model = _need_file(_path(ctx, r["quantize"]["source"]), "the source model")
    corpus = _need_file(im["corpus"], "the calibration text (imatrix.corpus)")
    out = ctx.work / "imatrix.gguf"  # the .gguf extension makes llama-imatrix write the format Forge can read back
    out.unlink(missing_ok=True)
    args = ["-m", model, "-f", corpus, "-o", str(out), "-c", str(im["context"])]
    if im["chunks"]:
        args += ["--chunks", str(im["chunks"])]
    if im["gpu_layers"]:
        args += ["-ngl", str(im["gpu_layers"])]
    args += _threads(im["threads"])
    state = {"total": im["chunks"] or 0, "done": 0}

    def on_line(line: str) -> None:
        m = CHUNKS_TOTAL.search(line)
        if m:
            state["total"] = int(m.group(1))
        done = [int(x) for x in CHUNK_DONE.findall(line)]
        if done:
            state["done"] = max(state["done"], *done)
        if state["total"] and state["done"]:
            eta = ETA.search(line)
            ctx.progress(
                state["done"] / state["total"],
                f"chunk {state['done']} of {state['total']}"
                + (f", about {eta.group(1)} {eta.group(2)} left" if eta else ""),
            )

    code = ctx.run(tool, args, on_line)
    if code != 0 or not out.is_file():
        raise StepError(f"llama-imatrix exited with code {code} and wrote no matrix (see the log)")
    return {"imatrix": str(out)}


# ------------------------------------------------------------------------------------------------ build


def run_build(ctx: Ctx) -> dict[str, Any]:
    r = ctx.recipe
    final = Path(ctx.artifacts["final"])
    out = ctx.work / "built.gguf" if r["reap"]["enabled"] else final
    if r["mode"] == "hybrid":
        return _hybrid(ctx, out)
    return _quantize(ctx, out)


def _hybrid(ctx: Ctx, out: Path) -> dict[str, Any]:
    from . import plan as planner

    r = ctx.recipe
    h = r["hybrid"]
    if not h["layers"] and not h["rules"]:
        raise StepError(
            "The recipe takes nothing from a donor, so the result would be a copy of the base. "
            "Choose layers on the Recipe step first."
        )
    for sid, s in r["sources"].items():
        if s["role"] != "reference":
            _need_file(s["path"], f"source {sid}")
    base = ggufio.open_model(_path(ctx, h["base"]))
    models = {h["base"]: base}
    for sid in {*h["layers"].values(), *(x["source"] for x in h["rules"])} - {h["base"]}:
        models[sid] = ggufio.open_model(_path(ctx, sid))
    size = planner.predicted_file_size(
        base,
        models,
        planner.assemble(
            base, models, {int(k): v for k, v in h["layers"].items()}, h["base"], planner.parse_rules(h["rules"])
        ),
    )
    ctx.log(f"The result will be {size / 1e9:.2f} GB")
    _free_check(out.parent, size, "The hybrid")
    result = build.build_hybrid(r, out, ctx.progress, ctx.cancelled, ctx.log)
    ctx.log(f"Wrote {out.name}: {result['size'] / 1e9:.2f} GB, SHA-256 {result['sha256'][:16]}...", "ok")
    return {"model": str(out), "model_sha256": result["sha256"], "model_size": result["size"]}


def _quantize(ctx: Ctx, out: Path) -> dict[str, Any]:
    r = ctx.recipe
    q = r["quantize"]
    tool = ctx.tool("llama-quantize")
    source = _need_file(_path(ctx, q["source"]), "the source model")
    part = out.with_name(out.name + ".part")
    part.unlink(missing_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    args: list[str] = []
    if q["imatrix"]:
        matrix = r["imatrix"]["file"] or str(ctx.artifacts.get("imatrix", ""))
        args += ["--imatrix", _need_file(matrix, "the importance matrix")]
    if q["allow_requantize"]:
        args.append("--allow-requantize")
    if q["output_type"]:
        args += ["--output-tensor-type", q["output_type"]]
    if q["embedding_type"]:
        args += ["--token-embedding-type", q["embedding_type"]]
    for rule in q["rules"]:
        args += ["--tensor-type", f"{rule['pattern']}={rule['type']}"]
    args += [source, str(part), q["type"]]
    threads = r["imatrix"]["threads"]
    if threads > 0:
        args.append(str(threads))
    t0 = time.monotonic()

    def on_line(line: str) -> None:
        m = PROGRESS.match(line)
        if m and int(m.group(2)):
            n, total = int(m.group(1)), int(m.group(2))
            spent = time.monotonic() - t0
            eta = f", about {spent * (total - n) / max(n, 1) / 60:.0f} min left" if n > 1 else ""
            ctx.progress(n / total, f"tensor {n} of {total}{eta}")

    try:
        code = ctx.run(tool, args, on_line)
        if code != 0 or not part.is_file():
            raise StepError(f"llama-quantize exited with code {code} (see the log)")
        part.replace(out)
    finally:
        if part.exists():
            part.unlink(missing_ok=True)
    return {"model": str(out), "model_size": out.stat().st_size}


# ------------------------------------------------------------------------------------------------ REAP


def run_reap_table(ctx: Ctx) -> dict[str, Any]:
    r = ctx.recipe
    rp = r["reap"]
    model = _model_path(ctx)
    out = ctx.work / "reap.tsv"
    tool = ctx.tools.get("llama-fle-reap")
    if tool and tool.ok:
        corpus = _need_file(r["imatrix"]["corpus"], "the calibration text (imatrix.corpus)")
        context = r["imatrix"]["context"]
        args = ["-m", model, "-f", corpus, "-c", str(context), "-b", str(context), "-ub", str(context)]
        if rp["chunks"]:
            args += ["--chunks", str(rp["chunks"])]
        args += _threads(rp["threads"])
        if rp["gpu_layers"]:
            args += ["-ngl", str(rp["gpu_layers"])]
            if tool.features.get("cmoe"):
                args.append("-cmoe")
            if tool.features.get("fit"):
                args += ["-fit", "off"]
        args += ["-o", str(out)]
        state = {"total": rp["chunks"] or 0, "done": 0}

        def on_line(line: str) -> None:
            m = CHUNKS_TOTAL.search(line)
            if m:
                state["total"] = int(m.group(1))
            done = [int(x) for x in CHUNK_DONE.findall(line)]
            if done:
                state["done"] = max(state["done"], *done)
            if state["total"] and state["done"]:
                ctx.progress(state["done"] / state["total"], f"chunk {state['done']} of {state['total']}")

        out.unlink(missing_ok=True)
        code = ctx.run(tool, args, on_line)
        if code != 0 or not out.is_file():
            raise StepError(f"llama-fle-reap exited with code {code} and wrote no table (see the log)")
        return {"reap_table": str(out)}
    matrix = r["imatrix"]["file"] or str(ctx.artifacts.get("imatrix", ""))
    if not matrix or not Path(matrix).is_file():
        raise StepError(
            "Pruning needs router statistics. Install an FLE engine (it ships llama-fle-reap), give a "
            "table of your own, or compute an importance matrix of this model first (imatrix.file)."
        )
    ctx.log(
        "llama-fle-reap was not found; deriving routing counts and input energy from the importance matrix "
        "(a weaker signal than the real REAP statistics)",
        "warn",
    )
    try:
        reap.write_table(reap.table_from_imatrix(matrix), out)
    except (reap.ReapError, ggufio.GGUFError) as exc:
        raise StepError(str(exc)) from None
    return {"reap_table": str(out)}


def run_reap_prune(ctx: Ctx) -> dict[str, Any]:
    r = ctx.recipe
    rp = r["reap"]
    model_path = _model_path(ctx)
    table_path = _need_file(rp["table"] or str(ctx.artifacts.get("reap_table", "")), "the router statistics table")
    final = Path(ctx.artifacts["final"])
    try:
        table = reap.load_table(table_path)
        model = ggufio.open_model(model_path)
        size = reap.predicted_prune_size(model, rp["keep"])
        ctx.log(f"The pruned model will be {size / 1e9:.2f} GB")
        _free_check(final.parent, size, "The pruned model")
        result = reap.prune(
            model_path, str(final), table, rp["keep"], rp["score"], ctx.progress, ctx.cancelled, ctx.log
        )
    except reap.ReapError as exc:
        raise StepError(str(exc)) from None
    built = ctx.work / "built.gguf"
    if built.is_file() and Path(model_path) == built:
        built.unlink()
    ctx.log(f"Wrote {final.name}: {result['size'] / 1e9:.2f} GB", "ok")
    return {
        "model": str(final),
        "model_sha256": result["sha256"],
        "model_size": result["size"],
        "reap_kept": result["keep"],
    }


# ------------------------------------------------------------------------------------------------ verify


def _perplexity_args(ctx: Ctx, tool: Any, model: str) -> list[str]:
    v = ctx.recipe["verify"]
    text = _need_file(v["text"], "the text for the comparison (verify.text)")
    args = ["-m", model, "-f", text, "-c", str(v["context"])]
    if v["chunks"]:
        args += ["--chunks", str(v["chunks"])]
    args += _threads(v["threads"])
    if v["gpu_layers"]:
        args += ["-ngl", str(v["gpu_layers"])]
    return args


def _chunk_progress(ctx: Ctx, total: int):
    seen = {"n": 0}

    def on_line(line: str) -> None:
        done = [int(x) for x in re.findall(r"\[(\d+)\]", line)]
        if done:
            seen["n"] = max(seen["n"], *done)
            if total:
                ctx.progress(min(seen["n"] / total, 0.99), f"chunk {seen['n']} of {total}")

    return on_line


def run_kld_base(ctx: Ctx) -> dict[str, Any]:
    v = ctx.recipe["verify"]
    tool = ctx.tool("llama-perplexity")
    reference = _need_file(_path(ctx, v["reference"]), "the reference model")
    out = ctx.work / "base.kld"
    out.unlink(missing_ok=True)
    args = [*_perplexity_args(ctx, tool, reference), "--kl-divergence-base", str(out)]
    ctx.log("The reference probabilities can take several GB of disk space for a long text.")
    code = ctx.run(tool, args, _chunk_progress(ctx, v["chunks"]))
    if code != 0 or not out.is_file():
        raise StepError(f"llama-perplexity exited with code {code} and stored no probabilities (see the log)")
    return {"base_logits": str(out)}


def _params(path: str) -> int:
    m = ggufio.open_model(path)
    return sum(t.n_elements for t in m.tensors.values())


def run_kld(ctx: Ctx) -> dict[str, Any]:
    r = ctx.recipe
    v = r["verify"]
    tool = ctx.tool("llama-perplexity")
    model = _model_path(ctx)
    base = _need_file(v["base_logits"] or str(ctx.artifacts.get("base_logits", "")), "the reference probabilities")
    args = [*_perplexity_args(ctx, tool, model), "--kl-divergence", "--kl-divergence-base", base]
    lines: list[str] = []
    progress = _chunk_progress(ctx, v["chunks"])

    def on_line(line: str) -> None:
        lines.append(line)
        progress(line)

    code = ctx.run(tool, args, on_line)
    stats = verify.parse("\n".join(lines))
    if code != 0 or not stats["complete"]:
        exit_note = f" (exit code {code})" if code else ""
        raise StepError("llama-perplexity printed no KL-divergence statistics" + exit_note + "; see the log")
    size = Path(model).stat().st_size
    ref = r["sources"].get(v["reference"], {}).get("file", "")
    card = verify.card(
        stats,
        name=r["name"],
        size=size,
        params=_params(model),
        reference=ref,
        text=Path(v["text"]).name,
        context=v["context"],
        chunks=v["chunks"],
        tool_version=tool.version,
    )
    (ctx.job.folder / "card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    ctx.job.data["result"]["card"] = card
    ctx.log(
        f"KL divergence {card['kld']:.5f}"
        + (f", top-1 agreement {card['top1'] * 100:.2f} %" if card["top1"] is not None else ""),
        "ok",
    )
    return {"kld": card["kld"]}


# ------------------------------------------------------------------------------------------------ finish


def run_finish(ctx: Ctx) -> dict[str, Any]:
    r = ctx.recipe
    final = Path(ctx.artifacts["final"])
    if not final.is_file():
        raise StepError(f"the result {final} is missing")
    sha = ctx.artifacts.get("model_sha256") if Path(str(ctx.artifacts.get("model", ""))) == final else None
    if not sha:
        ctx.log("Computing the SHA-256 of the result")
        size = final.stat().st_size

        sha = _hash(ctx, final, size)
    portable = recipes.portable(r)
    portable["output"] = {"file": final.name, "sha256": sha, "size": final.stat().st_size}
    recipe_file = final.with_name(final.stem + ".recipe.json")
    recipe_file.write_text(recipes.dumps(portable), encoding="utf-8")
    manifest = {
        "studio": __version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file": final.name,
        "size": final.stat().st_size,
        "sha256": sha,
        "name": r["name"],
        "mode": r["mode"],
        "sources": {k: {"file": s["file"], "size": s["size"]} for k, s in r["sources"].items()},
        "tools": {n: t.version for n, t in ctx.tools.items() if t.ok},
        "card": ctx.job.data["result"].get("card"),
    }
    manifest_file = final.with_name(final.stem + ".forge.json")
    manifest_file.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    ctx.job.data["result"].update(
        {
            "model": str(final),
            "size": manifest["size"],
            "sha256": sha,
            "recipe_file": str(recipe_file),
            "manifest": str(manifest_file),
        }
    )
    ctx.log(f"SHA-256 {sha}", "ok")
    ctx.log(f"Recipe saved as {recipe_file.name}", "ok")
    return {"final_sha256": sha, "recipe_file": str(recipe_file)}


def _hash(ctx: Ctx, path: Path, size: int) -> str:
    import hashlib

    h = hashlib.sha256()
    done = 0
    with open(path, "rb") as f:
        while True:
            data = f.read(ggufio.CHUNK)
            if not data:
                break
            ctx.check()
            h.update(data)
            done += len(data)
            ctx.progress(done / max(size, 1), f"{done / 1e9:.1f} of {size / 1e9:.1f} GB")
    return h.hexdigest()


RUNNERS = {
    "imatrix": run_imatrix,
    "build": run_build,
    "reap_table": run_reap_table,
    "reap_prune": run_reap_prune,
    "kld_base": run_kld_base,
    "kld": run_kld,
    "finish": run_finish,
}
