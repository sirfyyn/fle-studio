"""What the server calls: tool discovery, source inspection, the recipe calculator, jobs, and publishing a result."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import paths
from . import corpus, ggufio, plan, reap, tools, verify
from . import recipe as recipes
from .jobs import ForgeJobs

GB = plan.GB


class Forge:
    def __init__(
        self,
        data_dir: Path,
        setting: Callable[[str], Any],
        models_dir: Callable[[], Path],
        engine_paths: Callable[[], list[tuple[str, str]]],
        library: Any,
        profiles: Any,
    ) -> None:
        self.data_dir = data_dir
        self.setting = setting
        self.models_dir = models_dir
        self.engine_paths = engine_paths
        self.library = library
        self.profiles = profiles
        self._tools: tuple[float, dict[str, tools.Tool]] = (0.0, {})
        self.jobs = ForgeJobs(data_dir, lambda: Path(self.models_dir()) / "forge", self.tools)

    # ------------------------------------------------------------------------------------------ tools

    def tools(self, refresh: bool = False) -> dict[str, tools.Tool]:
        at, found = self._tools
        if refresh or time.time() - at > 10 or not found:
            dirs = tools.search_dirs(str(self.setting("forge.tools_dir") or ""), self.engine_paths())
            found = tools.find(dirs)
            self._tools = (time.time(), found)
        return found

    def overview(self) -> dict[str, Any]:
        found = self.tools(refresh=True)
        return {
            "tools": [
                found[n].public()
                if n in found
                else {
                    "name": n,
                    "ok": False,
                    "path": "",
                    "source": "",
                    "version": "",
                    "features": {},
                    "problem": "not found",
                }
                for n in tools.TOOL_NAMES
            ],
            "tools_dir": str(self.setting("forge.tools_dir") or ""),
            "jobs": self.jobs.list(),
            "defaults": recipes.normalize({}),
            "types": list(ggufio.QUANTIZE_TARGETS),
            "scores": list(reap.SCORES),
            "output_dir": str(Path(self.models_dir()) / "forge"),
            "references": verify.reference_cards(),
        }

    # ------------------------------------------------------------------------------------------ sources and plan

    def inspect(self, path: str) -> dict[str, Any]:
        p = Path(path.strip().strip('"')).expanduser()
        if not p.is_file():
            raise ValueError(f"No file at {p}")
        return plan.Source.load("S", str(p)).describe()

    def calculate(self, body: dict[str, Any]) -> dict[str, Any]:
        """The calculator: the layer choice for a budget (hybrid), or the size estimate for a quantisation."""
        mode = str(body.get("mode") or "hybrid")
        ram = int(float(body.get("ram_gb") or 0) * GB)
        vram = int(float(body.get("vram_gb") or 0) * GB)
        context = int(body.get("context") or 32768)
        if mode == "quantize":
            q = recipes.normalize({"quantize": body.get("quantize") or {}})["quantize"]
            model = ggufio.open_model(self._file(body.get("source_path")))
            est = plan.estimate_quantize(model, q["type"], q["rules"], q["output_type"], q["embedding_type"])
            return {"mode": mode, "estimate": est, "model": plan.Source.from_model("S", model).describe()}
        sources = {sid: plan.Source.load(sid, self._file(path)) for sid, path in (body.get("sources") or {}).items()}
        base = str(body.get("base") or "")
        if base not in sources:
            raise ValueError("Choose which of the files is the base")
        base_model = sources[base].model
        header_bytes = len(
            ggufio.build_header(
                base_model.kv,
                [(n, t.shape, t.type, t.nbytes) for n, t in base_model.tensors.items()],
                base_model.alignment,
            )[0]
        )
        budget = int(float(body.get("budget_gb") or 0) * GB)
        if body.get("layers"):  # a choice made earlier: only the preview is wanted
            pick = {int(k): v for k, v in body["layers"].items()}
            sel = None
        else:
            if budget <= 0:
                raise ValueError("Enter a size budget in GB")
            errors = body.get("errors") if body.get("objective") == "errors" else None
            sel = plan.choose_layers(sources, base, budget, errors, header_bytes)
            pick = sel.layers
        models = {sid: s.model for sid, s in sources.items()}
        rules = plan.parse_rules(list(body.get("rules") or []))
        assigned = plan.assemble(base_model, models, pick, base, rules)
        size = plan.predicted_file_size(base_model, models, assigned)
        out: dict[str, Any] = {
            "mode": mode,
            "selection": sel.public() if sel else None,
            "layers": {str(k): v for k, v in sorted(pick.items())},
            "file_bytes": size,
            "sources": [s.describe() for s in sources.values()],
            "preview": plan.memory_preview(base_model, models, assigned, ram, vram, context),
        }
        keep = int(body.get("reap_keep") or 0)
        if keep:
            out["pruned_bytes"] = reap.predicted_prune_size(base_model, keep)
        return out

    @staticmethod
    def _file(path: Any) -> str:
        p = Path(str(path or "").strip().strip('"')).expanduser()
        if not p.is_file():
            raise ValueError(f"No file at {p}")
        return str(p)

    # ------------------------------------------------------------------------------------------ corpus

    def make_corpus(self, body: dict[str, Any]) -> dict[str, Any]:
        out = paths.ensure(self.data_dir / "forge" / "corpus") / f"corpus-{int(time.time())}.txt"
        chars = int(body.get("chars") or 1_000_000)
        try:
            if body.get("stdlib"):
                return corpus.stdlib_sample(out, chars)
            inputs = [str(x) for x in body.get("inputs") or []]
            if not inputs:
                raise ValueError("Name at least one text file or folder")
            return corpus.build(inputs, out, chars=chars)
        except corpus.CorpusError as exc:
            raise ValueError(str(exc)) from None

    # ------------------------------------------------------------------------------------------ recipes

    def import_recipe(self, text: str) -> dict[str, Any]:
        try:
            r = recipes.loads(text)
        except recipes.RecipeError as exc:
            raise ValueError(str(exc)) from None
        return {
            "recipe": r,
            "needs": [
                {"id": sid, "file": s["file"], "role": s["role"], "size": s["size"], "origin": s["origin"]}
                for sid, s in r["sources"].items()
            ],
        }

    def recipe_of(self, ident: str, portable: bool = True) -> dict[str, Any]:
        r = self.jobs.get_job(ident).data["recipe"]
        return recipes.portable(r) if portable else r

    # ------------------------------------------------------------------------------------------ publish

    def publish(self, ident: str, *, engine: str, gpu: bool, unified: bool) -> dict[str, Any]:
        """Add the finished model to the library and save a start profile for it."""
        job = self.jobs.get_job(ident)
        if job.state != "done":
            raise RuntimeError("Only a finished job can be published")
        final = job.data["artifacts"]["final"]
        model = self.library.add(final)
        name = job.data["name"]
        args = ["-m", "{model}", "--jinja", "-c", "{context}"]
        if gpu and not unified:
            args += ["-ngl", "99", "-cmoe"]  # dense weights on the GPU, routed experts in RAM
        elif unified:
            args += ["-ngl", "99"]
        notes = f"Built with Forge from recipe {name}."
        card = job.data["result"].get("card")
        if card and card.get("kld") is not None:
            notes += f" KL divergence {card['kld']:.4f} against {card.get('reference') or 'the reference'}."
        prof = self.profiles.save(
            {"name": f"Forge: {name}", "engine": engine, "binary": "", "args": args, "env": {}, "notes": notes}
        )
        return {"model": model, "profile": prof.public()}
