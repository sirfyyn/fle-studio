"""Forge: build your own quant with the FLE recipe, from inside FLE Studio.

A plugin that ships with FLE Studio (entry point `forge`): it adds the Forge page and the /api/forge routes.

ggufio   reading and writing GGUF files (no dependencies)
recipe   the recipe JSON (schema 1)
tools    finding llama-imatrix, llama-quantize, llama-perplexity, llama-fle-reap
plan     the recipe calculator and the memory preview
build    writing a model from pieces of other models (hybrid build), resumable
reap     expert pruning
verify   KL divergence results
corpus   calibration text
jobs     background jobs with steps, log, cancel and resume
steps    the step implementations
api      the Forge object the routes below call
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

SCHEMA = 1


def register(ctx: Any) -> None:
    from fle_studio.server import ApiError  # noqa: PLC0415 - loaded when the plugin starts, after the server

    from . import plan as forge_plan  # noqa: PLC0415
    from . import recipe as recipes  # noqa: PLC0415
    from .api import Forge  # noqa: PLC0415

    studio = ctx.studio

    def engine_paths() -> list[tuple[str, str]]:
        """The llama-server files whose folders are searched for the Forge tools."""
        out = [("engine", e["path"]) for e in studio.engines.installed()]
        custom = studio.config.get("engine").get("custom_path")
        if custom:
            out.append(("custom engine", custom))
        return out

    forge = Forge(
        studio.data_dir,
        studio.config.get,
        lambda: studio.config.models_dir,
        engine_paths,
        studio.library,
        studio.profiles,
    )

    def overview(h: Any, query: dict[str, str]) -> None:
        h.send_json(200, forge.overview())

    def set_tools(h: Any, query: dict[str, str]) -> None:
        studio.config.update({"forge": {"tools_dir": str(h.read_json().get("tools_dir") or "")}})
        h.send_json(200, forge.overview())

    def inspect(h: Any, query: dict[str, str]) -> None:
        h.send_json(200, forge.inspect(str(h.read_json().get("path") or "")))

    def calculate(h: Any, query: dict[str, str]) -> None:
        try:
            h.send_json(200, forge.calculate(h.read_json()))
        except forge_plan.PlanError as exc:
            raise ApiError(400, str(exc)) from None

    def make_corpus(h: Any, query: dict[str, str]) -> None:
        h.send_json(200, forge.make_corpus(h.read_json()))

    def recipe_import(h: Any, query: dict[str, str]) -> None:
        body = h.read_json()
        h.send_json(200, forge.import_recipe(json.dumps(body.get("recipe") or body)))

    def recipe_share(h: Any, query: dict[str, str]) -> None:
        try:
            h.send_json(200, {"recipe": recipes.portable(h.read_json().get("recipe") or {})})
        except recipes.RecipeError as exc:
            raise ApiError(400, str(exc)) from None

    def check(h: Any, query: dict[str, str]) -> None:
        try:
            h.send_json(200, forge.jobs.check(h.read_json().get("recipe") or {}))
        except recipes.RecipeError as exc:
            raise ApiError(400, str(exc)) from None

    def job_create(h: Any, query: dict[str, str]) -> None:
        try:
            h.send_json(200, forge.jobs.create(h.read_json().get("recipe") or {}))
        except recipes.RecipeError as exc:
            raise ApiError(400, str(exc)) from None
        except RuntimeError as exc:
            raise ApiError(409, str(exc)) from None

    def job(h: Any, query: dict[str, str], id: str) -> None:  # noqa: A002
        h.send_json(200, forge.jobs.get(id))

    def job_log(h: Any, query: dict[str, str], id: str) -> None:  # noqa: A002
        h.send_json(200, forge.jobs.log_since(id, int(query.get("since") or 0)))

    def job_recipe(h: Any, query: dict[str, str], id: str) -> None:  # noqa: A002
        h.send_json(200, forge.recipe_of(id))

    def job_action(h: Any, query: dict[str, str], id: str, action: str) -> None:  # noqa: A002
        try:
            if action == "cancel":
                h.send_json(200, forge.jobs.cancel(id))
            elif action == "resume":
                h.send_json(200, forge.jobs.resume(id))
            elif action == "delete":
                forge.jobs.delete(id, keep_model=not h.read_json().get("delete_model"))
                h.send_json(200, {"ok": True})
            else:
                info = studio.sysinfo()
                gpus = info.get("gpus") or []
                installed = {e["kind"] for e in studio.engines.installed()}
                engine = "fle" if "fle" in installed else "upstream"
                h.send_json(
                    200, forge.publish(id, engine=engine, gpu=bool(gpus), unified=any(g.get("unified") for g in gpus))
                )
        except RuntimeError as exc:
            raise ApiError(409, str(exc)) from None

    job_id = r"(?P<id>[0-9a-f]{12})"
    for method, pattern, fn in (
        ("GET", r"^/api/forge$", overview),
        ("POST", r"^/api/forge/tools$", set_tools),
        ("POST", r"^/api/forge/inspect$", inspect),
        ("POST", r"^/api/forge/calculate$", calculate),
        ("POST", r"^/api/forge/corpus$", make_corpus),
        ("POST", r"^/api/forge/recipe/import$", recipe_import),
        ("POST", r"^/api/forge/recipe/share$", recipe_share),
        ("POST", r"^/api/forge/check$", check),
        ("POST", r"^/api/forge/jobs$", job_create),
        ("GET", rf"^/api/forge/jobs/{job_id}$", job),
        ("GET", rf"^/api/forge/jobs/{job_id}/log$", job_log),
        ("GET", rf"^/api/forge/jobs/{job_id}/recipe$", job_recipe),
        ("POST", rf"^/api/forge/jobs/{job_id}/(?P<action>cancel|resume|delete|publish)$", job_action),
    ):
        ctx.api(method, pattern, fn)

    ctx.describe("Forge", "Build your own FLE quant: hybrid builds, expert pruning, KL checks.")
    ctx.page("Forge", icon="forge", script="forge.js", path="/forge")
    ctx.web(resources.files(__package__) / "web")
