"""The API of the tuner, the quality measurements, the expert cache view and the flag explanations.

Kept apart from server.py so the two files do not grow into each other: server.py only holds the three lines that
make these routes reachable. Every handler is ``handler(studio, query, body, **groups) -> (status, json)``; ``body()``
reads the JSON request. Mistakes raise ValueError (answered as 400 with the message), a busy engine EngineBusy (409).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from . import flaghelp, quality, tuner
from .engine import EngineBusy

Result = tuple[int, Any]
Handler = Callable[..., Result]


def _model(studio: Any, ident: Any) -> dict[str, Any]:
    model = studio.library.get(str(ident or ""))
    if not model:
        raise ValueError("Choose a model first")
    return model


def _int(value: Any, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------------------------- tuner


def tuning_state(studio: Any, query: dict[str, str], body: Callable[[], dict[str, Any]]) -> Result:
    model = query.get("model", "")
    best = studio.tuner.tuned_profile(model) if model else None
    return 200, {
        "current": studio.tuner.status(),
        "results": studio.tuner.results(),
        "best": best,
        "params": [{"key": k, "label": v} for k, v in tuner.PARAM_LABELS.items()],
        "defaults": {"params": list(tuner.DEFAULT_PARAMS), "runs": 3, "budget_min": 20},
    }


def tuning_start(studio: Any, query: dict[str, str], body: Callable[[], dict[str, Any]]) -> Result:
    data = body()
    model = _model(studio, data.get("model") or studio.config.get("last_model"))
    keys = tuple(str(k) for k in data.get("params") or tuner.DEFAULT_PARAMS)
    if studio.engine.state not in ("stopped", "crashed") or studio.engine.unit is not None:
        raise EngineBusy("Stop the engine first: the tuner starts it itself, once for every setting it tries.")
    run = studio.tuner.start_tune(
        model_id=model["id"],
        profile=data.get("profile") or "auto",
        budget_s=_int(data.get("budget_min"), 20, 1, 1440) * 60,
        runs=_int(data.get("runs"), 3, tuner.MIN_RUNS, tuner.MAX_RUNS),
        params=keys,
        screen=bool(data.get("screen", True)),
    )
    return 200, run


def tuning_cancel(studio: Any, query: dict[str, str], body: Callable[[], dict[str, Any]]) -> Result:
    studio.tuner.cancel()
    return 200, {"ok": True}


def tuning_delete(studio: Any, query: dict[str, str], body: Callable[[], dict[str, Any]], id: str) -> Result:  # noqa: A002
    studio.tuner.delete(id)
    return 200, {"ok": True}


def tuning_aba(studio: Any, query: dict[str, str], body: Callable[[], dict[str, Any]]) -> Result:
    data = body()
    model = _model(studio, data.get("model") or studio.config.get("last_model"))
    a, b = str(data.get("a") or ""), str(data.get("b") or "")
    if not a or not b:
        raise ValueError("Pick the two settings to compare (launch profiles, or Automatic)")
    return 200, studio.tuner.start_aba(
        model_id=model["id"],
        profile_a=a,
        profile_b=b,
        runs=_int(data.get("runs"), 3, tuner.MIN_RUNS, tuner.MAX_RUNS),
        label=str(data.get("label") or ""),
    )


def best_profile(studio: Any, query: dict[str, str], body: Callable[[], dict[str, Any]]) -> Result:
    """The profile 'Best for my machine' starts: the tuned one for this model on this machine, or None."""
    return 200, {"profile": studio.tuner.tuned_profile(query.get("model", ""))}


# ---------------------------------------------------------------------------------------------- quality


def quality_state(studio: Any, query: dict[str, str], body: Callable[[], dict[str, Any]]) -> Result:
    q = studio.quality
    return 200, {
        "current": q.status(),
        "results": q.results(),
        "kld_by_model": q.latest_kld_by_model(),
        "builtin": quality.load_questions(quality.BUILTIN)[0],
        "mkqa_license": quality.MKQA_LICENSE,
        "kld_ctx": quality.KLD_CTX,
    }


def quality_kld(studio: Any, query: dict[str, str], body: Callable[[], dict[str, Any]]) -> Result:
    data = body()
    ref, cand = _model(studio, data.get("reference")), _model(studio, data.get("candidate"))
    engine = studio.engine_for("auto", studio.sysinfo())
    return 200, studio.quality.start_kld(
        reference=ref,
        candidate=cand,
        engine_path=engine["path"],
        text=str(data.get("text") or ""),
        ctx=_int(data.get("ctx"), quality.KLD_CTX, 64, 8192),
        chunks=_int(data.get("chunks"), 0, 0, 1000),
        threads=_int(data.get("threads"), 0, 0, 512),
    )


def quality_probe(studio: Any, query: dict[str, str], body: Callable[[], dict[str, Any]]) -> Result:
    data = body()
    state, base, headers = studio.engine.target()[:3]
    if state not in ("ready", "attached") or not base:
        raise ValueError("Load a model first: the knowledge probe asks the running engine")
    name = (studio.engine.status().get("model") or {}).get("name") or ""
    return 200, studio.quality.start_probe(
        base=base,
        headers=headers,
        model=name,
        source=str(data.get("source") or quality.BUILTIN),
        lang=str(data.get("lang") or ""),
        limit=_int(data.get("limit"), 0, 0, 1000),
        label=str(data.get("label") or ""),
    )


def quality_mkqa(studio: Any, query: dict[str, str], body: Callable[[], dict[str, Any]]) -> Result:
    data = body()
    return 200, quality.import_mkqa(
        studio.data_dir,
        lang=str(data.get("lang") or "en"),
        count=_int(data.get("count"), 48, 1, 500),
        offset=_int(data.get("offset"), 0, 0, 100000),
    )


def quality_cancel(studio: Any, query: dict[str, str], body: Callable[[], dict[str, Any]]) -> Result:
    studio.quality.cancel()
    return 200, {"ok": True}


def quality_delete(studio: Any, query: dict[str, str], body: Callable[[], dict[str, Any]], id: str) -> Result:  # noqa: A002
    studio.quality.delete(id)
    return 200, {"ok": True}


# ---------------------------------------------------------------------------------------------- cache, flags


def cache_view(studio: Any, query: dict[str, str], body: Callable[[], dict[str, Any]]) -> Result:
    layer = query.get("layer")
    return 200, studio.cache.snapshot(int(layer) if layer and layer.isdigit() else None)


def flags_table(studio: Any, query: dict[str, str], body: Callable[[], dict[str, Any]]) -> Result:
    return 200, flaghelp.table()


def flags_explain(studio: Any, query: dict[str, str], body: Callable[[], dict[str, Any]]) -> Result:
    data = body()
    args = [str(a) for a in data.get("args") or []]
    env = {str(k): str(v) for k, v in (data.get("env") or {}).items()}
    return 200, {"items": flaghelp.explain(args, env)}


ROUTES: list[tuple[str, re.Pattern[str], Handler]] = [
    (m, re.compile(p), h)
    for m, p, h in [
        ("GET", r"^/api/tuning$", tuning_state),
        ("POST", r"^/api/tuning/start$", tuning_start),
        ("POST", r"^/api/tuning/cancel$", tuning_cancel),
        ("POST", r"^/api/tuning/aba$", tuning_aba),
        ("POST", r"^/api/tuning/(?P<id>[0-9a-f]+)/delete$", tuning_delete),
        ("GET", r"^/api/tuning/best$", best_profile),
        ("GET", r"^/api/quality$", quality_state),
        ("POST", r"^/api/quality/kld$", quality_kld),
        ("POST", r"^/api/quality/probe$", quality_probe),
        ("POST", r"^/api/quality/mkqa$", quality_mkqa),
        ("POST", r"^/api/quality/cancel$", quality_cancel),
        ("POST", r"^/api/quality/(?P<id>[0-9a-f]+)/delete$", quality_delete),
        ("GET", r"^/api/cache$", cache_view),
        ("GET", r"^/api/flags$", flags_table),
        ("POST", r"^/api/flags/explain$", flags_explain),
    ]
]
