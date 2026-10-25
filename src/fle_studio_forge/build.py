"""Writing a GGUF from pieces of other GGUFs: the engine behind the hybrid build and the expert pruning.

Nothing is quantised here; bytes are copied. Every output tensor is a list of segments (file, offset, length) of its
sources, so a donor tensor is one segment and a pruned expert tensor is a few. The output is written to `NAME.part`
next to a small state file; a cancelled or crashed build starts again at the first tensor that was not finished, after
the header and the finished prefix have been checked against what would be written now.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fle_studio import gguf

from . import ggufio
from .ggufio import CHUNK

EDGE = 65536  # bytes compared at both ends of every tensor when a finished file is checked


@dataclass
class OutTensor:
    name: str
    shape: tuple[int, ...]
    type: int
    nbytes: int
    segments: list[tuple[str, int, int]] = field(default_factory=list)  # (file, offset, length), in order

    def __post_init__(self) -> None:
        if self.segments and sum(s[2] for s in self.segments) != self.nbytes:
            raise ValueError(
                f"{self.name}: segments add up to {sum(s[2] for s in self.segments)} bytes, not {self.nbytes}"
            )


def _segment_bytes(segments: list[tuple[str, int, int]]):
    for path, offset, length in segments:
        yield from ggufio.read_chunks(path, offset, length)


def _edges(segments: list[tuple[str, int, int]], total: int) -> tuple[bytes, bytes]:
    """The first and last EDGE bytes of the tensor the segments describe."""
    if total <= 2 * EDGE:
        return b"".join(_segment_bytes(segments)), b""
    head = bytearray()
    for data in _segment_bytes(segments):
        head += data
        if len(head) >= EDGE:
            break
    tail = bytearray()
    for path, offset, length in reversed(segments):
        take = min(length, EDGE - len(tail))
        with open(path, "rb") as f:
            f.seek(offset + length - take)
            tail[0:0] = f.read(take)
        if len(tail) >= EDGE:
            break
    return bytes(head[:EDGE]), bytes(tail)


def _state_path(out: Path) -> Path:
    return out.with_name(out.name + ".part.json")


def _write_state(out: Path, state: dict[str, Any]) -> None:
    p = _state_path(out)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(p)


def write_model(
    out: str | os.PathLike[str],
    kv: list[ggufio.KV],
    tensors: list[OutTensor],
    alignment: int,
    progress: Callable[[float, str], None] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Write the model to `out` (atomically: NAME.part is renamed at the end) and return its size and SHA-256.

    Resumes a partial NAME.part left by an earlier call with the same header. Raises Cancelled (the partial file and
    its state stay for the next call)."""
    out = Path(out)
    say = log or (lambda _: None)
    infos = [(t.name, t.shape, t.type, t.nbytes) for t in tensors]
    header, offsets = ggufio.build_header(kv, infos, alignment)
    header_sha = hashlib.sha256(header).hexdigest()
    total = len(header) + (offsets[-1] + -(-tensors[-1].nbytes // alignment) * alignment if tensors else 0)
    part = out.with_name(out.name + ".part")
    out.parent.mkdir(parents=True, exist_ok=True)

    hasher = hashlib.sha256()
    done, written = 0, 0
    state = None
    if part.exists() and _state_path(out).exists():
        try:
            state = json.loads(_state_path(out).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = None
    if (
        state
        and state.get("header_sha") == header_sha
        and 0 < state.get("offset", 0) <= part.stat().st_size
        and state.get("done", 0) <= len(tensors)
    ):
        done, written = int(state["done"]), int(state["offset"])
        say(f"Resuming {out.name}: {done} of {len(tensors)} tensors are already written")
        with open(part, "rb") as f:  # the hash of the finished prefix
            left = written
            while left > 0:
                data = f.read(min(CHUNK, left))
                if not data:
                    break
                hasher.update(data)
                left -= len(data)
        if left:
            done, written, hasher = 0, 0, hashlib.sha256()
    elif part.exists():
        say("A partial file from a different recipe was found; starting from the beginning")

    mode = "r+b" if written else "wb"
    with open(part, mode) as f:
        sink = ggufio.Sink(f, hasher, written, cancelled)
        if written:
            f.truncate(written)
            f.seek(written)
        else:
            sink.write(header)
        t0, last_state = time.monotonic(), 0.0
        started_at = written
        try:
            for i in range(done, len(tensors)):
                t = tensors[i]
                for data in _segment_bytes(t.segments):
                    sink.write(data)
                sink.pad(alignment)
                done = i + 1
                now = time.monotonic()
                if now - last_state > 1.0 or done == len(tensors):
                    f.flush()
                    _write_state(out, {"header_sha": header_sha, "done": done, "offset": sink.written})
                    last_state = now
                if progress:
                    rate = (sink.written - started_at) / max(now - t0, 1e-6)
                    progress(
                        sink.written / max(total, 1),
                        f"{done} of {len(tensors)} tensors, {sink.written / 1e9:.1f} of {total / 1e9:.1f} GB"
                        + (f", {rate / 1e6:.0f} MB/s" if rate > 0 else ""),
                    )
        except BaseException:
            f.flush()
            # The state records the end of the last finished tensor; bytes of an unfinished one are cut on resume.
            if done:
                boundary = len(header) + offsets[done - 1] + -(-tensors[done - 1].nbytes // alignment) * alignment
                _write_state(out, {"header_sha": header_sha, "done": done, "offset": boundary})
            raise
    if sink.written != total:
        raise RuntimeError(f"wrote {sink.written} bytes, expected {total}")
    problems = verify_file(part, kv, tensors, alignment)
    if problems:
        raise RuntimeError("the new file does not match its sources: " + "; ".join(problems[:5]))
    os.replace(part, out)
    _state_path(out).unlink(missing_ok=True)
    return {"path": str(out), "size": total, "sha256": sink.hasher.hexdigest(), "tensors": len(tensors)}


def verify_file(
    path: str | os.PathLike[str], kv: list[ggufio.KV], tensors: list[OutTensor], alignment: int
) -> list[str]:
    """Header, tensor table and the first and last bytes of every tensor of a finished file, against the plan."""
    problems: list[str] = []
    try:
        got = ggufio.read_gguf(path)
    except gguf.GGUFError as exc:
        return [f"unreadable: {exc}"]
    if [k.key for k in got.kv] != [k.key for k in kv]:
        problems.append("metadata keys differ")
    if [t.name for t in got.tensors] != [t.name for t in tensors]:
        return [*problems, "tensor names or order differ"]
    with open(path, "rb") as f:
        for have, want in zip(got.tensors, tensors, strict=True):
            if have.shape != want.shape or have.type != want.type:
                problems.append(f"type or shape: {want.name}")
                continue
            head, tail = _edges(want.segments, want.nbytes)
            f.seek(got.data_start + have.offset)
            if f.read(len(head)) != head:
                problems.append(f"first bytes: {want.name}")
                continue
            if tail:
                f.seek(got.data_start + have.offset + want.nbytes - len(tail))
                if f.read(len(tail)) != tail:
                    problems.append(f"last bytes: {want.name}")
    return problems


# ------------------------------------------------------------------------------------------------ the hybrid


def build_hybrid(
    recipe: dict[str, Any],
    out: str | os.PathLike[str],
    progress: Callable[[float, str], None] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Assemble the hybrid a recipe describes. Metadata and all tensors outside the chosen layers come from the base."""
    from . import plan as planner

    say = log or (lambda _: None)
    h = recipe["hybrid"]
    sources = {sid: s for sid, s in recipe["sources"].items() if s["role"] != "reference"}
    models: dict[str, ggufio.Model] = {}
    for sid, s in sources.items():
        if sid == h["base"] or sid in set(h["layers"].values()) or any(r["source"] == sid for r in h["rules"]):
            models[sid] = ggufio.open_model(s["path"])
            say(f"Source {sid}: {Path(s['path']).name}, {len(models[sid].order)} tensors")
    base = models[h["base"]]
    pick = {int(layer): sid for layer, sid in h["layers"].items()}
    rules = planner.parse_rules(h["rules"])
    plan = planner.assemble(base, models, pick, h["base"], rules)
    tensors = []
    for name in base.order:
        t = models[plan[name]].tensors[name]
        tensors.append(OutTensor(name, t.shape, t.type, t.nbytes, [models[plan[name]].location(name)]))
    origin: dict[str, int] = {}
    for sid in plan.values():
        origin[sid] = origin.get(sid, 0) + 1
    say("Tensors per source: " + ", ".join(f"{k} {v}" for k, v in origin.items()))
    result = write_model(out, base.kv, tensors, base.alignment, progress, cancelled, log)
    result["origin"] = origin
    result["layers"] = {str(k): v for k, v in sorted(pick.items())}
    return result
