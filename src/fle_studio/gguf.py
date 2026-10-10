"""Read GGUF headers: metadata and tensor sizes, without touching the weights.

Used to show what a model is (architecture, context length, experts) and to estimate how much of it has to live on
the GPU. Tensor sizes come from the offsets in the header, so quant types this module has never heard of (the FLE
files use some) are measured correctly too.
"""

from __future__ import annotations

import os
import re
import struct
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
_MAX_STRING = 16 * 1024 * 1024  # a chat template is a few KB; anything this long is a broken file

SPLIT_RE = re.compile(r"^(?P<base>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE)
# Routed expert weights. Shared experts (`*_shexp`) are dense and belong with the rest.
EXPERT_RE = re.compile(r"\.ffn_(?:gate_up|gate|up|down)_exps\.")
# Input tables llama.cpp keeps in system memory (memory-mapped) instead of on the GPU.
INPUT_PREFIXES = ("token_embd.", "per_layer_token_embd.")

# Bytes per element of the KV cache types llama.cpp accepts for -ctk/-ctv.
KV_TYPE_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
                 "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}


class GGUFError(ValueError):
    pass


class _Reader:
    def __init__(self, f: BinaryIO) -> None:
        self.f = f

    def read(self, n: int) -> bytes:
        data = self.f.read(n)
        if len(data) != n:
            raise GGUFError("the file ends inside the header (incomplete download?)")
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        n = self.u64()
        if n > _MAX_STRING:
            raise GGUFError("implausible string length in header")
        return self.read(n).decode("utf-8", errors="replace")

    def value(self, vtype: int, keep: int) -> Any:
        if vtype in _SCALAR:
            fmt = _SCALAR[vtype]
            return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            etype, count = self.u32(), self.u64()
            if etype in _SCALAR:
                size = struct.calcsize(_SCALAR[etype])
                if count <= keep:
                    return list(struct.unpack(f"<{count}{_SCALAR[etype][1]}", self.read(size * count)))
                self.f.seek(size * count, os.SEEK_CUR)
                return {"array_of": etype, "count": count}
            if etype == _STRING:
                if count <= keep:
                    return [self.string() for _ in range(count)]
                for _ in range(count):  # vocabularies: 150k strings, skipped without decoding
                    n = self.u64()
                    if n > _MAX_STRING:
                        raise GGUFError("implausible string length in header")
                    self.f.seek(n, os.SEEK_CUR)
                return {"array_of": etype, "count": count}
            if etype == _ARRAY:
                return [self.value(_ARRAY, keep) for _ in range(count)]
        raise GGUFError(f"unknown value type {vtype} in header")


@dataclass
class Tensor:
    name: str
    shape: tuple[int, ...]
    type: int
    size: int


@dataclass
class Header:
    path: str
    version: int
    meta: dict[str, Any]
    tensors: list[Tensor]
    file_size: int


def read_header(path: str | os.PathLike[str], keep_array: int = 64) -> Header:
    """Parse one GGUF file's header. Arrays longer than `keep_array` are summarised as {array_of, count}."""
    p = Path(path)
    with p.open("rb") as f:
        r = _Reader(f)
        if r.read(4) != b"GGUF":
            raise GGUFError("not a GGUF file")
        version = r.u32()
        if version not in (2, 3):
            raise GGUFError(f"GGUF version {version} is not supported")
        n_tensors, n_kv = r.u64(), r.u64()
        if n_tensors > 2_000_000 or n_kv > 1_000_000:
            raise GGUFError("implausible header (not a GGUF file?)")
        meta: dict[str, Any] = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.value(r.u32(), keep_array)
        raw = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            if n_dims > 8:
                raise GGUFError("implausible tensor rank in header")
            dims = struct.unpack(f"<{n_dims}Q", r.read(8 * n_dims))
            raw.append((name, tuple(int(d) for d in dims), r.u32(), r.u64()))
        align = meta.get("general.alignment", 32)
        align = align if isinstance(align, int) and align > 0 else 32
        data_start = -(-f.tell() // align) * align
    file_size = p.stat().st_size
    order = sorted(range(len(raw)), key=lambda i: raw[i][3])
    sizes = [0] * len(raw)
    for k, i in enumerate(order):
        end = raw[order[k + 1]][3] if k + 1 < len(order) else file_size - data_start
        sizes[i] = max(0, end - raw[i][3])
    tensors = [Tensor(t[0], t[1], t[2], sizes[i]) for i, t in enumerate(raw)]
    return Header(str(p), version, meta, tensors, file_size)


def split_parts(first: Path) -> list[Path]:
    """All files of a split model, given any part. A single file is its own list."""
    m = SPLIT_RE.match(first.name)
    if not m:
        return [first]
    count = int(m.group("count"))
    return [first.with_name(f"{m.group('base')}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]


@dataclass
class ModelInfo:
    path: str
    parts: list[str]
    missing: list[str]
    size: int
    arch: str
    name: str
    context_length: int | None
    layers: int | None
    experts: int | None
    experts_used: int | None
    expert_bytes: int
    input_bytes: int  # token tables that stay in system memory
    gpu_bytes: int  # everything else: what a GPU holds when all non-expert weights are offloaded
    has_chat_template: bool
    file_type: int | None
    meta: dict[str, Any] = field(default_factory=dict)  # the scalar metadata of the first part

    @property
    def is_moe(self) -> bool:
        return bool(self.experts and self.expert_bytes)

    @property
    def complete(self) -> bool:
        return not self.missing

    def kv_bytes_per_token(self, type_k: str = "f16", type_v: str = "f16") -> float:
        """Estimated KV cache bytes per context token. Hybrid models only count their full-attention layers."""
        m, a = self.meta, self.arch
        layers = int(m.get(f"{a}.block_count") or 0)
        heads = m.get(f"{a}.attention.head_count")
        heads_kv = m.get(f"{a}.attention.head_count_kv", heads)
        emb = m.get(f"{a}.embedding_length")
        head_dim = (emb // heads) if isinstance(emb, int) and isinstance(heads, int) and heads else 128
        k_len = int(m.get(f"{a}.attention.key_length") or head_dim)
        v_len = int(m.get(f"{a}.attention.value_length") or k_len)
        per_layer = heads_kv if isinstance(heads_kv, list) else [heads_kv or 0] * layers
        interval = m.get(f"{a}.full_attention_interval")
        if isinstance(interval, int) and interval > 1:
            per_layer = [h for i, h in enumerate(per_layer) if (i + 1) % interval == 0]
        bk = KV_TYPE_BYTES.get(type_k, 2.0)
        bv = KV_TYPE_BYTES.get(type_v, 2.0)
        return float(sum(int(h or 0) * (k_len * bk + v_len * bv) for h in per_layer))

    def public(self) -> dict[str, Any]:
        return {
            "arch": self.arch,
            "name": self.name,
            "context_length": self.context_length,
            "layers": self.layers,
            "experts": self.experts,
            "experts_used": self.experts_used,
            "expert_bytes": self.expert_bytes,
            "input_bytes": self.input_bytes,
            "gpu_bytes": self.gpu_bytes,
            "moe": self.is_moe,
            "chat_template": self.has_chat_template,
            "kv_bytes_per_token_q8": round(self.kv_bytes_per_token("q8_0", "q8_0")),
        }


def inspect(first: str | os.PathLike[str]) -> ModelInfo:
    """Read every part of a (possibly split) model and add up what goes where."""
    first = Path(first)
    parts = split_parts(first)
    present = [p for p in parts if p.exists()]
    missing = [p.name for p in parts if not p.exists()]
    if not present or present[0] != parts[0]:
        raise GGUFError("the first part of the model is missing")
    head = read_header(parts[0])
    meta = {k: v for k, v in head.meta.items() if not isinstance(v, (list, dict)) and k != "tokenizer.chat_template"}
    arch = str(head.meta.get("general.architecture", "unknown"))
    expert = inputs = other = 0
    size = 0
    for i, p in enumerate(present):
        h = head if i == 0 else read_header(p, keep_array=0)
        size += h.file_size
        for t in h.tensors:
            if EXPERT_RE.search(t.name):
                expert += t.size
            elif t.name.startswith(INPUT_PREFIXES):
                inputs += t.size
            else:
                other += t.size

    def num(key: str) -> int | None:
        v = head.meta.get(f"{arch}.{key}")
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    name = head.meta.get("general.name") or SPLIT_RE.sub(r"\g<base>", first.name).removesuffix(".gguf")
    return ModelInfo(
        path=str(parts[0]),
        parts=[str(p) for p in parts],
        missing=missing,
        size=size,
        arch=arch,
        name=str(name),
        context_length=num("context_length"),
        layers=num("block_count"),
        experts=num("expert_count"),
        experts_used=num("expert_used_count"),
        expert_bytes=expert,
        input_bytes=inputs,
        gpu_bytes=other,
        has_chat_template=isinstance(head.meta.get("tokenizer.chat_template"), str),
        file_type=head.meta.get("general.file_type") if isinstance(head.meta.get("general.file_type"), int) else None,
        meta=meta,
    )


_cache: dict[tuple[str, int, int], ModelInfo] = {}
_cache_lock = threading.Lock()


def inspect_cached(first: str | os.PathLike[str]) -> ModelInfo:
    """inspect() remembered per (path, size, mtime) of the first part; a model's header is read once per change."""
    p = Path(first)
    st = p.stat()
    key = (str(p), st.st_size, int(st.st_mtime))
    with _cache_lock:
        hit = _cache.get(key)
    if hit and not hit.missing:
        return hit
    info = inspect(p)
    with _cache_lock:
        _cache[key] = info
    return info
