"""Reading and writing GGUF files without any dependency (no gguf-py, no numpy).

`fle_studio.gguf` only needs sizes and a few metadata values, so it skips the weights and does not keep the raw
bytes of the metadata. Forge has to copy a file: the metadata is carried over byte for byte, and tensors are
streamed from the sources into the new file in chunks, so a model of any size needs only a few MiB of memory.

Shapes are kept in GGUF order (`ne0` first, the fastest axis). For the tensors of a mixture-of-experts model the
expert axis is therefore the last entry of the shape, and the slowest axis in the data: one expert is one
contiguous slice of `nbytes / n_expert` bytes, whatever the quantisation.
"""

from __future__ import annotations

import hashlib
import os
import struct
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from ..gguf import GGUFError, split_parts
from ..gguf import _Reader as _HeaderReader

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
CHUNK = 8 * 1024 * 1024

# GGUF metadata value types.
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STRING, T_ARRAY, T_U64, T_I64, T_F64 = range(13)
_SCALAR_FMT = {
    T_U8: "<B",
    T_I8: "<b",
    T_U16: "<H",
    T_I16: "<h",
    T_U32: "<I",
    T_I32: "<i",
    T_F32: "<f",
    T_BOOL: "<?",
    T_U64: "<Q",
    T_I64: "<q",
    T_F64: "<d",
}

# ggml tensor types: id -> (name, elements per block, bytes per block). Only types whose layout is long settled;
# a file with a type this table does not know is still copied correctly (sizes then come from the header offsets).
_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
}
TYPE_IDS = {name: i for i, (name, _, _) in _TYPES.items()}
# Types llama-quantize accepts as a target (the names it takes on its command line, lower case).
QUANTIZE_TARGETS = tuple(
    sorted(n.lower() for n in TYPE_IDS if n not in ("Q8_K", "Q8_1", "I8", "I16", "I32", "I64", "F64", "MXFP4"))
)

# The named mixtures llama-quantize accepts as its target (the "Q4_K_M" of a download name). They are not tensor types:
# the tool picks a type per tensor. Only the default type of a recipe may be one; for a size estimate each is
# approximated by the tensor type it is built around.
FTYPE_BASE = {
    "q2_k_s": "q2_k",
    "q3_k_s": "q3_k",
    "q3_k_m": "q3_k",
    "q3_k_l": "q3_k",
    "q4_k_s": "q4_k",
    "q4_k_m": "q4_k",
    "q5_k_s": "q5_k",
    "q5_k_m": "q5_k",
    "iq3_xs": "iq3_xxs",
    "iq3_m": "iq3_s",
    "iq2_m": "iq2_s",
}
QUANTIZE_FTYPES = tuple(sorted({*QUANTIZE_TARGETS, *FTYPE_BASE}))


def type_name(type_id: int) -> str:
    info = _TYPES.get(type_id)
    return info[0] if info else f"TYPE_{type_id}"


def type_id(name: str) -> int | None:
    return TYPE_IDS.get(name.upper())


def row_bytes(type_: int, ne0: int) -> int | None:
    """Bytes of one row of `ne0` elements, or None for an unknown type or a row that is not a whole number of blocks."""
    info = _TYPES.get(type_)
    if not info or ne0 % info[1]:
        return None
    return ne0 // info[1] * info[2]


def tensor_nbytes(type_: int, shape: tuple[int, ...]) -> int | None:
    if not shape:
        return None
    row = row_bytes(type_, shape[0])
    if row is None:
        return None
    rows = 1
    for d in shape[1:]:
        rows *= d
    return row * rows


def bits_per_weight(type_: int) -> float | None:
    info = _TYPES.get(type_)
    return info[2] * 8 / info[1] if info else None


# ------------------------------------------------------------------------------------------------ reading


@dataclass
class KV:
    """One metadata entry. `raw` is the encoded value (without key and type) and is written back unchanged."""

    key: str
    vtype: int
    raw: bytes
    value: Any = None  # decoded scalars and strings; None for arrays

    def encoded(self) -> bytes:
        return encode_string(self.key) + struct.pack("<I", self.vtype) + self.raw


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    type: int
    offset: int  # from the start of the data section of its file
    nbytes: int
    part: int = 0  # index of the file this tensor lives in

    @property
    def type_name(self) -> str:
        return type_name(self.type)

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass
class GGUFFile:
    path: str
    version: int
    kv: list[KV]
    tensors: list[TensorInfo]
    data_start: int
    alignment: int
    file_size: int


def read_gguf(path: str | os.PathLike[str], part: int = 0) -> GGUFFile:
    """Parse the whole header of one file: raw metadata, tensor table, where the data starts."""
    p = Path(path)
    with p.open("rb") as f:
        r = _HeaderReader(f)
        if r.read(4) != MAGIC:
            raise GGUFError("not a GGUF file")
        version = r.u32()
        if version not in (2, 3):
            raise GGUFError(f"GGUF version {version} is not supported")
        n_tensors, n_kv = r.u64(), r.u64()
        if n_tensors > 2_000_000 or n_kv > 1_000_000:
            raise GGUFError("implausible header (not a GGUF file?)")
        spans: list[tuple[str, int, int, int, Any]] = []
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            start = f.tell()
            value = r.value(vtype, 0)  # arrays are skipped (keep=0), scalars and strings are decoded
            spans.append((key, vtype, start, f.tell(), None if isinstance(value, (list, dict)) else value))
        raw = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            if n_dims > 8:
                raise GGUFError("implausible tensor rank in header")
            dims = struct.unpack(f"<{n_dims}Q", r.read(8 * n_dims))
            raw.append((name, tuple(int(d) for d in dims), r.u32(), r.u64()))
        header_end = f.tell()
        kv = []
        for key, vtype, start, end, value in spans:
            f.seek(start)
            kv.append(KV(key, vtype, f.read(end - start), value))
    alignment = next((k.value for k in kv if k.key == "general.alignment"), DEFAULT_ALIGNMENT)
    alignment = alignment if isinstance(alignment, int) and alignment > 0 else DEFAULT_ALIGNMENT
    data_start = -(-header_end // alignment) * alignment
    file_size = p.stat().st_size
    order = sorted(range(len(raw)), key=lambda i: raw[i][3])
    sizes = [0] * len(raw)
    for k, i in enumerate(order):
        name, shape, type_, offset = raw[i]
        end = raw[order[k + 1]][3] if k + 1 < len(order) else file_size - data_start
        exact = tensor_nbytes(type_, shape)
        sizes[i] = exact if exact is not None else max(0, end - offset)
    tensors = [TensorInfo(t[0], t[1], t[2], t[3], sizes[i], part) for i, t in enumerate(raw)]
    return GGUFFile(str(p), version, kv, tensors, data_start, alignment, file_size)


class Model:
    """A model of one or more GGUF files (a split model is one model). Tensors keep the order of their files."""

    def __init__(self, files: list[GGUFFile]) -> None:
        self.files = files
        self.tensors: dict[str, TensorInfo] = {}
        self.order: list[str] = []
        for f in files:
            for t in f.tensors:
                if t.name in self.tensors:
                    raise GGUFError(f"tensor {t.name} appears twice in {files[0].path}")
                self.tensors[t.name] = t
                self.order.append(t.name)
        # `split.*` describes how a model is cut into files, not the model; the output is one file.
        self.kv = [k for k in files[0].kv if not k.key.startswith("split.")]
        self.meta = {k.key: k.value for k in self.kv if k.value is not None}

    @property
    def arch(self) -> str:
        return str(self.meta.get("general.architecture", ""))

    def number(self, suffix: str) -> int | None:
        v = self.meta.get(f"{self.arch}.{suffix}")
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    @property
    def paths(self) -> list[str]:
        return [f.path for f in self.files]

    @property
    def total_bytes(self) -> int:
        return sum(f.file_size for f in self.files)

    def location(self, name: str) -> tuple[str, int, int]:
        """(file, absolute offset, size) of a tensor's data."""
        t = self.tensors[name]
        f = self.files[t.part]
        return f.path, f.data_start + t.offset, t.nbytes

    @property
    def alignment(self) -> int:
        return self.files[0].alignment


def open_model(first: str | os.PathLike[str]) -> Model:
    parts = split_parts(Path(first))
    missing = [p.name for p in parts if not p.exists()]
    if missing:
        raise GGUFError(f"part(s) of the model are missing: {', '.join(missing)}")
    return Model([read_gguf(p, i) for i, p in enumerate(parts)])


# ------------------------------------------------------------------------------------------------ encoding


def encode_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def kv_scalar(key: str, vtype: int, value: Any) -> KV:
    if vtype == T_STRING:
        return KV(key, vtype, encode_string(str(value)), str(value))
    fmt = _SCALAR_FMT[vtype]
    return KV(key, vtype, struct.pack(fmt, value), value)


def kv_string_array(key: str, items: Iterable[str]) -> KV:
    items = list(items)
    raw = struct.pack("<IQ", T_STRING, len(items)) + b"".join(encode_string(s) for s in items)
    return KV(key, T_ARRAY, raw, None)


def with_value(kv: list[KV], key: str, value: Any) -> list[KV]:
    """The metadata with one scalar replaced by `value` (same type); raises if the key is missing or not a scalar."""
    out = []
    found = False
    for k in kv:
        if k.key == key:
            if k.vtype not in _SCALAR_FMT:
                raise GGUFError(f"{key} is not a number")
            k, found = kv_scalar(key, k.vtype, value), True
        out.append(k)
    if not found:
        raise GGUFError(f"the model has no {key}")
    return out


def layout(tensors: list[tuple[str, tuple[int, ...], int, int]], alignment: int) -> tuple[list[int], int]:
    """Offsets of tensors in the data section (each starts on the alignment) and the size of the section."""
    offsets, pos = [], 0
    for _, _, _, nbytes in tensors:
        offsets.append(pos)
        pos += -(-nbytes // alignment) * alignment
    return offsets, pos


def build_header(
    kv: list[KV],
    tensors: list[tuple[str, tuple[int, ...], int, int]],
    alignment: int = DEFAULT_ALIGNMENT,
    version: int = 3,
) -> tuple[bytes, list[int]]:
    """Header bytes (padded to the alignment, so they are also the offset of the data) and the tensor offsets.

    `tensors` is a list of (name, shape, type id, bytes). The result is a pure function of its arguments, which is
    what lets a build be resumed: the header of the partial file is compared with the one computed again."""
    if alignment != DEFAULT_ALIGNMENT and not any(k.key == "general.alignment" for k in kv):
        kv = [*kv, kv_scalar("general.alignment", T_U32, alignment)]
    offsets, _ = layout(tensors, alignment)
    out = bytearray(MAGIC + struct.pack("<IQQ", version, len(tensors), len(kv)))
    for k in kv:
        out += k.encoded()
    for (name, shape, type_, _), offset in zip(tensors, offsets, strict=True):
        out += encode_string(name) + struct.pack("<I", len(shape))
        out += b"".join(struct.pack("<Q", d) for d in shape) + struct.pack("<IQ", type_, offset)
    out += b"\0" * (-len(out) % alignment)
    return bytes(out), offsets


def write_simple(
    path: str | os.PathLike[str],
    kv: list[KV],
    tensors: list[tuple[str, tuple[int, ...], int, bytes]],
    alignment: int = DEFAULT_ALIGNMENT,
) -> None:
    """Write a small GGUF from memory (tests, fixtures). `tensors` carry their data as bytes."""
    infos = [(n, s, t, len(d)) for n, s, t, d in tensors]
    header, _ = build_header(kv, infos, alignment)
    with open(path, "wb") as f:
        f.write(header)
        for _, _, _, data in tensors:
            f.write(data)
            f.write(b"\0" * (-len(data) % alignment))


# ------------------------------------------------------------------------------------------------ copying


class Cancelled(Exception):
    """Raised inside a step when the user cancelled the job."""


def read_chunks(path: str, offset: int, nbytes: int, chunk: int = CHUNK) -> Iterator[bytes]:
    with open(path, "rb") as f:
        f.seek(offset)
        left = nbytes
        while left > 0:
            data = f.read(min(chunk, left))
            if not data:
                raise GGUFError(f"{path} ends before the data of a tensor does (truncated file?)")
            left -= len(data)
            yield data


@dataclass
class Sink:
    """The output file while it is being written: counts bytes, hashes them, and checks for a cancel request."""

    f: BinaryIO
    hasher: Any = field(default_factory=hashlib.sha256)
    written: int = 0
    cancelled: Callable[[], bool] = lambda: False  # noqa: E731 - a default, replaced by the caller

    def write(self, data: bytes) -> None:
        if self.cancelled():
            raise Cancelled
        self.f.write(data)
        self.hasher.update(data)
        self.written += len(data)

    def pad(self, alignment: int) -> None:
        gap = -self.written % alignment
        if gap:
            self.write(b"\0" * gap)
