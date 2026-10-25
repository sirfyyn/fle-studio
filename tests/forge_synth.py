"""Tiny synthetic mixture-of-experts GGUF files for the Forge tests (a few KB, no real model anywhere).

Every tensor holds pseudo-random bytes derived from (seed, tensor name), so two files with different seeds differ in
every tensor and a copied tensor can be told apart from one that came from the wrong source.
"""

from __future__ import annotations

import random
import struct
from pathlib import Path

from fle_studio_forge import ggufio

HIDDEN, INTER, VOCAB = 64, 32, 16


def data(seed: int, name: str, nbytes: int) -> bytes:
    return random.Random(f"{seed}/{name}").randbytes(nbytes)


def tensor(seed: int, name: str, shape: tuple[int, ...], type_name: str) -> tuple[str, tuple[int, ...], int, bytes]:
    tid = ggufio.type_id(type_name)
    assert tid is not None
    nbytes = ggufio.tensor_nbytes(tid, shape)
    assert nbytes is not None, f"{name}: {shape} is not a whole number of {type_name} blocks"
    return name, shape, tid, data(seed, name, nbytes)


def kv_list(layers: int, experts: int, used: int = 2, extra: list[ggufio.KV] | None = None) -> list[ggufio.KV]:
    u32, string = ggufio.T_U32, ggufio.T_STRING
    kv = [
        ggufio.kv_scalar("general.architecture", string, "testmoe"),
        ggufio.kv_scalar("general.name", string, "Forge test MoE"),
        ggufio.kv_scalar("general.file_type", u32, 7),
        ggufio.kv_scalar("testmoe.context_length", u32, 4096),
        ggufio.kv_scalar("testmoe.block_count", u32, layers),
        ggufio.kv_scalar("testmoe.embedding_length", u32, HIDDEN),
        ggufio.kv_scalar("testmoe.expert_count", u32, experts),
        ggufio.kv_scalar("testmoe.expert_used_count", u32, used),
        ggufio.kv_scalar("testmoe.attention.head_count", u32, 4),
        ggufio.kv_scalar("testmoe.attention.head_count_kv", u32, 2),
        ggufio.kv_string_array("tokenizer.ggml.tokens", [f"tok{i}" for i in range(VOCAB)]),
    ]
    return kv + (extra or [])


DEFAULT_TYPES = ("Q4_0", "Q4_0", "Q4_0")  # gate, up, down


def make_model(
    path: str | Path,
    *,
    seed: int,
    layers: int = 4,
    experts: int = 8,
    used: int = 2,
    types: list[tuple[str, str, str]] | None = None,
    with_bias: bool = True,
) -> Path:
    """Write a model of `layers` MoE layers; `types[layer]` is the (gate, up, down) type of that layer."""
    types = types or [DEFAULT_TYPES] * layers
    tensors = [tensor(seed, "token_embd.weight", (HIDDEN, VOCAB), "Q8_0")]
    for layer in range(layers):
        gate, up, down = types[layer]
        p = f"blk.{layer}"
        tensors += [
            tensor(seed, f"{p}.attn_norm.weight", (HIDDEN,), "F32"),
            tensor(seed, f"{p}.attn_q.weight", (HIDDEN, HIDDEN), "Q8_0"),
            tensor(seed, f"{p}.ffn_gate_inp.weight", (HIDDEN, experts), "F32"),
            tensor(seed, f"{p}.ffn_gate_exps.weight", (HIDDEN, INTER, experts), gate),
            tensor(seed, f"{p}.ffn_up_exps.weight", (HIDDEN, INTER, experts), up),
            tensor(seed, f"{p}.ffn_down_exps.weight", (INTER, HIDDEN, experts), down),
        ]
        if with_bias:
            tensors.append(tensor(seed, f"{p}.exp_probs_b.bias", (experts,), "F32"))
    tensors += [
        tensor(seed, "output_norm.weight", (HIDDEN,), "F32"),
        tensor(seed, "output.weight", (HIDDEN, VOCAB), "Q8_0"),
    ]
    ggufio.write_simple(path, kv_list(layers, experts, used), tensors)
    return Path(path)


def make_split(first: str | Path, *, seed: int, parts: int = 2, layers: int = 4, experts: int = 8) -> Path:
    """The same model as make_model, cut into `parts` files named NAME-00001-of-0000N.gguf; returns the first."""
    whole = Path(first).with_name("whole.gguf")
    make_model(whole, seed=seed, layers=layers, experts=experts)
    m = ggufio.open_model(whole)
    names = m.order
    per = -(-len(names) // parts)
    base = Path(first).name.removesuffix(".gguf")
    out = []
    for i in range(parts):
        chunk = names[i * per : (i + 1) * per]
        tensors = []
        for n in chunk:
            path, offset, nbytes = m.location(n)
            with open(path, "rb") as f:
                f.seek(offset)
                tensors.append((n, m.tensors[n].shape, m.tensors[n].type, f.read(nbytes)))
        kv = [k for k in m.kv] + [
            ggufio.kv_scalar("split.no", ggufio.T_U16, i),
            ggufio.kv_scalar("split.count", ggufio.T_U16, parts),
        ]
        target = Path(first).with_name(f"{base}-{i + 1:05d}-of-{parts:05d}.gguf")
        ggufio.write_simple(target, kv, tensors)
        out.append(target)
    whole.unlink()
    return out[0]


def make_imatrix(path: str | Path, layers: int, experts: int, counts: dict[int, list[int]], width: int = 4) -> None:
    """An importance-matrix GGUF with the per-expert counts and input energy of the layers' gate experts."""
    tensors = []
    for layer in range(layers):
        c = counts[layer]
        energy = [float(x) * (1 + i % 3) for x in c for i in range(width)]
        tensors.append(
            (
                f"blk.{layer}.ffn_gate_exps.weight.counts",
                (experts,),
                0,
                struct.pack(f"<{experts}f", *[float(x) for x in c]),
            )
        )
        tensors.append(
            (
                f"blk.{layer}.ffn_gate_exps.weight.in_sum2",
                (width, experts),
                0,
                struct.pack(f"<{width * experts}f", *energy),
            )
        )
    ggufio.write_simple(path, [ggufio.kv_scalar("general.type", ggufio.T_STRING, "imatrix")], tensors)
