"""Hugging Face Hub: a repo's files, the GGUF models in it, and a search. Public API; a token is only needed for
gated or private repos."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote, urlencode

from . import net
from .gguf import SPLIT_RE

HF = "https://huggingface.co"
_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_QUANT_RE = re.compile(
    r"(?i)(?:^|[-_.])((?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_MOE)?|TQ\d_\d)(?:-REAP\d+)?)(?=$|[-_.])"
)


def parse_repo(text: str) -> tuple[str, str, str]:
    """(repo, revision, subfolder or quant hint) from 'owner/name', a huggingface.co URL or
    'hf.co/owner/name:Q4_K_M'."""
    t = text.strip().strip("/")
    t = re.sub(r"^(https?://)?(www\.)?(huggingface\.co|hf\.co)/", "", t)
    hint = ""
    if ":" in t and "/" in t.split(":", 1)[0]:
        t, hint = t.split(":", 1)
    m = re.match(r"^([^/]+/[^/]+)(?:/(?:tree|blob|resolve)/([^/]+)(?:/(.*))?)?$", t)
    if not m or not _REPO_RE.match(m.group(1)):
        raise ValueError("Enter a Hugging Face repo like owner/name or its URL")
    return m.group(1), m.group(2) or "main", hint or (m.group(3) or "")


def resolve_url(repo: str, path: str, revision: str = "main") -> str:
    return f"{HF}/{repo}/resolve/{quote(revision, safe='')}/{quote(path)}"


def tree(repo: str, revision: str = "main", token: str = "") -> list[dict[str, Any]]:
    """All files of a repo (recursive), following the API's pagination."""
    url = f"{HF}/api/models/{repo}/tree/{quote(revision, safe='')}?recursive=true&expand=false"
    files: list[dict[str, Any]] = []
    for _ in range(50):
        data, headers = net.get_json(url, token=token)
        files += [f for f in data if f.get("type") == "file"]
        m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", "") or "")
        if not m:
            break
        url = m.group(1)
    return files


def quant_label(filename: str) -> str:
    stem = SPLIT_RE.sub(r"\g<base>", filename)
    stem = re.sub(r"\.gguf$", "", stem, flags=re.I)
    found = _QUANT_RE.findall(stem)
    return found[-1].upper() if found else stem


def gguf_groups(files: list[dict[str, Any]], hint: str = "") -> list[dict[str, Any]]:
    """The downloadable GGUF units of a repo: split parts belong together; projectors and MTP heads are marked."""
    groups: dict[str, dict[str, Any]] = {}
    for f in files:
        path = f.get("path", "")
        name = path.rsplit("/", 1)[-1]
        if not name.lower().endswith(".gguf"):
            continue
        m = SPLIT_RE.match(name)
        folder = path.rsplit("/", 1)[0] if "/" in path else ""
        key = f"{folder}/{m.group('base')}" if m else path
        low = name.lower()
        kind = "mmproj" if "mmproj" in low else "draft" if low.startswith("mtp-") else "model"
        g = groups.setdefault(key, {"key": key, "name": quant_label(name), "kind": kind, "files": [], "size": 0})
        lfs = f.get("lfs") or {}
        g["files"].append({"path": path, "size": int(f.get("size") or lfs.get("size") or 0), "sha256": lfs.get("oid")})
        g["size"] += int(f.get("size") or 0)
    out = sorted(groups.values(), key=lambda g: (g["kind"] != "model", g["size"]))
    for g in out:
        g["files"].sort(key=lambda x: x["path"])
        g["first"] = g["files"][0]["path"]
        g["parts"] = len(g["files"])
    if hint:
        h = hint.lower()
        out.sort(key=lambda g: (h not in g["key"].lower() and h not in g["name"].lower(), g["kind"] != "model"))
    return out


def repo_info(repo: str, token: str = "") -> dict[str, Any]:
    data, _ = net.get_json(f"{HF}/api/models/{repo}", token=token)
    card = data.get("cardData") or {}
    return {
        "id": data.get("id", repo),
        "gated": bool(data.get("gated")),
        "private": bool(data.get("private")),
        "downloads": data.get("downloads"),
        "likes": data.get("likes"),
        "license": card.get("license") or next((t[8:] for t in data.get("tags", []) if t.startswith("license:")), ""),
        "updated": data.get("lastModified"),
    }


def search(query: str, limit: int = 20, token: str = "") -> list[dict[str, Any]]:
    params = urlencode({"search": query, "filter": "gguf", "sort": "downloads", "direction": "-1", "limit": limit})
    data, _ = net.get_json(f"{HF}/api/models?{params}", token=token)
    return [
        {"id": d.get("id"), "downloads": d.get("downloads"), "likes": d.get("likes"), "updated": d.get("lastModified")}
        for d in data
    ]
