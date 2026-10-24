"""The local model library: GGUF files in the models folder, plus files elsewhere that the user added by path.

Split models (`…-00001-of-00002.gguf`) are one entry, listed under their first part, which is also what the engine
gets with -m. A model whose parts are still downloading shows as incomplete.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from . import catalog, gguf, paths

# Folders inside the models folder that never hold models.
_SKIP_DIRS = {"_downloads", "_engines", ".cache", ".git"}


def model_id(path: Path) -> str:
    key = str(path.resolve())
    return hashlib.sha1((key.lower() if paths.IS_WIN else key).encode("utf-8")).hexdigest()[:12]


def _kind(name: str) -> str:
    low = name.lower()
    if low.startswith("mtp-") or "-mtp-" in low:
        return "draft"
    if "mmproj" in low:
        return "mmproj"
    return "model"


class Library:
    def __init__(self, models_dir_getter: Any, data_dir: Path) -> None:
        self._models_dir = models_dir_getter  # callable: the folder can change in the settings
        self.file = data_dir / "library.json"
        self._lock = threading.Lock()
        self.imports: list[dict[str, Any]] = []
        try:
            data = json.loads(self.file.read_text(encoding="utf-8"))
            self.imports = [i for i in data.get("imports", []) if isinstance(i, dict) and i.get("path")]
        except (OSError, ValueError):
            pass

    @property
    def models_dir(self) -> Path:
        return Path(self._models_dir())

    def _save(self) -> None:
        paths.ensure(self.file.parent)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"imports": self.imports}, indent=2), encoding="utf-8")
        tmp.replace(self.file)

    # ------------------------------------------------------------------------------------------ scan

    def scan(self) -> list[dict[str, Any]]:
        """Every model on disk, newest state. Cheap after the first call: headers are cached per file version."""
        root = self.models_dir
        firsts: dict[str, tuple[Path, str]] = {}
        extras: dict[str, Path] = {}
        if root.exists():
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
                for fn in filenames:
                    p = Path(dirpath) / fn
                    if fn in (catalog.MTP_HEAD.name, catalog.MTP_VOCAB.name):
                        extras[fn] = p
                    if not fn.lower().endswith(".gguf"):
                        continue
                    m = gguf.SPLIT_RE.match(fn)
                    if m and m.group("no") != "00001":
                        first = p.with_name(f"{m.group('base')}-00001-of-{m.group('count')}.gguf")
                        if not first.exists():
                            firsts.setdefault(str(first), (first, "folder"))  # incomplete, first part still loading
                        continue
                    firsts[str(p)] = (p, "folder")
        with self._lock:
            imports = list(self.imports)
        for imp in imports:
            p = Path(imp["path"])
            firsts.setdefault(str(p), (p, "import"))
            for extra in (catalog.MTP_HEAD.name, catalog.MTP_VOCAB.name):
                for cand in (p.parent / extra, p.parent.parent / "mtp" / extra, p.parent / "mtp" / extra):
                    if extra not in extras and cand.exists():
                        extras[extra] = cand
        out = [self._describe(p, source, extras) for p, source in firsts.values()]
        out.sort(key=lambda m: (m["kind"] != "model", m["name"].lower()))
        return out

    def _describe(self, first: Path, source: str, extras: dict[str, Path]) -> dict[str, Any]:
        parts = gguf.split_parts(first)
        part_info = []
        for p in parts:
            partial = p.with_name(p.name + ".part")
            if p.exists():
                part_info.append({"name": p.name, "size": p.stat().st_size, "state": "present"})
            elif partial.exists():
                part_info.append({"name": p.name, "size": partial.stat().st_size, "state": "downloading"})
            else:
                part_info.append({"name": p.name, "size": 0, "state": "missing"})
        complete = all(pi["state"] == "present" for pi in part_info)
        cat = catalog.BY_FILENAME.get(first.name)
        info: dict[str, Any] | None = None
        error = ""
        if complete:
            try:
                info = gguf.inspect_cached(first).public()
            except (OSError, gguf.GGUFError) as exc:
                error = str(exc)
        name = cat.name if cat else (info or {}).get("name") or gguf.SPLIT_RE.sub(r"\g<base>", first.name)
        head = extras.get(catalog.MTP_HEAD.name)
        vocab = extras.get(catalog.MTP_VOCAB.name)
        return {
            "id": model_id(first),
            "name": str(name).removesuffix(".gguf"),
            "file": first.name,
            "path": str(first),
            "folder": str(first.parent),
            "parts": part_info,
            "size": sum(pi["size"] for pi in part_info),
            "complete": complete,
            "source": source,
            "catalog_id": cat.id if cat else None,
            "kind": _kind(first.name),
            "info": info,
            "error": error,
            "mtp": {
                "supported": bool(cat and cat.mtp),
                "head": str(head) if head and cat and cat.mtp else None,
                "vocab": str(vocab) if vocab and cat and cat.mtp else None,
            },
            "sampling": dict(cat.sampling) if cat else {},
        }

    def get(self, ident: str) -> dict[str, Any] | None:
        """A model by id or by path."""
        for m in self.scan():
            if ident in (m["id"], m["path"]):
                return m
        return None

    # ------------------------------------------------------------------------------------------ change

    def add(self, path_text: str) -> dict[str, Any]:
        """Add a GGUF file that lives outside the models folder (nothing is copied)."""
        p = Path(path_text.strip().strip('"')).expanduser()
        if not p.is_file():
            raise ValueError(f"No file at {p}")
        if p.suffix.lower() != ".gguf":
            raise ValueError("That is not a .gguf file")
        m = gguf.SPLIT_RE.match(p.name)
        if m and m.group("no") != "00001":
            p = p.with_name(f"{m.group('base')}-00001-of-{m.group('count')}.gguf")
        gguf.read_header(p, keep_array=0)  # raises GGUFError with a readable reason
        with self._lock:
            if not any(i["path"] == str(p) for i in self.imports):
                self.imports.append({"path": str(p), "added": int(time.time())})
                self._save()
        found = self.get(str(p))
        if not found:
            raise ValueError("The file was added but could not be read back")
        return found

    def remove(self, ident: str, delete_files: bool = False) -> None:
        m = self.get(ident)
        if not m:
            raise KeyError(ident)
        with self._lock:
            before = len(self.imports)
            self.imports = [i for i in self.imports if i["path"] != m["path"]]
            if len(self.imports) != before:
                self._save()
        if not delete_files:
            return
        root = self.models_dir.resolve()
        first = Path(m["path"]).resolve()
        if root not in first.parents:
            raise ValueError("Only files inside the models folder are deleted; files you added by path stay")
        for p in gguf.split_parts(first):
            for candidate in (p, p.with_name(p.name + ".part")):
                if candidate.exists():
                    candidate.unlink()
