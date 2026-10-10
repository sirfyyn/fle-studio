"""Engines: which llama-server builds fit this machine, installing them from GitHub releases, and checking a
llama-server of your own.

- FLE engine (sirfyyn/FastLocalExperts releases): NVIDIA GPUs on Linux and Windows x64 with AVX2.
- Upstream llama.cpp (ggml-org/llama.cpp, the tag in the settings): Metal on Apple Silicon, Vulkan for AMD and
  Intel GPUs, CUDA as the alternative on NVIDIA, CPU everywhere.

Asset names are matched by pattern against the release's own asset list, and each download is checked against the
SHA-256 digest GitHub publishes for it.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

from . import net, paths
from .downloads import Downloads, move_tree

UPSTREAM_REPO = "ggml-org/llama.cpp"
FLE_REPO = "sirfyyn/FastLocalExperts"
GITHUB_API = "https://api.github.com/repos"

# CUDA major.minor -> lowest NVIDIA driver branch that runs it.
CUDA_DRIVER = {"13": 580, "12.8": 570, "12.9": 575, "12.6": 560, "12.4": 550}


def _os_arch() -> tuple[str, str]:
    osname = "macos" if paths.IS_MAC else "win" if paths.IS_WIN else "linux"
    m = platform.machine().lower()
    arch = "arm64" if m in ("arm64", "aarch64") else "x64" if m in ("x86_64", "amd64") else m
    return osname, arch


def _driver_major(sysinfo: dict[str, Any]) -> int:
    for g in sysinfo.get("gpus", []):
        if g.get("vendor") == "nvidia" and g.get("driver"):
            try:
                return int(str(g["driver"]).split(".")[0])
            except ValueError:
                pass
    return 0


def _cuda_ok(version: str, driver: int) -> bool:
    need = CUDA_DRIVER.get(version) or CUDA_DRIVER.get(version.split(".")[0]) or 999
    return driver >= need


def variants(sysinfo: dict[str, Any]) -> list[dict[str, Any]]:
    """Engine choices for this machine, best first, each with the reason it is offered."""
    osname, arch = _os_arch()
    has_nvidia = any(g.get("vendor") == "nvidia" for g in sysinfo.get("gpus", []))
    avx2 = bool(sysinfo.get("cpu", {}).get("avx2"))
    others = sysinfo.get("other_gpus") or []
    out: list[dict[str, Any]] = []

    def add(ident: str, kind: str, backend: str, label: str, why: str) -> None:
        out.append({"id": ident, "kind": kind, "backend": backend, "label": label, "why": why})

    if osname == "macos":
        if arch == "arm64":
            add("upstream-metal", "upstream", "metal", "llama.cpp for Apple Silicon (Metal)",
                "Runs the model on the Apple GPU with unified memory.")
        else:
            add("upstream-macos-x64", "upstream", "cpu", "llama.cpp for Intel Macs", "CPU build for Intel Macs.")
        return out
    if has_nvidia and arch == "x64":
        if avx2:
            add("fle-cuda", "fle", "cuda", "FLE engine (NVIDIA CUDA)",
                "The FLE build: GPU expert cache, MTP drafting, pinned expert loading.")
        add("upstream-cuda", "upstream", "cuda", "llama.cpp (NVIDIA CUDA)", "Upstream llama.cpp on the NVIDIA GPU.")
    if others or has_nvidia:
        add("upstream-vulkan", "upstream", "vulkan", "llama.cpp (Vulkan)",
            "Works on AMD, Intel and NVIDIA GPUs." if others else "Fallback for GPUs without a CUDA build.")
    add("upstream-cpu", "upstream", "cpu", "llama.cpp (CPU only)", "Works everywhere; slow for large models.")
    return out


def _asset_patterns(variant: str, osname: str, arch: str) -> list[str]:
    """Regexes for the release assets of a variant; the first is the engine, CUDA builds add their runtime."""
    ext = r"\.zip" if osname == "win" else r"\.tar\.gz"
    plat = {"linux": "ubuntu", "win": "win", "macos": "macos"}[osname]
    if variant == "upstream-metal" or variant == "upstream-macos-x64":
        return [rf"^llama-b\d+-bin-macos-{arch}{ext}$"]
    if variant == "upstream-cpu":
        cpu = "cpu-" if osname == "win" else ""
        return [rf"^llama-b\d+-bin-{plat}-{cpu}{arch}{ext}$"]
    if variant == "upstream-vulkan":
        return [rf"^llama-b\d+-bin-{plat}-vulkan-{arch}{ext}$"]
    if variant == "upstream-cuda":
        return [rf"^llama-b\d+-bin-{plat}-cuda-(?P<cuda>[\d.]+)-{arch}{ext}$"]
    if variant == "fle-cuda":
        return [rf"^fle-.+-bin-{plat}-cuda-(?P<cuda>[\d.]+)-{arch}{ext}$"]
    raise ValueError(f"unknown engine {variant}")


def _cudart_pattern(osname: str, arch: str, cuda: str) -> str:
    if osname == "win":
        return rf"^cudart-llama-bin-win-cuda-{re.escape(cuda)}-{arch}\.zip$"
    return rf"^cudart-llama-b\d+-bin-ubuntu-cuda-{re.escape(cuda)}-{arch}\.tar\.gz$"


class Engines:
    def __init__(self, data_dir: Path, downloads: Downloads, settings: Any) -> None:
        self.dir = paths.ensure(data_dir / "engines")
        self.file = self.dir / "engines.json"
        self.downloads = downloads
        self.settings = settings
        self._lock = threading.Lock()
        self._detect_cache: dict[tuple[str, int], dict[str, Any]] = {}
        try:
            self.items: list[dict[str, Any]] = json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.items = []
        downloads.hooks["engine"] = self._install_downloaded

    def _save(self) -> None:
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.items, indent=1), encoding="utf-8")
        tmp.replace(self.file)

    def installed(self) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self.items if Path(e["path"]).exists()]

    # ------------------------------------------------------------------------------------------ install

    def _release(self, repo: str, tag: str) -> dict[str, Any]:
        url = f"{GITHUB_API}/{repo}/releases/" + ("latest" if tag in ("", "latest") else f"tags/{tag}")
        data, _ = net.get_json(url, headers={"Accept": "application/vnd.github+json"})
        return data

    def plan_install(self, variant: str, sysinfo: dict[str, Any]) -> dict[str, Any]:
        """Which release assets an install would download (raises with a readable reason if none fits)."""
        osname, arch = _os_arch()
        eng = self.settings.get("engine")
        if variant.startswith("fle"):
            rel = self._release(FLE_REPO, eng.get("fle_tag", "latest"))
            source = FLE_REPO
        else:
            rel = self._release(UPSTREAM_REPO, eng.get("upstream_tag", "b11540"))
            source = UPSTREAM_REPO
        assets = rel.get("assets") or []
        pattern = _asset_patterns(variant, osname, arch)[0]
        driver = _driver_major(sysinfo)
        matches = []
        for a in assets:
            m = re.match(pattern, a["name"])
            if m:
                cuda = m.groupdict().get("cuda")
                if cuda and not _cuda_ok(cuda, driver):
                    continue
                matches.append((tuple(int(x) for x in (cuda or "0").split(".")), a, cuda))
        if not matches:
            hint = ""
            if "cuda" in variant and driver:
                hint = f" (NVIDIA driver {driver} is too old for the CUDA builds in it)"
            raise RuntimeError(f"No {variant} build for {osname}-{arch} in {source} {rel.get('tag_name', '')}{hint}")
        matches.sort(key=lambda t: t[0], reverse=True)
        _, main, cuda = matches[0]
        files = [main]
        if cuda:
            pat = _cudart_pattern(osname, arch, cuda)
            runtime = next((a for a in assets if re.match(pat, a["name"])), None)
            if runtime is None and source == FLE_REPO:  # FLE builds use upstream's CUDA runtime package
                up = self._release(UPSTREAM_REPO, eng.get("upstream_tag", "b11540"))
                runtime = next((a for a in up.get("assets") or [] if re.match(pat, a["name"])), None)
            if runtime is None and osname == "win":
                raise RuntimeError(f"The CUDA {cuda} runtime package is missing from the release")
            if runtime:
                files.append(runtime)
        return {"variant": variant, "tag": rel.get("tag_name", ""), "source": source, "cuda": cuda,
                "assets": [{"name": a["name"], "url": a["browser_download_url"], "size": a.get("size", 0),
                            "sha256": (a.get("digest") or "").removeprefix("sha256:") or None} for a in files]}

    def install(self, variant: str, sysinfo: dict[str, Any]) -> dict[str, Any]:
        plan = self.plan_install(variant, sysinfo)
        stage = paths.ensure(self.dir / "_downloads")
        files = [{"url": a["url"], "dest": stage / a["name"], "size": a["size"], "sha256": a["sha256"]}
                 for a in plan["assets"]]
        label = next((v["label"] for v in variants(sysinfo) if v["id"] == variant), variant)
        return self.downloads.add(title=f"{label} {plan['tag']}", files=files, kind="engine",
                                  meta={"variant": variant, "tag": plan["tag"], "cuda": plan["cuda"]},
                                  dedupe=f"engine:{variant}:{plan['tag']}")

    def _install_downloaded(self, job: dict[str, Any]) -> None:
        """Download hook: unpack the archives into engines/<variant>-<tag>/ and register the engine."""
        meta = job["meta"]
        name = re.sub(r"[^\w.-]+", "_", f"{meta['variant']}-{meta['tag']}")
        target = self.dir / name
        tmp = self.dir / f".{name}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        for f in job["files"]:
            _extract(Path(f["dest"]), tmp)
        server = _find_server(tmp)
        if server is None:
            raise RuntimeError("The engine archive has no llama-server in it")
        rel = server.relative_to(tmp)
        bindir = server.parent
        # CUDA runtime libraries land beside the server, wherever the archive put them.
        for lib in list(tmp.rglob("*")):
            if lib.is_file() and lib.parent != bindir and re.match(r"^(lib)?cu(da|blas|blasLt|rt)", lib.name, re.I):
                lib.replace(bindir / lib.name)
        if not paths.IS_WIN:
            for p in bindir.iterdir():
                if p.is_file() and (p.suffix == "" or ".so" in p.name or p.suffix == ".dylib"):
                    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        move_tree(tmp, target)
        path = target / rel
        info = self.detect(str(path))
        kind = "fle" if meta["variant"].startswith("fle") else "upstream"
        entry = {"id": name, "variant": meta["variant"], "kind": kind,
                 "flavor": info.get("flavor"), "backend": info.get("backend"), "tag": meta["tag"], "path": str(path),
                 "version": info.get("version", ""), "installed": time.time(), "ok": info.get("ok", False),
                 "problem": info.get("problem", "")}
        with self._lock:
            self.items = [e for e in self.items if e["id"] != name] + [entry]
            self._save()
        for f in job["files"]:  # the archives are no longer needed
            try:
                Path(f["dest"]).unlink()
            except OSError:
                pass
        if not entry["ok"]:
            raise RuntimeError(f"Installed, but the engine does not start: {entry['problem']}")

    def remove(self, ident: str) -> None:
        with self._lock:
            entry = next((e for e in self.items if e["id"] == ident), None)
            if not entry:
                raise KeyError(ident)
            self.items = [e for e in self.items if e["id"] != ident]
            self._save()
        folder = self.dir / ident
        if folder.exists() and self.dir in folder.resolve().parents:
            shutil.rmtree(folder, ignore_errors=True)

    # ------------------------------------------------------------------------------------------ use

    def detect(self, path: str) -> dict[str, Any]:
        """Run `llama-server --version` and `--help` once per binary version.

        Tells whether it starts, whether it is the FLE engine, and which backend it uses."""
        p = Path(path).expanduser()
        if not p.is_file():
            return {"ok": False, "problem": f"No file at {p}"}
        key = (str(p), int(p.stat().st_mtime))
        if key in self._detect_cache:
            return self._detect_cache[key]
        cmd = [str(p)]
        if p.suffix == ".py":
            cmd = [sys.executable, str(p)]
        env = os.environ.copy()
        if paths.IS_LINUX:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(x for x in (str(p.parent), env.get("LD_LIBRARY_PATH")) if x)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            ver = subprocess.run([*cmd, "--version"], capture_output=True, text=True, timeout=60, env=env,
                                 cwd=str(p.parent), creationflags=flags, errors="replace")
            hlp = subprocess.run([*cmd, "--help"], capture_output=True, text=True, timeout=60, env=env,
                                 cwd=str(p.parent), creationflags=flags, errors="replace")
        except (OSError, subprocess.SubprocessError) as exc:
            return {"ok": False, "problem": str(exc)}
        out = (ver.stdout or "") + (ver.stderr or "")
        m = re.search(r"version:\s*(\S+)(?:\s*\(([0-9a-f]+)\))?", out)
        text = out.lower() + " " + " ".join(x.name.lower() for x in p.parent.iterdir())
        backend = ("cuda" if "cuda" in text else "metal" if "metal" in text or paths.IS_MAC else
                   "vulkan" if "vulkan" in text else "cpu")
        flavor = "fle" if "--moe-expert-cache-inserts" in (hlp.stdout or "") + (hlp.stderr or "") else "upstream"
        ok = ver.returncode == 0 and bool(m)
        result = {"ok": ok, "version": (m.group(1) + (f" ({m.group(2)})" if m.group(2) else "")) if m else "",
                  "backend": backend, "flavor": flavor,
                  "problem": "" if ok else (out.strip().splitlines() or ["no output"])[-1][:300]}
        self._detect_cache[key] = result
        return result

    def resolve(self, want: str, sysinfo: dict[str, Any]) -> dict[str, Any]:
        """The engine a launch should use. want: auto | fle | upstream | custom."""
        eng = self.settings.get("engine")
        if want == "custom" or (want == "auto" and eng.get("choice") == "custom"):
            path = eng.get("custom_path", "")
            if not path:
                raise RuntimeError("Set the path of your llama-server on the Engine page")
            info = self.detect(path)
            if not info.get("ok"):
                raise RuntimeError(f"Your llama-server does not start: {info.get('problem')}")
            return {"kind": "custom", "flavor": info["flavor"], "backend": info["backend"],
                    "path": str(Path(path).expanduser()), "version": info.get("version", "")}
        if want == "auto":
            want = eng.get("choice", "auto")
        order = [v["id"] for v in variants(sysinfo)]
        have = sorted(self.installed(), key=lambda e: (order.index(e["variant"]) if e["variant"] in order else 99,
                                                       -e.get("installed", 0)))
        if want in ("fle", "upstream"):
            have = [e for e in have if e["kind"] == want]
        if not have:
            raise RuntimeError("Install an engine first (Engine page)")
        e = have[0]
        return {"kind": e["kind"], "flavor": e.get("flavor") or e["kind"], "backend": e.get("backend"),
                "path": e["path"], "version": e.get("version", ""), "id": e["id"]}


def _find_server(root: Path) -> Path | None:
    name = paths.exe("llama-server")
    hits = sorted(root.rglob(name), key=lambda p: len(p.parts))
    return hits[0] if hits else None


def _extract(archive: Path, dest: Path) -> None:
    """Unpack a .zip or .tar.gz safely: nothing may land outside dest."""
    root = dest.resolve()
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                target = (dest / member.filename).resolve()
                if root != target and root not in target.parents:
                    raise RuntimeError(f"unsafe path in archive: {member.filename}")
            z.extractall(dest)
        return
    with tarfile.open(archive, "r:*") as t:
        if hasattr(tarfile, "data_filter"):
            t.extractall(dest, filter="data")
            return
        for member in t.getmembers():  # Python without extraction filters
            target = (dest / member.name).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"unsafe path in archive: {member.name}")
            if not (member.isfile() or member.isdir() or member.issym()):
                raise RuntimeError(f"unsupported entry in archive: {member.name}")
            if member.issym():
                link = (target.parent / member.linkname).resolve()
                if root != link and root not in link.parents:
                    raise RuntimeError(f"unsafe link in archive: {member.name}")
        t.extractall(dest)  # noqa: S202 - members checked above
