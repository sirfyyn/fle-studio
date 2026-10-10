"""Downloads: models from Hugging Face and engines from GitHub, one job at a time.

A file is written as `<name>.part` and renamed when it is complete and its SHA-256 matches. An interrupted file
continues where it stopped (HTTP Range); the bytes already on disk are hashed again first, so the final check still
covers the whole file. Jobs survive a restart of FLE Studio and continue by themselves.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import secrets
import shutil
import threading
import time
import urllib.error
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import net, paths, system

CHUNK = 1024 * 1024
MAX_RETRIES = 30


class _Stop(Exception):
    def __init__(self, why: str) -> None:
        super().__init__(why)
        self.why = why  # pause | cancel


class Downloads:
    def __init__(self, data_dir: Path, token: Callable[[], str]) -> None:
        self.file = data_dir / "downloads.json"
        self._token = token
        self._cv = threading.Condition()
        self.jobs: list[dict[str, Any]] = []
        self.hooks: dict[str, Callable[[dict[str, Any]], None]] = {}  # job kind -> called when a job is done
        try:
            for job in json.loads(self.file.read_text(encoding="utf-8")):
                if job.get("state") in ("downloading", "verifying", "queued"):
                    job["state"] = "queued"  # continue after a restart
                job["speed"] = 0.0
                job.pop("_flag", None)
                self.jobs.append(job)
        except (OSError, ValueError):
            pass
        threading.Thread(target=self._worker, name="downloads", daemon=True).start()

    # ------------------------------------------------------------------------------------------ API

    def add(self, *, title: str, files: list[dict[str, Any]], kind: str = "model", meta: dict[str, Any] | None = None,
            dedupe: str = "") -> dict[str, Any]:
        """Queue files [{url, dest, size, sha256}]. A job with the same dedupe key that is not finished is reused."""
        with self._cv:
            if dedupe:
                for job in self.jobs:
                    if job.get("dedupe") == dedupe and job["state"] not in ("done", "cancelled"):
                        if job["state"] in ("failed", "paused"):
                            job["state"], job["error"] = "queued", ""
                            self._cv.notify_all()
                        return self._public(job)
            job = {
                "id": secrets.token_hex(6),
                "title": title,
                "kind": kind,
                "dedupe": dedupe,
                "meta": meta or {},
                "state": "queued",
                "error": "",
                "created": time.time(),
                "finished": None,
                "files": [{"url": f["url"], "dest": str(f["dest"]), "size": int(f.get("size") or 0),
                           "sha256": (f.get("sha256") or "").lower() or None, "done": 0} for f in files],
                "speed": 0.0,
            }
            self.jobs.append(job)
            self._save_locked()
            self._cv.notify_all()
            return self._public(job)

    def action(self, ident: str, what: str) -> dict[str, Any]:
        with self._cv:
            job = next((j for j in self.jobs if j["id"] == ident), None)
            if not job:
                raise KeyError(ident)
            if what == "pause" and job["state"] in ("queued", "downloading", "verifying"):
                if job["state"] == "queued":
                    job["state"] = "paused"
                else:
                    job["_flag"] = "pause"
            elif what == "resume" and job["state"] in ("paused", "failed"):
                job["state"], job["error"] = "queued", ""
            elif what == "cancel" and job["state"] not in ("done", "cancelled"):
                if job["state"] in ("downloading", "verifying"):
                    job["_flag"] = "cancel"
                else:
                    job["state"] = "cancelled"
                    self._discard_parts(job)
            elif what == "remove" and job["state"] in ("done", "cancelled", "failed"):
                self.jobs.remove(job)
            self._save_locked()
            self._cv.notify_all()
            return self._public(job)

    def list(self) -> list[dict[str, Any]]:
        with self._cv:
            return [self._public(j) for j in self.jobs]

    def active_for(self, dest_prefix: str) -> dict[str, Any] | None:
        with self._cv:
            for j in self.jobs:
                if j["state"] in ("queued", "downloading", "verifying") and \
                        any(f["dest"].startswith(dest_prefix) for f in j["files"]):
                    return self._public(j)
        return None

    @staticmethod
    def _public(job: dict[str, Any]) -> dict[str, Any]:
        total = sum(f["size"] for f in job["files"])
        done = sum(f["done"] for f in job["files"])
        return {k: v for k, v in job.items() if not k.startswith("_")} | {
            "bytes_total": total, "bytes_done": done,
            "progress": (done / total) if total else (1.0 if job["state"] == "done" else 0.0)}

    # ------------------------------------------------------------------------------------------ worker

    def _save_locked(self) -> None:
        paths.ensure(self.file.parent)
        tmp = self.file.with_suffix(".tmp")
        keep = [{k: v for k, v in j.items() if not k.startswith("_")} for j in self.jobs
                if j["state"] != "done" or time.time() - (j.get("finished") or 0) < 7 * 86400]
        tmp.write_text(json.dumps(keep, indent=1), encoding="utf-8")
        tmp.replace(self.file)

    def _worker(self) -> None:
        while True:
            with self._cv:
                while not any(j["state"] == "queued" for j in self.jobs):
                    self._cv.wait()
                job = next(j for j in self.jobs if j["state"] == "queued")
                job["state"], job["error"] = "downloading", ""
                job.pop("_flag", None)
                self._save_locked()
            try:
                self._run(job)
                hook = self.hooks.get(job["kind"])
                if hook:
                    with self._cv:
                        job["state"] = "installing"
                    hook(job)
                with self._cv:
                    job["state"], job["finished"] = "done", time.time()
            except _Stop as stop:
                with self._cv:
                    job["state"] = "paused" if stop.why == "pause" else "cancelled"
                    if stop.why == "cancel":
                        self._discard_parts(job)
            except Exception as exc:  # report every other failure on the job
                with self._cv:
                    job["state"], job["error"] = "failed", _reason(exc)
            with self._cv:
                job["speed"] = 0.0
                job.pop("_flag", None)
                self._save_locked()

    def _check(self, job: dict[str, Any]) -> None:
        flag = job.get("_flag")
        if flag:
            raise _Stop(flag)

    def _run(self, job: dict[str, Any]) -> None:
        need = sum(max(0, f["size"] - f["done"]) for f in job["files"] if not Path(f["dest"]).exists())
        if job["files"]:
            target = Path(job["files"][0]["dest"]).parent
            paths.ensure(target)
            free = system.disk_free(target)
            if need and free and free < need + 512 * 1024 * 1024:
                raise RuntimeError(f"Not enough disk space in {target}: {need / 1e9:.1f} GB needed, "
                                   f"{free / 1e9:.1f} GB free")
        for f in job["files"]:
            self._fetch(job, f)

    def _fetch(self, job: dict[str, Any], f: dict[str, Any]) -> None:
        dest = Path(f["dest"])
        part = dest.with_name(dest.name + ".part")
        paths.ensure(dest.parent)
        if dest.exists() and (not f["size"] or dest.stat().st_size == f["size"]):
            f["done"] = dest.stat().st_size
            return
        h = hashlib.sha256() if f["sha256"] else None
        offset = part.stat().st_size if part.exists() else 0
        if f["size"] and offset > f["size"]:
            part.unlink()
            offset = 0
        if offset and h:
            job["state"] = "verifying"
            with part.open("rb") as src:
                seen = 0
                while seen < offset:
                    self._check(job)
                    block = src.read(min(8 * CHUNK, offset - seen))
                    if not block:
                        break
                    h.update(block)
                    seen += len(block)
                    f["done"] = seen
        f["done"] = offset
        job["state"] = "downloading"
        token = self._token() or ""
        tries = 0
        window_t, window_b = time.time(), offset
        while True:
            self._check(job)
            try:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with net.open_url(f["url"], headers=headers, token=token, timeout=60) as r:
                    if offset and r.status == 200:  # no range support: start over
                        offset, h = 0, (hashlib.sha256() if f["sha256"] else None)
                        part.unlink(missing_ok=True)
                    if not f["size"]:
                        length = r.headers.get("Content-Length")
                        f["size"] = offset + int(length) if length and length.isdigit() else 0
                    with part.open("ab") as out:
                        while True:
                            self._check(job)
                            chunk = r.read(CHUNK)
                            if not chunk:
                                break
                            out.write(chunk)
                            if h:
                                h.update(chunk)
                            offset += len(chunk)
                            f["done"] = offset
                            now = time.time()
                            if now - window_t >= 2:
                                rate = (offset - window_b) / (now - window_t)
                                job["speed"] = rate if not job["speed"] else 0.7 * job["speed"] + 0.3 * rate
                                window_t, window_b = now, offset
                if f["size"] and offset < f["size"]:
                    raise http.client.IncompleteRead(b"", f["size"] - offset)
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise RuntimeError("Access denied by the server. Gated repo? Accept its terms on Hugging Face "
                                       "and add a token in Settings.") from None
                if exc.code == 404:
                    raise RuntimeError(f"Not found: {f['url']}") from None
                if exc.code == 416 and f["size"] and offset >= f["size"]:
                    break
                tries = self._backoff(job, tries, exc)
            except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
                tries = self._backoff(job, tries, exc)
        if h and f["sha256"] and h.hexdigest() != f["sha256"]:
            bad = dest.with_name(dest.name + ".bad")
            part.replace(bad)
            f["done"] = 0
            raise RuntimeError(f"{dest.name} is damaged (checksum mismatch); it was moved to {bad.name}. "
                               "Resume to download it again.")
        part.replace(dest)

    def _backoff(self, job: dict[str, Any], tries: int, exc: Exception) -> int:
        tries += 1
        if tries > MAX_RETRIES:
            raise RuntimeError(f"Download failed after {MAX_RETRIES} attempts: {_reason(exc)}")
        job["error"] = f"Connection problem, retrying ({tries}/{MAX_RETRIES}): {_reason(exc)}"
        job["speed"] = 0.0
        deadline = time.time() + min(60, 2 ** min(tries, 6))
        while time.time() < deadline:
            self._check(job)
            time.sleep(0.5)
        job["error"] = ""
        return tries

    @staticmethod
    def _discard_parts(job: dict[str, Any]) -> None:
        for f in job["files"]:
            part = Path(f["dest"] + ".part")
            try:
                part.unlink(missing_ok=True)
            except OSError:
                pass
            f["done"] = 0 if not Path(f["dest"]).exists() else f["done"]


def _reason(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code} {exc.reason}"
    if isinstance(exc, urllib.error.URLError):
        return str(exc.reason)
    return str(exc) or exc.__class__.__name__


def move_tree(src: Path, dest: Path) -> None:
    """Replace dest by src (same file system), used when an extracted engine is complete."""
    if dest.exists():
        shutil.rmtree(dest)
    src.replace(dest)
