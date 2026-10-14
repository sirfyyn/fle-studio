"""Switching the model of an attached service by hand, from the model menu at the top right.

When a model switcher runs next to the service (for FYOS Titan: `rhea-wechsel`, Rhea and Katy on one GPU, address
http://127.0.0.1:18424), FLE Studio lists its models in the model menu and asks it to switch. The switcher decides
whether and when it can (free slots, no measurement running); FLE Studio only asks and shows the progress. Nothing
switches by itself: the loaded model stays until the next click.

Setting `switch_url` (or the environment variable FLE_STUDIO_SWITCH_URL); empty = no switching, the menu stays as it was.
The switcher speaks:  GET /status -> {"familien": {id: {"alias": ...}}, "geladen": [alias] | null, "wechsel_laeuft": bool,
"wechsel_ziel": id | null}   and   POST /umschalten {"ziel": id} -> {"ok": bool, ...} (answers when the switch is done).
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable

SWITCH_TIMEOUT = 1200  # a switch waits for free slots and a model load; the switcher itself gives up after 10 minutes


class Switch:
    def __init__(self, url: Callable[[], str]) -> None:
        self._url = url
        self._lock = threading.Lock()
        self._job: dict[str, Any] | None = None  # {"target", "started", "done", "ok", "error"}

    def url(self) -> str:
        return (os.environ.get("FLE_STUDIO_SWITCH_URL") or self._url() or "").rstrip("/")

    def _get(self, path: str) -> Any:
        with urllib.request.urlopen(self.url() + path, timeout=4) as r:
            return json.load(r)

    def status(self) -> dict[str, Any]:
        base = self.url()
        if not base:
            return {"available": False}
        out: dict[str, Any] = {"available": True, "targets": [], "active": None, "busy": False, "target": None, "error": ""}
        try:
            st = self._get("/status")
        except Exception as e:  # noqa: BLE001 - the page shows the reason, the menu still works
            out["available"] = False
            out["error"] = f"Model switcher not reachable: {e}"
            return out
        fams = st.get("familien") or {}
        loaded = set(st.get("geladen") or [])
        for fid, fam in fams.items():
            alias = fam.get("alias") or fid
            out["targets"].append({"id": fid, "name": alias, "active": alias in loaded})
            if alias in loaded:
                out["active"] = fid
        out["busy"] = bool(st.get("wechsel_laeuft"))
        out["target"] = st.get("wechsel_ziel")
        with self._lock:
            job = dict(self._job) if self._job else None
        if job:
            if not job["done"]:
                out["busy"] = True
                out["target"] = job["target"]
            elif not job["ok"] and time.time() - job["done_at"] < 120:
                out["error"] = job["error"]
        return out

    def start(self, target: str) -> dict[str, Any]:
        st = self.status()
        if not st.get("available"):
            raise ValueError(st.get("error") or "No model switcher is set up (setting switch_url).")
        if target not in {t["id"] for t in st["targets"]}:
            raise ValueError(f"Unknown model {target!r}.")
        if st.get("active") == target and not st["busy"]:
            return st
        with self._lock:
            if self._job and not self._job["done"]:
                raise RuntimeError("A switch is already running.")
            self._job = {"target": target, "started": time.time(), "done": False, "ok": False, "error": "", "done_at": 0.0}
            job = self._job
        threading.Thread(target=self._run, args=(job,), daemon=True, name="model-switch").start()
        return self.status()

    def _run(self, job: dict[str, Any]) -> None:
        try:
            req = urllib.request.Request(
                self.url() + "/umschalten",
                data=json.dumps({"ziel": job["target"], "grund": "FLE Studio"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=SWITCH_TIMEOUT) as r:
                    res = json.load(r)
            except urllib.error.HTTPError as e:
                try:
                    res = json.load(e)
                except Exception:  # noqa: BLE001
                    res = {"ok": False, "fehler": str(e)}
            job["ok"] = bool(res.get("ok"))
            if not job["ok"]:
                err = res.get("fehler") or (res.get("error") or {}).get("message") or "the switch did not work"
                job["error"] = f"Could not switch to {job['target']}: {err}"
        except Exception as e:  # noqa: BLE001
            job["error"] = f"Could not switch to {job['target']}: {e}"
        finally:
            job["done_at"] = time.time()
            job["done"] = True
