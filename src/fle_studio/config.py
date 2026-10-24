"""Settings: one JSON file in the data folder, merged over the defaults below.

Only keys that exist in DEFAULTS can be set, and every value is coerced to the type of its default, so a
hand-edited or outdated file cannot put the app into a state the UI does not know. Secrets (the Hugging Face
token and the access key for other machines) never leave the server: `public()` replaces them with flags.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any

from . import paths

DEFAULTS: dict[str, Any] = {
    "models_dir": "",  # empty: <data folder>/models
    "server": {
        "host": "127.0.0.1",  # 0.0.0.0 makes FLE Studio reachable from other machines (needs access_key)
        "port": 8642,
        "open_browser": True,
        "allowed_hosts": "",  # LAN mode: extra host names the page may be opened under (IP addresses always work)
        "tls": True,  # beyond this machine (host mode): TLS with a self-made certificate; --no-tls turns it off
    },
    "remote": {
        "name": "",  # how this machine calls itself to clients and hosts; empty: its host name
    },
    "access_key": "",  # required from other machines whenever host is not loopback
    "hf_token": "",  # optional, only for gated Hugging Face repos
    "engine": {
        "choice": "auto",  # auto | fle | upstream | custom
        "custom_path": "",  # a llama-server binary of your own
        "upstream_tag": "b11540",  # ggml-org/llama.cpp release that FLE Studio was tested with
        "fle_tag": "latest",  # sirfyyn/FastLocalExperts release with prebuilt FLE engines
    },
    "launch": {
        "profile": "auto",  # auto, or the id of a saved launch profile (a complete command)
        "context": 32768,
        "parallel": 1,  # requests served at the same time (each gets context / parallel tokens)
        "mtp": True,  # use the MTP head when the FLE engine and the head file are there
        "spec": "auto",  # speculation: auto, none, mtp, ngram-mod or ngram-simple
        "gpu_cache_mib": 0,  # GPU expert cache budget; 0 = size it from free GPU memory
        "gpus": "auto",  # auto (the largest NVIDIA GPU) | all | a list like "0" or "0,1" (nvidia-smi numbering)
        "threads": 0,  # 0 = one thread per physical core
        "cpu_mask": "",  # e.g. 0x0000FFFF: pin the engine threads to these cores (-C … --cpu-strict 1)
        "pinned_load": "auto",  # auto | on | off: --load-mode none (experts in pinned RAM)
        "prompt_cache_mib": -1,  # RAM prompt cache in MiB; -1 = automatic, 0 = off
        "port": 0,  # engine port; 0 = pick a free one
        "extra_args": "",  # appended to the engine command line, for experts
        "extra_env": "",  # KEY=VALUE lines added to the engine environment
        "restart_on_crash": False,  # start the engine again if it exits on its own (servers that should stay up)
        "autostart": False,  # load the last model with the last profile when FLE Studio starts
    },
    "chat": {
        "sampling": "model",  # model: the model's recommended values; custom: the values below
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "max_tokens": 0,  # 0 = until the model stops or the context is full
        "thinking": True,
        "system_prompt": "",
        "max_upload_mb": 25,  # largest file that can be attached to a message (1 to 512)
    },
    "tools": {  # what the model may call in a chat; see tools.py
        "enabled": True,  # master switch; everything below is off or careful by default
        "max_rounds": 8,  # model → tool → model steps for one answer
        "call_timeout": 60,  # seconds one MCP tool call may take
        "calculator": True,
        "datetime": True,
        "files": False,  # read files in files_dir, nowhere else
        "files_dir": "",
        "web_search": False,
        "web_search_confirm": True,  # ask before every search, because the query leaves this computer
        "web_search_url": "",  # a SearXNG address (…/search); empty = DuckDuckGo's plain HTML page
        "mcp_enabled": True,
        "mcp_json": "",  # {"mcpServers": {…}}
        "mcp_trusted": [],  # names of MCP servers whose tools run without asking
    },
    "capture": {  # keep what you do with the model as files on this computer; see capture.py
        "enabled": False,  # off until you switch it on; nothing is sent anywhere
        "max_mb": 2048,  # stop recording at this size (0 = no limit)
    },
    "forge": {
        "tools_dir": "",  # folder with llama-imatrix / llama-quantize / llama-perplexity; empty = engines, then PATH
    },
    "attach_url": "",  # talk to an already running llama.cpp/FLE server instead of starting one
    "attach_service": "",  # Linux: the systemd service that runs it (FLE Studio can then start and stop it)
    # a model switcher next to the service (FYOS: rhea-wechsel); the model menu then switches Rhea/Katy by hand
    "switch_url": "",
    "attach_key": "",  # API key of that server, if it wants one
    "attach_auto": False,  # connect again when FLE Studio starts (Connect sets it, Disconnect clears it)
    "last_model": "",
    "theme": "system",  # system | dark | light
    "updates": {
        "check": True,  # ask GitHub for a newer FLE Studio (and FLE engine) about once a day
        "prereleases": False,  # also offer pre-releases (release candidates, betas)
        "dismissed": "",  # the version whose notice was hidden with "Later"
        "dismissed_engines": "",  # the engine updates whose notice was hidden, comma separated
    },
}

SECRETS = ("hf_token", "access_key", "attach_key")


def _coerce(value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(default, int):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    if isinstance(default, str):
        return "" if value is None else str(value)
    return value


def _merge(base: dict[str, Any], patch: dict[str, Any], *, strict: bool) -> dict[str, Any]:
    """Return base with patch applied. Unknown keys are dropped; with strict=True a bad value raises."""
    out = copy.deepcopy(base)
    for key, value in (patch or {}).items():
        if key not in base:
            continue
        if isinstance(base[key], dict):
            if isinstance(value, dict):
                out[key] = _merge(base[key], value, strict=strict)
            continue
        try:
            out[key] = _coerce(value, base[key])
        except (TypeError, ValueError):
            if strict:
                raise ValueError(f"invalid value for {key}: {value!r}") from None
    return out


class Config:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or paths.data_dir() / "settings.json"
        self._lock = threading.Lock()
        self.data = copy.deepcopy(DEFAULTS)
        if self.path.exists():
            try:
                self.data = _merge(DEFAULTS, json.loads(self.path.read_text(encoding="utf-8")), strict=False)
            except (OSError, ValueError):
                # A broken file must not stop the app; keep it for the user and start from defaults.
                self.path.replace(self.path.with_suffix(".broken.json"))

    def get(self, dotted: str) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            node = node[part]
        return node

    def update(self, patch: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.data = _merge(self.data, patch, strict=True)
            self._save()
        return self.public()

    def _save(self) -> None:
        paths.ensure(self.path.parent)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        if not paths.IS_WIN:
            os.chmod(tmp, 0o600)  # the file can hold a Hugging Face token
        tmp.replace(self.path)

    def public(self) -> dict[str, Any]:
        out = copy.deepcopy(self.data)
        for key in SECRETS:
            out[f"has_{key}"] = bool(out.pop(key))
        out["models_dir"] = str(self.models_dir)
        return out

    @property
    def models_dir(self) -> Path:
        # The default sits next to settings.json, so --home and portable installs keep their models with them.
        configured = self.data["models_dir"]
        return Path(configured).expanduser() if configured else self.path.parent / "models"
