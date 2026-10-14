# Changelog

## [0.2.4] - 2026-10-14
### Fixes
- Remote: the block on pairing and host administration now also covers the bare `/api/remote` path and dot segments (`..`, `%2e`).
- Remote: large known-length answers are streamed instead of being read into memory whole (buffered up to 8 MiB).
- Tests: the fake forge tools no longer need an installed package.
### Cleanup
- Removed four unused functions.

## [0.2.3] - 2026-10-14

### Model menu

- **Switch the model by hand** from the model menu at the top right, for a service that runs next to a model switcher
  (setting `switch_url`, or the environment variable `FLE_STUDIO_SWITCH_URL`; for example Rhea and Katy on one GPU).
  The menu lists the models, marks the loaded one and switches on a click; the model that is loaded stays until you
  switch again. FLE Studio only asks: the switcher decides when it is safe (free slots, no measurement running) and
  FLE Studio shows the progress and, if it cannot switch, the reason. Without `switch_url` the menu is as before.

## [0.2.2] - 2026-10-13

### Live chat

- **Live**: a fixed entry at the top of the chat list. Everything that goes through the `/v1` address of FLE Studio
  (this chat page and any other program) shows up there as a temporary chat. Requests that repeat the earlier messages
  are joined into one conversation. It lives in memory only, keeps the newest 100 messages, and writes nothing to disk.
  Pictures are left out. The star on a conversation, or on one message, copies it (up to that message) into your
  normal chats as a favorite; chats that are favorites show a star in the list.

## [0.2.1] - 2026-10-12

### Capture

- **Capture** (Settings → Capture, off by default): keeps every chat answer that goes through FLE Studio as one line
  of JSON in `<data folder>/capture/`, with the request, the answer, timings and accepted draft tokens, for building your
  own corpus. Streams are put back together, pictures are left out, files are private, a size limit stops the recording.

### Fixed

- Forge: the folder typed into the tools box is no longer lost when the page redraws while the first search is still
  running.

## [0.2.0] - 2026-10-11

Studio 0.2 adds Forge, a chat with attachments, tools, MCP and apps, tuning and quality
measurements, an expert cache page, an update check and host/client mode.

### Forge: build your own quant

- Forge accepts the named quantization mixtures of `llama-quantize` (`q3_k_m`, `q4_k_m`, ...) as the default type.
- **Forge**, a new page for building your own quant (see [docs/FORGE.md](docs/FORGE.md)):
  - a hybrid builder that takes whole expert layers from several quants of the same model, with a size-budget
    calculator (exact knapsack over the layers), the exact file size and a RAM/VRAM preview before anything is written;
    the file is written without requantizing, with a correct header, and can be resumed after a stop;
  - quantizing a full-precision model with `llama-quantize` (per-tensor rules, optional importance matrix from
    `llama-imatrix`);
  - optional REAP pruning: the best experts per layer are kept, the router and the metadata are made consistent;
  - a check with `llama-perplexity` (KL divergence, top-1 agreement, perplexity ratio) and a result card;
  - jobs with progress and a log, cancel, resume and a record that survives a restart of Studio;
  - recipes (JSON schema 1) that carry no folders and no commands, to save, share and load;
  - publishing a result to the library with a start profile.
- A calibration text that ships with Studio (about 1 MB: English and German prose, code, dialogue, numbers; public
  domain, CC0 or MIT, with the origin, license and checksum of every file in `forge/corpus/SOURCES.md`), used by Forge
  as the default, split into a calibration and a held-out comparison part.
- Tests with synthetic mini GGUF files and stand-in llama.cpp tools; the browser test walks through Forge.

### Chat, tools and MCP

Details in [docs/CHAT.md](docs/CHAT.md) and [docs/MCP.md](docs/MCP.md).

- Attachments in the chat: text and code files, PDFs (optional `pypdf`, `pip install "fle-studio[pdf]"`) and pictures
  (only when the loaded model reports vision; otherwise a clear message). Choose, drop or paste files. A token
  estimate shows against the context window and warns from 80 %. The upload limit is a setting (Settings → Chat →
  Largest upload, 25 MB by default) instead of a fixed refusal.
- Tool calling through llama-server's OpenAI-style `tools` field, with the model, tool, model loop in the page and
  an openable card for every call (arguments and result). Built-in tools, each one switchable: calculator, current
  time, read files in one folder you choose (read only, nothing outside it), and web search (off by default, asks
  before each search, DuckDuckGo or your own SearXNG address).
- An MCP client for stdio and Streamable HTTP servers, configured in Settings → Tools with the usual `mcpServers`
  JSON. Their tools appear in the chat. A tool of an MCP server asks for approval before every call unless you allow
  that server; the server side enforces this too. A `autoApprove` entry in the JSON is ignored on purpose.
- MCP, complete (see [docs/MCP.md](docs/MCP.md)): web servers that need a sign-in get a **Sign in** button (OAuth per
  the MCP authorization specification 2025-11-25: 401 challenge, protected resource metadata, authorization server
  metadata, dynamic client registration or a client id you enter, PKCE S256, resource indicators, refresh, sign out).
  Tokens live in the data folder with mode 0600 and never in settings, logs or exports. The older HTTP+SSE transport is
  the fallback when Streamable HTTP is refused with a 4xx (or `"type": "sse"`). MCP Apps (SEP-1865, 2026-01-26): a tool
  with a `ui://` resource shows an app under its card in a sandboxed iframe (no allow-same-origin, own CSP from the
  resource metadata) with the postMessage bridge: tool input and result, light/dark, resize, `tools/call` to its own
  server with the usual confirmation, a message into the chat and links only after you agree.
- Edit your own messages: the edit becomes another version and the model answers it. "Answer again" keeps the
  previous answer as a version too. A `< 2 / 3 >` switch moves between versions.
- Search over the titles and the text of all chats (every version, attachment names and attachment text).
- Export a chat as JSON (every version, pictures inline) or Markdown (the version on screen), and import both again,
  plus plain OpenAI-style message lists.
- `GET /api/chat/capabilities`, `/api/tools`, `/api/attachments`, `/api/chats/search`, `/api/chats/<id>/export` and
  `/api/chats/import`.
- Tests: the stand-in engine can return `tool_calls` and accepts pictures with `FAKE_VISION=1`, a stand-in MCP server
  (`tests/fake_mcp_server.py`), unit and integration tests for all of it, a Node check of the request builder, and
  browser steps for attachments, versions, export and import, search, tool cards and MCP approval.

### Start, tuning and measuring

Details in [docs/TUNING-AND-QUALITY.md](docs/TUNING-AND-QUALITY.md).

- **Simple and advanced Engine page.** Simple (the default) shows the model, "Best for my machine" and Start; Advanced
  shows everything as before. Under the command, every flag and environment variable has a one-line explanation (also as a
  tooltip); the full command stays.
- **Best for my machine:** an auto-tuner that tries threads, expert cache or `--n-cpu-moe`, batch and context, measures
  each several times, keeps only what is clearly faster, and saves the winner as a launch profile. Probe run first,
  adjustable time budget, optional `llama-bench` screening for stock llama.cpp.
- **Repeats with median and 95 % interval** in Experiments, a comparison that says "within noise" when it is, and an
  **A/B/A comparison** of two profiles.
- **Quality next to speed:** KL divergence via `llama-perplexity`, and a knowledge probe with original CC0 questions
  (optional MKQA download). The speed test can ask the questions too.
- **Expert cache page** from the engine's log lines; [docs/EXPERT-CACHE-DATA.md](docs/EXPERT-CACHE-DATA.md) says what the
  engine would have to print for the per-layer view.
- Catalog: gpt-oss-20b and gpt-oss-120b (a second mixture-of-experts family, unmeasured).
- [docs/HARDWARE-TESTS.md](docs/HARDWARE-TESTS.md), [docs/TUNING-AND-QUALITY.md](docs/TUNING-AND-QUALITY.md).

### Updates

- Update check: about once a day (and with **Check now**) FLE Studio asks GitHub for the latest release, without
  identifiers or telemetry. A discreet card in the sidebar says "FLE Studio X.Y.Z is available", with the release notes
  as plain text, a link and **Later**. Switchable in Settings → Updates, with a switch for pre-releases. Offline it stays
  silent.
- One-click update for installs made with uv tool (not in service mode): installs the release, then restarts with the
  same command line. A failed install keeps the old version running and shows the log. On Windows a helper script does
  the install after Studio has ended. Other installs get the fitting command to copy.
- The same notice reports a newer FLE engine build for the engines you installed.

### Host and client

Details in [docs/REMOTE.md](docs/REMOTE.md).

- Host and client mode ([docs/REMOTE.md](docs/REMOTE.md)). A host (`--host`) serves TLS with a self-made certificate
  whose fingerprint is shown and pinned by clients; clients pair with a 10-minute single-use code and get a revocable
  token of their own (list with Revoke in Settings → Remote access). `--pair-code`, `--pair`, `--tls`, `--no-tls`.
- Client switcher at the top of the sidebar ("This machine / host / Add host…"). Hosts are added by address and code,
  or by SSH target; Studio runs and watches `ssh -N -L` itself (keys only). All pages, streaming chat included, run
  against the chosen host through a generic pass-through, so pages added later work too. The window shows the host's
  name and colour and reports an unreachable host with Reconnect.
- Rights per client: chat, look and measure by default; Admin for settings, installs and models; installs and updates
  ask for a confirmation on the client. Pairing is rate limited; tokens are stored as hashes and never logged.
- Optional read-only local copy of a host's chats for offline reading.
- Everything that changes something on a host needs the Admin right for a paired client (tools and MCP servers, Forge jobs, tuning, quality runs, updates); chat, attachments and speed tests do not.

### Changed

- Chats are stored as a tree of messages (version 2 of the file). Older chat files open unchanged. Pictures are kept
  as files next to the chats, not inside them. The size limit of one saved chat is 32 MB (was 4 MB).
- A request that carries tools and a forced answer format (`response_format`, `json_schema` or `grammar`) is refused
  by the proxy with HTTP 400, and the page never sends both: the model would not call the tools and would answer
  with substitute JSON. A test keeps this.
- The `tools.py` of the tuning work (the programs that ship with llama.cpp) is `engine_tools.py` now; `tools.py`
  holds the tools the model can call.
- `ruff format` is checked in CI, with a pinned ruff version.

## [0.1.1] - 2026-10-11

Installers, CI and docs. The app itself does not change in this release.

### Added

- Both installers can install from a folder or a wheel file on disk: `FLE_STUDIO_SOURCE=<path> sh install.sh`, or
  `install.ps1 -Source <path>`. No GitHub, no git and no PyPI are needed for that.
- CI: a `package` job on macOS, Linux and Windows builds the wheel and the source archive, installs the wheel into a
  clean environment, starts it and checks that the web files are inside.
- CI: an `installer` job on macOS, Linux and Windows runs `install.sh` and `install.ps1` from the checkout, so an
  installer change is tested before it is published.
- A test (`tests/test_release.py`) that keeps the version in the code, both installers, the README and this file
  the same.
- [docs/SERVICE-MODE.md](docs/SERVICE-MODE.md): running FLE Studio on a Linux server next to a systemd service
  (journal access, sudoers line, the two log options for the live panel, SSH tunnel, a user unit for Studio itself,
  and what the common messages mean).
- README: how to install from a local folder or wheel, and a pointer to the service-mode page.

### Changed

- The installers install `v0.1.1` by default (`FLE_STUDIO_REF` or `-Ref` still choose another release).

### Not in this release

- New speed measurements and screenshots of the Rhea runs. The figures and pictures in the README and the model
  catalog are still the ones from 0.1.0.

## [0.1.0] - 2026-10-11

First public release: chat, live inference data (speed, draft acceptance, GPU use, expert-cache hit rate), model
loader with fit check and verified downloads, launch profiles, speed experiments, engine installer for the FLE engine
and llama.cpp (CUDA, Metal, Vulkan, CPU), service mode for `llama-server` under systemd, installers for macOS, Linux
and Windows, tests and CI on all three systems with a browser end-to-end run against a stand-in engine.
