# Changelog

## [Unreleased]

Version `0.2.0.dev0`. Not released.

### Added

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

### Changed

- The release test accepts a development version: installers and README name the last release until the new one
  is cut.

## Unreleased

Chat 0.2: attachments, tools and MCP, and the comforts of a daily chat. See [docs/CHAT.md](docs/CHAT.md).

### Added

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

### Changed

- Chats are stored as a tree of messages (version 2 of the file). Older chat files open unchanged. Pictures are kept
  as files next to the chats, not inside them. The size limit of one saved chat is 32 MB (was 4 MB).
- A request that carries tools and a forced answer format (`response_format`, `json_schema` or `grammar`) is refused
  by the proxy with HTTP 400, and the page never sends both: the model would not call the tools and would answer
  with substitute JSON. A test keeps this.

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
