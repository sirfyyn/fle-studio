# Changelog

## [Unreleased]

Version `0.2.0.dev0`. Not released.

### Added

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

### Changed

- The release test accepts a development version: installers and README name the last release until the new one
  is cut.

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
