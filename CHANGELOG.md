# Changelog

## [Unreleased]

Start, tuning and measuring (branch `feature/tuning-0.2`).

### Added

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
