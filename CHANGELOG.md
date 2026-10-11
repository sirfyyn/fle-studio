# Changelog

## Unreleased

### Added

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
