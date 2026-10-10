# FLE Studio

Run large Mixture-of-Experts models on your own computer. FLE Studio is one local app: a chat in your browser,
live numbers while the model answers, a model loader and launch profiles. It works with the
[FLE engine](https://github.com/sirfyyn/FastLocalExperts) and with plain
[llama.cpp](https://github.com/ggml-org/llama.cpp) on macOS, Linux and Windows.

<!-- screenshot: chat -->

- No account, no telemetry, nothing runs in the cloud. Chats, settings and models stay on your machine.
- Python standard library only, no runtime dependencies. The installer brings its own Python.
- MIT licensed.

Status: early release (0.x). It is new, so expect rough edges, and please open an issue when you find one.

## Install

macOS and Linux:

```sh
curl -LsSf https://raw.githubusercontent.com/sirfyyn/fle-studio/main/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/sirfyyn/fle-studio/main/install.ps1 | iex
```

To try it without installing it as a tool (needs [uv](https://docs.astral.sh/uv/) and git):

```sh
uvx --from git+https://github.com/sirfyyn/fle-studio fle-studio
```

What the installers do:

1. Install [uv](https://docs.astral.sh/uv/) if it is missing.
2. Install FLE Studio as a uv tool. uv brings its own Python 3.12, so your system Python is not touched.
3. On Windows, add a Start Menu shortcut named "FLE Studio".
4. Start FLE Studio, which opens in your browser.

They never ask for administrator rights or `sudo` and write only below your home directory. They need `git`,
because uv fetches FLE Studio from GitHub with it (macOS: `xcode-select --install`, Windows:
`winget install --id Git.Git -e`).

Options for the installers:

| What | macOS and Linux | Windows |
| --- | --- | --- |
| Install without starting | `curl -LsSf <url> \| sh -s -- --no-start` | `$env:FLE_STUDIO_NO_START = '1'` before `irm ... \| iex`, or run the script with `-NoStart` |
| Install another release, branch or commit | `curl -LsSf <url> \| FLE_STUDIO_REF=main sh` | `$env:FLE_STUDIO_REF = 'main'` before `irm ... \| iex`, or run the script with `-Ref main` |

On Windows, `& ([scriptblock]::Create((irm <url>))) -NoStart` runs the script with parameters without saving it.

### Start, update, uninstall

```sh
fle-studio                          # start; opens http://127.0.0.1:8642 in your browser
uv tool upgrade fle-studio          # update (stays on the release you installed, see below)
uv tool uninstall fle-studio        # remove the program (your data stays)
```

The installer pins the release it installs (v0.1.0 at the time of writing), and `uv tool upgrade` follows that pin. To
move to a newer release, run the installer again; it always installs the release named in its own script, or the one
you set with `FLE_STUDIO_REF`.

Options of `fle-studio`:

| Option | Meaning |
| --- | --- |
| `--port N` | Port of the web page (default 8642). |
| `--host ADDRESS` | Address to listen on (default 127.0.0.1). `0.0.0.0` opens FLE Studio to your network; it then asks for an access key, which it creates and prints. |
| `--no-browser` | Do not open the browser. |
| `--attach URL` | Use a `llama-server` that is already running instead of starting one. |
| `--service UNIT` | Linux: connect to a `llama-server` that systemd runs as a service, for example `rhea.service` (see below). |
| `--home FOLDER` | Keep all data in this folder (portable installs, tests). |

Your data (settings, chats, engines, models) lives in `~/Library/Application Support/FLE Studio` on macOS,
`~/.local/share/fle-studio` on Linux and `%LOCALAPPDATA%\FLE Studio` on Windows. Uninstalling does not delete it.

### From source

```sh
git clone https://github.com/sirfyyn/fle-studio
cd fle-studio
python -m pip install -e .
fle-studio
```

## What you get

**Chat.** An Open WebUI-style chat in the browser: Markdown and code blocks, a collapsible block for the model's
reasoning, streaming answers, the speed of each answer, and a history that stays on your computer.

<!-- screenshot: chat-reasoning -->

**Live inference data.** While a model works: generation and prompt speed, the share of drafted tokens that were
accepted (when the engine uses a draft or MTP head), NVIDIA GPU use, and the expert-cache hit rate of the FLE engine.

<!-- screenshot: live-data -->

**Model loader.** The Models page lists the FLE quants of Qwen3.8-Flash-Next on Hugging Face
([FLE-GGUF](https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF),
[FLE-REAP448-GGUF](https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-REAP448-GGUF)) and tells you whether each
one fits your machine before you download anything. It also loads any GGUF repo from Hugging Face (search, or paste
`owner/name` or a link) and adds files you already have. Downloads can be paused and resumed, and each file is
checked against the SHA-256 that the Hub lists for it.

<!-- screenshot: models -->

**Launch profiles.** FLE Studio picks a sensible start command for your machine and shows it in full. You can save
your own profiles as a complete command with its environment, change the context size, GPU, threads and cache
budget, and see the exact command line before anything starts.

<!-- screenshot: engine -->

**Experiments.** A short, repeatable speed test (a short answer, a piece of code, a long prompt) that records prefill
and decode speed with the profile and engine used, so you can compare settings side by side.

<!-- screenshot: experiments -->

**Simple settings.** Theme, sampling (the model's recommended values or your own), the folder for models, the
Hugging Face token for gated repos, and network access.

## Engines

FLE Studio installs the engine that suits your machine. You can always choose another one, use your own
`llama-server`, or attach to a server that is already running.

| Your machine | Engine |
| --- | --- |
| Linux or Windows (x64) with an NVIDIA GPU and a CPU with AVX2 | FLE engine, CUDA build |
| Other NVIDIA setups on x64 | llama.cpp, CUDA build |
| macOS on Apple Silicon | llama.cpp, Metal build |
| Intel Mac | llama.cpp, CPU build |
| AMD or Intel GPU on Linux or Windows | llama.cpp, Vulkan build |
| Anything else | llama.cpp, CPU build (it works, but is slow for large models) |
| Your own build | Engine page: set the path to your `llama-server` |
| A server that is already running | Engine page, or `fle-studio --attach http://127.0.0.1:8080` |
| A server that systemd runs as a service (Linux) | Engine page (Service field), or `fle-studio --service rhea.service` |

The FLE engine is a llama.cpp fork that keeps a model's experts in RAM and caches the busy ones in GPU memory, so
that a model larger than your GPU memory can still run on a single GPU. It also supports a separate MTP head for
drafting. FLE builds come from the
[FastLocalExperts releases](https://github.com/sirfyyn/FastLocalExperts/releases), llama.cpp builds from the
[llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) (FLE Studio is tested with tag b11540). Each
download is checked against the SHA-256 digest that GitHub publishes for the file.

### A server that runs as a systemd service

On a machine where `llama-server` runs as a systemd unit, FLE Studio can work with the service instead of starting a
second server next to it. It reads the unit's own command (model, port, alias), follows the unit's journal, shows how
far the server has read the model while it loads, and offers Start, Stop and Restart for the unit. It never changes
the unit: systemd starts the server with the unit's command, as before. Disconnect leaves the service as it is. The
command shown on the Engine page hides `--api-key` and the Hugging Face token.

Starting or stopping a system service needs privileges. FLE Studio first asks systemd (enough as root, for a user
unit, or with a polkit rule), then tries `sudo -n`. It never asks for a password; when neither works, the Engine page
says so and shows the line to add, for example with `sudo visudo -f /etc/sudoers.d/fle-studio`:

```
alice ALL=(root) NOPASSWD: /usr/bin/systemctl --no-block start rhea.service, /usr/bin/systemctl --no-block stop rhea.service, /usr/bin/systemctl --no-block restart rhea.service
```

Large models need a lot of memory. The FLE quants of Qwen3.8-Flash-Next are roughly 70 to 90 GB on disk. Speed depends
on your hardware: the speed figures in the FLE report and in the model catalog were measured on one machine
(RTX PRO 4500 32 GB, 120 GB DDR5, Ryzen 9 9950X3D), and yours will differ. Method and measurements are in the
[FLE report](https://sirfyyn.github.io/FastLocalExperts/).

## Privacy

- FLE Studio listens on `127.0.0.1` only, unless you ask for more with `--host`. The engine it starts listens on
  `127.0.0.1` too.
- There is no telemetry, no analytics and no account.
- Chats, settings and your Hugging Face token (if you set one) are stored in the data folder on your machine.
- Nothing leaves your machine except the downloads you start: model files and searches on Hugging Face, and engine
  releases from GitHub. The installers additionally download uv (astral.sh) and FLE Studio itself (GitHub).

## Roadmap

**Studio Forge** (planned, not in this release): build your own quant with the FLE recipe, from inside FLE Studio.
That means computing an importance matrix (imatrix) on your data, choosing the quant type per tensor, and, optionally,
pruning experts with REAP. The aim is that anyone can reproduce the FLE files for another MoE model, or tune one to a
specific machine.

Ideas and bug reports are welcome as GitHub issues.

## Development

```sh
python -m pip install -e . pytest ruff
ruff check src tests
pytest -q
```

`tests/fake_llama_server.py` is a stand-in for `llama-server` (standard library only, runs on macOS, Linux and
Windows). It imitates the real server's command line, start-up log, HTTP API, streaming and timings, so the tests and
the browser end-to-end run in `.github/workflows/ci.yml` need neither a GPU nor a model. Try it by hand (any existing
file will do as the "model"):

```sh
touch stub.gguf
python tests/fake_llama_server.py -m stub.gguf --port 18080 --metrics
```

The browser end-to-end test starts Studio with the stand-in engine, sends one chat message, visits every page in light
and dark mode and saves a screenshot of each into `e2e-out/`. It fails on any page error or console error. To run it
yourself (the second command downloads Chromium once):

```sh
python -m pip install -e . playwright
python -m playwright install chromium
python tests/e2e/run_e2e.py --out e2e-out
```

`--channel chrome` (or `msedge`) uses a browser you already have instead of the download.

## Licenses

- FLE Studio: [MIT](LICENSE).
- Fonts DM Sans and JetBrains Mono: SIL Open Font License 1.1, licence texts next to the font files in
  `src/fle_studio/web/fonts/`.
- No third-party JavaScript: the page, including its Markdown renderer, is FLE Studio's own code.
- Engines and models are not part of this repository and are not redistributed by it. llama.cpp and the FLE engine
  are downloaded from their own GitHub releases under their own licences. Models are downloaded from Hugging Face
  under the licence on each model card; for example, the FLE files of Qwen3.8-Flash-Next fall under the Qwen
  Community License. Read the licence before you use a model.
