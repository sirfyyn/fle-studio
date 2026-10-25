#!/bin/sh
# FLE Studio installer for macOS and Linux (POSIX sh).
#
#   curl -LsSf https://raw.githubusercontent.com/sirfyyn/fle-studio/main/install.sh | sh
#
# Options go after "sh -s --":
#
#   curl -LsSf https://raw.githubusercontent.com/sirfyyn/fle-studio/main/install.sh | sh -s -- --no-start
#
# What it does:
#   1. Installs uv (https://docs.astral.sh/uv/) from astral.sh if it is not there yet.
#   2. Installs FLE Studio as a uv tool. uv brings its own Python 3.12, so your system Python stays untouched.
#   3. Starts FLE Studio (skip this with --no-start).
#
# It never uses sudo and writes only below your home directory: uv itself, the tool environment, the
# "fle-studio" launcher in ~/.local/bin and whatever the uv installer adds to your shell profile.
#
# Options:
#   --no-start    install only; do not start FLE Studio afterwards
#   -h, --help    show this help
#
# Environment:
#   FLE_STUDIO_REF     git tag, branch or commit to install (default: v0.3.2)
#   FLE_STUDIO_SOURCE  install from this folder or wheel file instead of GitHub (offline installs, CI);
#                      FLE_STUDIO_REF and git are not used then

set -eu

REPO_URL="https://github.com/sirfyyn/fle-studio"
UV_INSTALLER_URL="https://astral.sh/uv/install.sh"
REF="${FLE_STUDIO_REF:-v0.3.2}"
SOURCE="${FLE_STUDIO_SOURCE:-}"
START=1
UV_TMP=""
ORIG_PATH="$PATH"

say() {
  printf '==> %s\n' "$*"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

cleanup() {
  if [ -n "$UV_TMP" ]; then
    rm -f "$UV_TMP"
    UV_TMP=""
  fi
}

usage() {
  cat <<'EOF'
Usage: install.sh [--no-start]

Installs FLE Studio with uv (installing uv first if it is missing) and starts it.
Nothing is installed system-wide and sudo is never used.

Options:
  --no-start    install only; do not start FLE Studio afterwards
  -h, --help    show this help

Environment:
  FLE_STUDIO_REF     git tag, branch or commit to install (default: v0.3.2)
  FLE_STUDIO_SOURCE  install from this folder or wheel file instead of GitHub (needs no git)
EOF
}

have() {
  command -v "$1" >/dev/null 2>&1
}

path_has() {
  case ":$PATH:" in
    *":$1:"*) return 0 ;;
  esac
  return 1
}

orig_path_has() {
  case ":$ORIG_PATH:" in
    *":$1:"*) return 0 ;;
  esac
  return 1
}

add_to_path() {
  # Make a directory visible to the commands below. This affects only this run of the installer.
  if [ -d "$1" ] && ! path_has "$1"; then
    PATH="$1:$PATH"
    export PATH
  fi
}

find_uv() {
  # uv can exist without being on PATH: a fresh uv install only changes the shell profile, and
  # this script runs in a shell that has not read it.
  if have uv; then
    return 0
  fi
  for uv_dir in "${UV_INSTALL_DIR:-}" "${XDG_BIN_HOME:-}" "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    if [ -n "$uv_dir" ] && [ -x "$uv_dir/uv" ]; then
      add_to_path "$uv_dir"
      return 0
    fi
  done
  return 1
}

download() {
  # download <url> <file>
  if have curl; then
    curl -LsSf "$1" -o "$2"
  elif have wget; then
    wget -q -O "$2" "$1"
  else
    return 127
  fi
}

need_git() {
  if git --version >/dev/null 2>&1; then
    return 0
  fi
  printf 'error: git is needed, because uv fetches FLE Studio from GitHub with it.\n' >&2
  printf '       Install git, then run this installer again:\n' >&2
  printf '         macOS:           xcode-select --install\n' >&2
  printf '         Debian/Ubuntu:   sudo apt install git\n' >&2
  printf '         Fedora:          sudo dnf install git\n' >&2
  printf '         Arch:            sudo pacman -S git\n' >&2
  exit 1
}

install_uv() {
  say "uv is not installed. Installing it from astral.sh (no sudo, goes to your home directory)."
  if ! have curl && ! have wget; then
    die "need curl or wget to download uv. Install one of them, or install uv yourself (https://docs.astral.sh/uv/getting-started/installation/) and run this again."
  fi
  # The official installer is "curl -LsSf $UV_INSTALLER_URL | sh". It goes through a file here so that a
  # failed or cut-off download stops with a clear message instead of running half a script.
  UV_TMP="$(mktemp "${TMPDIR:-/tmp}/fle-studio-uv.XXXXXX")" || die "could not create a temporary file in ${TMPDIR:-/tmp}."
  if ! download "$UV_INSTALLER_URL" "$UV_TMP"; then
    die "could not download the uv installer from $UV_INSTALLER_URL. Check your internet connection."
  fi
  # stdin is closed for the child: when this script itself comes from "curl | sh", stdin carries the script.
  if ! sh "$UV_TMP" </dev/null; then
    die "the uv installer failed. Install uv by hand (https://docs.astral.sh/uv/getting-started/installation/) and run this again."
  fi
  cleanup
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-start) START=0 ;;
    -h|--help) usage; exit 0 ;;
    *)
      printf 'install.sh: unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

case "$REF" in
  *[!A-Za-z0-9._/+-]*) die "FLE_STUDIO_REF may only contain letters, digits and . _ / + - (got: $REF)." ;;
esac

if [ -z "${HOME:-}" ]; then
  die "HOME is not set."
fi

OS="$(uname -s 2>/dev/null || echo unknown)"
case "$OS" in
  Darwin|Linux) ;;
  MINGW*|MSYS*|CYGWIN*) die "this installer is for macOS and Linux. On Windows, use install.ps1 (see the README)." ;;
  *) die "unsupported operating system: $OS. FLE Studio runs on macOS, Linux and Windows." ;;
esac

if [ -n "$SOURCE" ]; then
  say "FLE Studio installer (source: $SOURCE)"
  [ -e "$SOURCE" ] || die "FLE_STUDIO_SOURCE does not exist: $SOURCE"
  # uv takes a relative path relative to the folder it runs in; the absolute path is the same everywhere.
  case "$SOURCE" in
    /*) ;;
    *) SOURCE="$(pwd)/$SOURCE" ;;
  esac
else
  say "FLE Studio installer (release: $REF)"
  need_git
fi

if find_uv; then
  say "Found $(uv --version)."
else
  install_uv
  if ! find_uv; then
    die "uv was installed, but this script cannot find it. Open a new terminal and run the installer again."
  fi
  say "Installed $(uv --version)."
fi

if [ -n "$SOURCE" ]; then
  say "Installing FLE Studio from $SOURCE (uv sets up Python 3.12 for it if needed) ..."
  if ! uv tool install --python 3.12 --force "$SOURCE" </dev/null; then
    die "installing FLE Studio from $SOURCE failed (see the messages above)."
  fi
else
  say "Installing FLE Studio from GitHub (uv sets up Python 3.12 for it if needed) ..."
  if ! uv tool install --python 3.12 --force "git+${REPO_URL}@${REF}" </dev/null; then
    die "installing FLE Studio failed (see the messages above). Common causes: no internet access, or the release '$REF' does not exist (try FLE_STUDIO_REF=main)."
  fi
fi

BIN_DIR="$(uv tool dir --bin 2>/dev/null)" || BIN_DIR="$HOME/.local/bin"
LAUNCHER="$BIN_DIR/fle-studio"
if [ ! -x "$LAUNCHER" ]; then
  die "FLE Studio was installed, but its launcher is missing in $BIN_DIR. 'uv tool list' shows what uv installed."
fi
VERSION="$("$LAUNCHER" --version 2>&1)" || die "FLE Studio was installed, but '$LAUNCHER --version' failed: $VERSION"

if [ -n "${FLE_STUDIO_HOME:-}" ]; then
  DATA_DIR="$FLE_STUDIO_HOME"
elif [ "$OS" = "Darwin" ]; then
  DATA_DIR="$HOME/Library/Application Support/FLE Studio"
else
  DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/fle-studio"
fi

printf '\n'
say "$VERSION is installed."
printf '\n'
printf '  Start:      fle-studio\n'
printf '  Update:     uv tool upgrade fle-studio\n'
if [ -n "$SOURCE" ]; then
  printf '              (this install came from %s; run the installer again to update it)\n' "$SOURCE"
else
  printf '              (this install follows "%s"; to move to another release, run the installer again\n' "$REF"
  printf '               with FLE_STUDIO_REF=<tag>)\n'
fi
printf '  Uninstall:  uv tool uninstall fle-studio\n'
printf '\n'
printf 'Settings, chats and downloaded models live in\n  %s\nand stay there when you uninstall. Delete that folder to remove them as well.\n' "$DATA_DIR"
if ! orig_path_has "$BIN_DIR"; then
  printf '\n'
  printf 'Note: %s is not on the PATH of this terminal. New terminals get it from your shell profile;\n' "$BIN_DIR"
  printf 'if "fle-studio" is not found there, run "uv tool update-shell" and open a new terminal.\n'
fi
printf '\n'

if [ "$START" -eq 1 ]; then
  say "Starting FLE Studio. It opens in your browser. Press Ctrl+C here to stop it."
  cleanup
  trap - EXIT
  exec "$LAUNCHER"
fi
