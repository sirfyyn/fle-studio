# Update

```sh
fle-studio update                     # the latest release
fle-studio update --version v0.4.1    # a named release
fle-studio update --rollback          # back to the version before the last update (`--zurueck` works too)
fle-studio update --dry-run           # print the install command and do nothing
```

The button in Settings → Updates installs the same way. One command, nothing to do by hand afterwards.

## What it does

1. **Reads uv's receipt** (`uv-receipt.toml` in the tool folder, found with `uv tool dir`). It lists FLE Studio and
   every package that was installed next to it with `--with`, for example the Ginny plugin `fle-studio-ginny`.
   `uv tool install --force` builds the environment again from the command line alone and forgets those packages, so the
   command gives them back: local folders (`--with /path`), editable folders (`--with-editable`), wheel files, git
   addresses with their revision, index packages with their version range, and the extras of Studio itself
   (`fle-studio[pdf]`). Example for a Mac with the Ginny plugin:

   ```
   uv tool install --python 3.12 --force --with /Users/me/fle-studio-ginny git+https://github.com/sirfyyn/fle-studio@v0.4.1
   ```

   A package index given at the first install (`--index-url`, `--extra-index-url`) is kept in the receipt too and is
   passed again, so a plugin from a private index is found again.

2. **Checks before it changes anything:** `git` is there; every plugin folder still exists (otherwise it stops and
   offers `--drop NAME`); the uv folder belongs to you (see "Errors" below).
3. **Remembers what was installed** (`update-previous.json` in the data folder), then installs.
4. **Checks the result:** the plugins are still in the receipt, and the program reports the new version.
5. **Refreshes the Dock starter on macOS** (see below).
6. **Restarts the service** that runs Studio and asks it which version it has:
   - macOS: a launchd agent in `~/Library/LaunchAgents` whose first program argument is `fle-studio`
     (`launchctl kickstart -k gui/UID/LABEL`).
   - Linux: a systemd user unit in `~/.config/systemd/user` whose `ExecStart` is `fle-studio`
     (`systemctl --user restart UNIT`).
   - A service that is not loaded or not running is left alone: you stopped it.
   - No service: nothing is restarted. If a Studio that was started by hand still runs the old version, the command
     says so and ends with exit code 3.
   - The port to ask is the one in the service's arguments (`--port N`) or in the settings (8642 by default). The
     command waits up to 60 seconds for the new version.

   It prints both versions, so a service that did not take the update cannot hide:

   ```
   FLE Studio 0.4.0 is installed; the target is 0.4.1.
   Kept next to FLE Studio: fle-studio-ginny
   Installing …
   Restarted launchd de.sirfyyn.fle-studio.
   Running: 0.4.1 on port 8642.
   FLE Studio 0.4.1 is installed.
   To undo this: fle-studio update --rollback
   ```

If the version you ask for is already installed, nothing is installed, but the service is still checked and
restarted when it runs something else. That is the case where an earlier update left the service behind.

A restart stops a model that Studio itself started. A model that a service (`--service`) or another server
(`--attach`) runs is not touched.

## Rollback

Every update stores the packages and the version it started from. `fle-studio update --rollback` installs that
state again, with the same steps and checks as an update. The command stores the state it replaces, so
`--rollback` a second time goes forward again.

## Errors

Every stop prints what is wrong and what to do about it. The ones that happened:

| Message | Meaning and fix |
| --- | --- |
| "uv's folder for FLE Studio belongs to another user" | An earlier update ran with `sudo` (or through a root shell) and left root-owned files. The message contains the exact command, `sudo chown -R YOU /path/to/uv/tools/fle-studio`. Run it once, then the update again. |
| "These packages came from a folder that is not there any more" | A plugin folder was moved or deleted. Put it back, or `fle-studio update --drop NAME`. |
| "No connection to GitHub" | Nothing was changed. Run the command again when the network is back. |
| "There is no release vX.Y.Z" | Wrong name. See <https://github.com/sirfyyn/fle-studio/releases>, or leave out `--version`. |
| "these packages are missing from the new environment" | The install ran but a plugin did not come back. `fle-studio update --rollback`. |
| "nothing answers on port N" / "answers with X, expected Y" | The service was restarted but is not up, or something else holds the port. The message names the log to read. |
| "Failed to connect to bus" (Linux) | No user session for systemd. Log in on the machine, or `export XDG_RUNTIME_DIR=/run/user/$(id -u)`. |

Never run `fle-studio update` or `uv tool install` with `sudo`, nor through a shell that runs as root.

Everything is also written to `update.log` in the data folder.

## The Dock starter (macOS)

"FLE Studio Starter.app" in `~/Applications` is a shim around `fle-studio open`:

```
fle-studio open      # check that the Studio is the installed version (restart the service if not), open the window
```

`open` starts Chrome in app mode with a profile of its own (`<data folder>/chrome`), so no cache of another page
counts, or the default browser when there is no Chrome. `--dry-run` shows what it would do.

The logic lives in the installed package, so an update changes it without touching the app. The app itself is
rewritten only when the shim changes: `fle-studio update` writes it into an existing app and then signs it again
(ad hoc, which is what a locally made app needs):

```sh
codesign --force --sign - "$HOME/Applications/FLE Studio Starter.app"
codesign --verify --deep --strict "$HOME/Applications/FLE Studio Starter.app"
```

If the signing step fails, the update prints this command. An app that is already current is not touched and not
signed again. `fle-studio update` does not create a starter; it refreshes the one that exists.

## Windows

The running program file is locked on Windows, so the command line cannot replace it. The button in Settings →
Updates starts a PowerShell helper that waits for Studio to end, installs the release (with the packages next to
Studio, from the receipt) and starts Studio again.

## Example: Ginny on a Mac and on a server

`fle-studio-ginny` is installed next to Studio as a folder:

```sh
uv tool install --python 3.12 --with ~/fle-studio-ginny git+https://github.com/sirfyyn/fle-studio@v0.4.0   # once
fle-studio update                                                                                       # from now on
```

After the update, `uv tool list --show-with` still shows `fle-studio-ginny`, Settings → Plugins still lists the page,
and the launchd or systemd service runs the new version.
