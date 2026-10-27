# FLE Studio next to a systemd service

For a Linux machine where `llama-server` (the FLE engine or plain llama.cpp) already runs as a systemd unit, for
example `rhea.service`. FLE Studio then works with the service instead of starting a second server. This page is the
checklist for setting that up on a server and opening Studio from another computer.

Studio never edits the unit. Systemd starts the server with the unit's own command; Studio reads it, follows the
journal and offers Start, Stop and Restart. Disconnecting leaves the service as it is.

## 1. Install Studio on the server

```sh
curl -LsSf https://raw.githubusercontent.com/sirfyyn/fle-studio/main/install.sh | sh -s -- --no-start
```

Without internet access on the server, copy the project folder (or a wheel from `python -m build`) to it and use
`FLE_STUDIO_SOURCE=/path/to/fle-studio sh install.sh --no-start`.

## 2. Make the live panel useful

Two options in the unit fill the live panel (a drop-in is enough, the unit file stays untouched):

```ini
# /etc/systemd/system/rhea.service.d/85-studio-view.conf
[Service]
Environment=LLAMA_ARG_ENDPOINT_METRICS=1
Environment=LLAMA_ARG_LOG_VERBOSITY=4
```

`LLAMA_ARG_ENDPOINT_METRICS=1` (`--metrics`) gives the speeds second by second. `LLAMA_ARG_LOG_VERBOSITY=4`
(`-lv 4`) lets the FLE engine's expert-cache hit rate into the journal. Both take effect at the next start of the
service. Without them Studio still works: it shows the speed of each finished answer, and says that the log hides
the hit rate.

## 3. Let the Studio account read the journal and control the unit

The account that runs Studio needs:

- **Reading the journal of the unit**: membership of the `systemd-journal` group (`sudo usermod -aG systemd-journal USER`,
  then log in again), or run Studio as the unit's own user.
- **Start, Stop, Restart (optional)**: a sudoers line, or a polkit rule. Studio first asks systemd, then tries
  `sudo -n`, and never asks for a password. When neither works, the Engine page says so and shows the line to add:

  ```
  USER ALL=(root) NOPASSWD: /usr/bin/systemctl --no-block start rhea.service, /usr/bin/systemctl --no-block stop rhea.service, /usr/bin/systemctl --no-block restart rhea.service
  ```

  Put it in a file made with `sudo visudo -f /etc/sudoers.d/fle-studio`.

Keep Stop and Restart away from anyone who must not interrupt a running measurement or answer: they act on the real
service.

## 4. Start Studio against the service

```sh
fle-studio --service rhea.service --no-browser
```

The first connect is remembered, so later starts reconnect by themselves. Check the page on the server first:

```sh
curl -s http://127.0.0.1:8642/api/ping
```

### Open it from another computer

Studio listens on `127.0.0.1` only. The safest way to reach it is an SSH tunnel, which needs no access key and opens
nothing to the network:

```sh
ssh -L 8642:127.0.0.1:8642 USER@SERVER
```

Then open <http://127.0.0.1:8642/> on your own computer. Use another local port if 8642 is taken there
(`-L 18642:127.0.0.1:8642`).

`--host 0.0.0.0` opens Studio to the whole network instead. It then creates an access key, prints it, and shows it
under Settings, Network. Use this only on a network you trust.

## 5. Keep it running (optional)

A user unit starts Studio at login or, with lingering, at boot:

```ini
# ~/.config/systemd/user/fle-studio.service
[Unit]
Description=FLE Studio
After=network.target

[Service]
ExecStart=%h/.local/bin/fle-studio --service rhea.service --no-browser
Restart=on-failure

[Install]
WantedBy=default.target
```

```sh
systemctl --user daemon-reload
systemctl --user enable --now fle-studio.service
sudo loginctl enable-linger USER      # only if it should run without a login
```

To update Studio on such a machine, run `fle-studio update`. It finds `fle-studio.service` (any user unit whose
`ExecStart` is `fle-studio`), restarts it and checks the version through `/api/ping`; see [UPDATE.md](UPDATE.md).

Studio is light (a few polls per second while a page is open), but on a machine that is being measured, pin it away
from the engine's cores, for example with `CPUAffinity=` in the unit above.

## If it does not work

| What you see | What it means |
| --- | --- |
| "Could not connect to rhea.service" | The unit name is wrong or systemd is not reachable. `systemctl status rhea.service` should work for this account. |
| The log shows a `journalctl` message instead of the server's output | The account may not read the journal (step 3). |
| Start, Stop or Restart answer "FLE Studio may not start or stop ..." | Add the sudoers line from step 3. |
| "failed too often in a row" | systemd's start limit hit; `systemctl reset-failed rhea.service` lets it start again. |
| Speeds appear only after each answer | `--metrics` is missing in the unit (step 2). |
| No expert-cache hit rate | The unit's log level hides it (step 2), or the engine is plain llama.cpp, which has no expert cache. |
