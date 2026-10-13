# Host and client: one FLE Studio computes for another

A Studio on your Mac can work on a machine with a GPU, for example a server in the next room. The browser only ever
talks to the Studio on the Mac; that Studio passes everything on to the host. Chat, live data, models, engine,
experiments and settings all run against the host, and so does every page that is added later: nothing in the
pass-through lists endpoints.

```
browser ── http://127.0.0.1:8642 ──> Studio (client) ── TLS or ssh ──> Studio (host) ──> engine
```

## Set up the host

Start the Studio on the machine that has the GPU so that it listens beyond itself:

```sh
fle-studio --host 0.0.0.0 --no-browser          # the LAN; or the address of your tailnet interface
fle-studio --host 0.0.0.0 --service rhea.service --no-browser   # next to a systemd service, same thing
```

- It serves **TLS with a certificate of its own making** (no tools needed; ECDSA P-256, in `<data>/tls/`). It prints
  the SHA-256 **fingerprint**, and Settings → Remote access shows it too. `--no-tls` serves plain http, which is for
  browsers on a trusted network only; paired clients should use TLS or an SSH tunnel.
- The **access key** of the browser route (Settings → Access) works as before.
- A host that listens on `127.0.0.1` only can still be reached by an SSH host (below).

## Pair a client

1. On the host: Settings → Remote access → **New pairing code**, or `fle-studio --pair-code` in a terminal (it works
   while the server runs). The code has 8 characters, is valid for 10 minutes and works once.
2. On the client: the switcher at the top of the sidebar → **Add host…**, the host's address (`gpu-box` or
   `gpu-box:8642`) and the code.

The client does not send the code. It sends a proof that only someone who knows the code and saw the host's real
certificate can make, so a machine in between learns neither the code nor gets a token. The certificate's fingerprint
is pinned (trust on first use). You can also type the fingerprint from the host into the dialog; then it must match.
In exchange the host hands out a **client token** of its own. The host keeps only a hash of it and lists every
paired computer, with **Revoke**; the client is out at once. Wrong codes are rate limited (5 per address and 10
minutes) and a code that is guessed at is thrown away after 10 wrong tries.

If the host's certificate ever differs from the pinned one, the client **refuses to send anything** and says so, with
both fingerprints. If you know why (a new installation of the host), pair again; otherwise leave it.

## Rights

| | may |
| --- | --- |
| every paired computer | chat, look at everything, measure (experiments) |
| **Admin** (switch per computer on the host) | also change settings, install or remove engines, load, add or delete models, start and stop the engine or the service, save launch profiles |

Installing, updating and removing (engines, models, Studio updates) additionally ask for a **confirmation** on the
client. A paired computer can never manage clients or hosts, and cannot hand the host on to a third one. Tokens are
never written to logs, and the client keeps its tokens in `hosts.json` of its data folder, readable by its user only.

## SSH instead of a certificate

Add host → **SSH**, with a target like `max@gpu-box`, the SSH port, and the port the Studio on the host listens on.
FLE Studio then starts the system `ssh -N -L` itself, with `BatchMode=yes` (keys or the ssh-agent, **no
passwords**), watches it, and starts it again when it dies; **Reconnect** does that at once. The host can stay on
`127.0.0.1`. Pairing works the same way, through the tunnel. Note that anybody who may log in to the host account over
ssh can also reach the host Studio from there; the client token only governs what a paired Studio may do.

## Chats, uploads, downloads

- Chats are stored on the host. Per host you can switch on **a read-only copy** on the client; when the host is away,
  the chats list and the chats that were seen are shown from the copy.
- Attachments go from the client to the host, exports come back from the host: the pass-through streams in both
  directions.

## What you see

The switcher shows which machine you are on. On a host the window carries the host's colour (a bar on top and a stripe
on the sidebar) and the host's name in the tab title. If the host cannot be reached, a notice says so and offers
**Reconnect**, or **Pair again…** when access was revoked or the certificate changed.

## Security in short

TLS with pinned fingerprint or ssh; single-use codes bound to the certificate; per-client revocable tokens stored as
hashes; rate limit on pairing; rights per client with confirmation for installs; the local page token still guards
the client's own page.
