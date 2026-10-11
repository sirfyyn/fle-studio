# MCP: local, remote, sign-in, apps

FLE Studio is an MCP client. Servers are listed in **Settings → Tools → MCP servers** in the usual `mcpServers` JSON.
Their tools reach the model in the chat; a tool that writes something or starts a program asks first, unless you
allowed that server.

Implemented specifications:

| Part | Version |
| --- | --- |
| MCP base protocol (the version Studio asks for; the server may answer with an older one) | 2025-11-25 |
| MCP authorization (OAuth 2.1, RFC 9728, RFC 8414, OpenID Connect Discovery, RFC 7591, RFC 8707, PKCE RFC 7636) | 2025-11-25 |
| Older HTTP+SSE transport (fallback) | 2024-11-05 |
| MCP Apps (extension `io.modelcontextprotocol/ui`, SEP-1865) | stable, 2026-01-26 |

## Local servers (stdio)

```json
{ "mcpServers": { "files": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/folder"] } } }
```

Studio starts the program when you save and ends it when Studio quits.

## Remote servers (web)

```json
{ "mcpServers": {
  "plain":  { "url": "https://example.com/mcp" },
  "token":  { "url": "https://example.com/other/mcp", "headers": { "Authorization": "Bearer …" } },
  "client": { "url": "https://example.com/third/mcp", "oauth": { "client_id": "…", "scope": "read" } },
  "old":    { "type": "sse", "url": "https://example.com/sse" }
} }
```

* **Streamable HTTP** is tried first (a POST to the URL, answered with JSON or an event stream).
* **Older HTTP+SSE.** When that POST fails with a 4xx other than 401/403, Studio opens the URL with GET and
  `Accept: text/event-stream`, reads the `endpoint` event for the address that takes the POSTs, and gets the answers as
  `message` events, as the 2024-11-05 revision describes. `"type": "sse"` skips the first try. The message address must
  be on the same host as the URL; Settings shows "(older HTTP+SSE)" next to such a server.

## Signing in (OAuth)

A server that answers `401` shows **Sign-in needed** and a **Sign in** button in Settings → Tools. Studio then:

1. reads `WWW-Authenticate: Bearer resource_metadata=…, scope=…` from the 401;
2. fetches the Protected Resource Metadata (RFC 9728): the address from the header, else
   `/.well-known/oauth-protected-resource/<path>`, else the root one;
3. fetches the authorization server metadata in the specification's order: `oauth-authorization-server` (RFC 8414),
   then `openid-configuration`, with the path inserted and appended. It refuses a server whose
   `code_challenge_methods_supported` does not list `S256`, an issuer that does not match, and any address that is not
   `https` (plain `http` only for `127.0.0.1`/`localhost`);
4. uses the client id you put in `"oauth": {"client_id": …}`, else registers itself (Dynamic Client Registration,
   RFC 7591, as a public client with no secret). A server with neither gets a message and a "Client ID" field in
   Settings; register Studio's redirect address `http://127.0.0.1:<Studio's port>/oauth/callback` with that server;
5. opens the authorization page in your browser: authorization code with PKCE S256, a random `state` that is checked,
   and `resource=<the server's canonical address>` (RFC 8707) in both the authorization and the token request. The scope
   is the one from the 401, else all `scopes_supported`, else none;
6. receives the browser at its own callback, exchanges the code and keeps the token.

Tokens are refreshed a minute before they end and after a 401 (the refresh token is replaced when the server rotates
it). When the server refuses the refresh, the token is removed and the server shows **Sign-in needed** again.
**Sign out** asks the server to revoke the token when it has a revocation endpoint, and deletes it either way.

Where tokens live: one file per server in `oauth/` inside Studio's data folder (on Windows
`%LOCALAPPDATA%\FLE Studio`, which is in your profile), mode `0600`, folder `0700`. The file name is a hash of the
server's name and address, so a changed address never receives the old token. A token is never in `settings.json`, the
settings the page reads, a chat, a log or an export; the page learns only "signed in" and when it ends. Bearer tokens
go in the `Authorization` header, never in the address.

The callback is on Studio's own loopback port, so signing in works when the browser runs on the computer where Studio
runs (for a Studio on another machine, forward its port, for example `ssh -L 8642:127.0.0.1:8642`).

## MCP Apps

A tool whose description carries `_meta.ui.resourceUri` (also read: the older flat `_meta["ui/resourceUri"]` and the
Apps SDK's `_meta["openai/outputTemplate"]`, when they name a `ui://` address) has its result shown by an app. Studio
announces the extension when it connects, reads the resource with `resources/read` (it must be
`text/html;profile=mcp-app`, at most 4 MB) and shows the app under the tool card in the chat.

Tools with `_meta.ui.visibility` that does not contain `"model"` are not offered to the model; an app may only call
tools whose visibility contains `"app"` (the default is both).

**Isolation.** The app is untrusted HTML from the server:

* `<iframe sandbox="allow-scripts">`, without `allow-same-origin`: the document's origin is opaque (`window.origin` is
  `null`), so it has no access to Studio's page, cookies, storage or API;
* Studio serves it from `/app-frame/<random id>` with its own `Content-Security-Policy` built from the resource's
  `_meta.ui.csp`. Default: `default-src 'none'`; inline scripts and styles; images, fonts and media only from `data:`;
  `connect-src 'none'`; no frames, forms or `object`. `connectDomains`, `resourceDomains`, `frameDomains` and
  `baseUriDomains` are added when they are plain `https`/`wss` origins (`http`/`ws` only for loopback); anything else is
  dropped. `'self'` is never allowed. No camera, microphone, location or clipboard permission is ever granted;
* messages are accepted only from that frame's window.

**The bridge** (JSON-RPC over `postMessage`):

| App asks | Studio does |
| --- | --- |
| `ui/initialize` | Answers with host info, capabilities and the host context (theme `light`/`dark`, colour variables, locale, time zone, inline display). Nothing is sent before `ui/notifications/initialized`. |
| after `initialized` | Sends `ui/notifications/tool-input` and `ui/notifications/tool-result` (the whole result with `structuredContent`). |
| `ui/notifications/size-changed` | Sets the frame height (60 to 800 px). |
| `tools/call` | Only to the app's own server and only tools meant for apps. Asks like a model's call unless the server is trusted; the server side refuses an unapproved call too. |
| `ui/message` | Asks you; after "Send" the text becomes your next message in the chat. |
| `ui/open-link` | Asks you; only `http`/`https`; opens in a new tab with `noopener`. |
| `ui/request-display-mode` | Always answers `inline`. |
| `ui/update-model-context` | Kept with the call; not sent to the model. |
| `notifications/message`, `ping` | Logged to the browser console / answered. |

Studio sends `ui/notifications/host-context-changed` when the theme changes and `ui/resource-teardown` when you hide
the app. Apps start by themselves only for calls made in the current page; after a reload the card has a **Show the
app** button (that loads the server's page again, it does not run the tool again). When the chat is rebuilt, a running
app stays alive in browsers that have `Node.moveBefore` (Chrome 133+, Firefox 144+); elsewhere it loads again.

## Deliberately missing

* Client ID Metadata Documents (they need a public HTTPS page of Studio's own); an `https://` client id you
  configure works like any other pre-registered id. No `private_key_jwt` or client secrets from registration beyond
  what a server returns.
* Step-up authorization after a `403 insufficient_scope` (Studio shows the error; sign in again).
* Sign-in from a browser on another machine without a port forward.
* Apps: fullscreen and picture-in-picture, `resources/read` and `resources/list` from an app, streaming of partial tool
  input (`tool-input-partial`), device permissions, `ui/update-model-context` reaching the model, and the Apps SDK's
  `window.openai` runtime (only `openai/outputTemplate` is mapped to the resource address).
* A separate sandbox-proxy origin: Studio is a local desktop host, and the opaque origin of the sandboxed frame plus the
  CSP header do the isolating.
* MCP prompts, sampling, elicitation and server-initiated requests (answered with "not offered").
