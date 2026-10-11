# Chat: attachments, tools, versions

This page describes the chat features added in 0.2 and the limits they keep.

## Attachments

Use the paperclip, drop files on the chat, or paste them.

| Kind | How it is read | Notes |
| --- | --- | --- |
| Text and code | UTF-8 (also BOM, UTF-16 and Windows code pages) | Binary files, archives and Office files are refused with a reason. |
| PDF | Text layer, with the optional `pypdf` library | `pip install "fle-studio[pdf]"`. Scans are not read (no text recognition). |
| Pictures | Sent to the model as `image_url` parts | Only when the loaded model reports vision (`/props`). Otherwise the page says so before uploading. |

A text attachment reaches the model as `<attachment name="notes.txt">...</attachment>` in front of your text. The
meter beside the Send button shows the context use: exact up to the last answer, an estimate (about 3.6 characters
per token) for what you are about to send. It warns from 80 % and turns red above 100 %.

The largest upload is **Settings → Chat → Largest upload** (1 to 512 MB, default 25). Larger files get a message
that names the limit; nothing is refused without saying why.

## Tools

Turn tools on or off with the **Tools** button above the message box or in **Settings → Tools**. The model needs a
chat template that supports tools (llama-server with `--jinja`, which FLE Studio's own start commands use). If the
engine says the template has no tool support, the button says so.

Built in, each one on its own switch:

- **Calculator.** Arithmetic and common functions. Nothing but numbers is evaluated.
- **Clock.** The current date and time, optionally in a time zone.
- **Files.** `list_files` and `read_file`, read only, inside the one folder you choose. Paths that leave it, including
  through symbolic links, are refused. Off until you pick a folder.
- **Web search.** Off by default. Asks before each search. Uses DuckDuckGo's plain HTML page, or the address of your
  own SearXNG (`http://host:port/search`, with JSON output enabled).

A tool call shows as a card: name, arguments, result, and how long it took. Open it to read the details.

### MCP servers

Add servers in **Settings → Tools** in the usual form:

```json
{
  "mcpServers": {
    "files": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/some/folder"] },
    "remote": { "url": "https://example.org/mcp", "headers": { "Authorization": "Bearer ..." } }
  }
}
```

`command` starts a program on this computer (stdio). `url` is a Streamable HTTP server. The old HTTP+SSE transport is
not supported. The tools of a connected server show up in the chat, named `server__tool`.

**Approval.** A call to an MCP tool waits for your click on its card (Allow once, Decline, or Always allow this
server). The server side enforces it: a call without approval is answered with HTTP 409, whatever the page does.
"Always allow" is a setting in FLE Studio (Settings → Tools lists those servers and can take it back). A field such as
`autoApprove` inside the JSON is ignored on purpose, so a pasted configuration cannot grant itself trust.

Only add servers you trust. A program listed there runs with your rights, and what a tool returns is read by the model.

### Tools and a forced answer format

llama-server turns `response_format`, `json_schema` and `grammar` into a grammar for the answer. With tools in the same
request the model then does not call them and answers with substitute JSON, and nothing says why. FLE Studio therefore
never sends both: the page drops the tools from a request that forces a format, and the proxy refuses such a request
with HTTP 400 and a message. `tests/test_tools.py`, `tests/test_chat_api.py` and `tests/js/check_request.mjs` keep this.

## Versions, search, export

- **Edit** one of your messages: the new text becomes version 2 and the model answers it. The first version and its
  answers stay; switch with `< 1 / 2 >`.
- **Answer again** keeps the earlier answer as another version.
- **Search** (sidebar) looks through titles and the text of every chat, all versions, attachment names and attachment
  text, and highlights the hit.
- **Export** (the `...` menu in the chat): JSON holds every version and the pictures; Markdown holds the version on
  screen. **Import** reads both, and also a plain `[{"role": ..., "content": ...}]` list.

Chats are stored as a tree of messages (`id`, `parent`, and `leaf` for the version on screen). Older chat files open
as one chain.
