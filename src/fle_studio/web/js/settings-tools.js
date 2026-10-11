// Settings → Tools: what the model may call in a chat (built-in tools, MCP servers) and what it needs permission for.

import { get, post } from "./api.js";
import { h, icon, toast } from "./ui.js";

let timer = 0;
export function stopToolsPolling() {
  clearTimeout(timer);
  timer = 0;
}

const EXAMPLE = `{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/a/folder"]
    },
    "remote": {
      "url": "https://example.com/mcp",
      "headers": { "Authorization": "Bearer …" }
    }
  }
}`;

// ui: the small helpers of settings.js ({ section, field, toggle, save }).
export function toolsSection(s, ui) {
  const t = s.tools;
  const status = h("div", { class: "stack" });
  const folderNote = h("span", { class: "hint" });

  const row = (checked, onFlip, label, hint) => h("label", { class: "row top" }, ui.toggle(checked, onFlip),
    h("span", null, label, hint ? h("span", { class: "muted small" }, ` ${hint}`) : null));

  async function refresh(poll = false) {
    let info;
    try { info = await get("/api/tools"); } catch { return; }
    draw(info);
    folderNote.textContent = !t.files ? "" : info.files_dir_ok ? "The folder exists." : "No such folder yet, so the file tools stay off.";
    stopToolsPolling();
    if (poll && info.connecting) timer = setTimeout(() => refresh(true), 1200);
  }

  function draw(info) {
    const servers = info.servers || [];
    if (!servers.length) {
      status.replaceChildren(h("span", { class: "small faint" }, t.mcp_json?.trim() ? "No server is connected." : "No MCP server is set up."));
      return;
    }
    status.replaceChildren(...servers.map((sv) => {
      const trusted = (t.mcp_trusted || []).includes(sv.name);
      const pill = sv.status === "ready" ? ["good", `${sv.tools} tool${sv.tools === 1 ? "" : "s"}`] :
        sv.status === "error" ? ["bad", "Not connected"] : ["info live", "Connecting…"];
      return h("div", { class: "mcp-row", "data-server": sv.name },
        h("div", { class: "row between" },
          h("div", null, h("b", null, sv.name), h("span", { class: "muted small" }, sv.transport ? ` ${sv.transport === "stdio" ? "program" : "web address"}${sv.info ? ` · ${sv.info}` : ""}` : "")),
          h("span", { class: `pill ${pill[0]}` }, pill[1])),
        sv.error ? h("div", { class: "small bad-text" }, sv.error) : null,
        sv.status === "ready" ? h("label", { class: "row top" },
          ui.toggle(trusted, async (v) => {
            const list = new Set(t.mcp_trusted || []);
            if (v) list.add(sv.name); else list.delete(sv.name);
            if (await ui.save({ tools: { mcp_trusted: [...list] } }, true)) { t.mcp_trusted = [...list]; refresh(); }
          }),
          h("span", null, "Run its tools without asking ", h("span", { class: "muted small" }, "(otherwise every call waits for your click in the chat)"))) : null);
    }));
  }

  const json = h("textarea", { class: "textarea mono", rows: 9, spellcheck: "false", value: t.mcp_json, placeholder: EXAMPLE,
    "aria-label": "MCP servers (JSON)" });
  const filesDir = h("input", { class: "input mono", value: t.files_dir, placeholder: "/Users/you/Documents/notes" });
  const searchUrl = h("input", { class: "input mono", value: t.web_search_url, placeholder: "https://searx.example.org/search (optional)" });
  const rounds = h("input", { class: "input num", type: "number", min: 1, max: 20, value: t.max_rounds,
    onchange: (e) => ui.save({ tools: { max_rounds: Math.max(1, Math.min(20, Number(e.target.value) || 8)) } }, true) });

  refresh(true);

  return ui.section("wrench", "Tools", "What the model may call while it answers. Tools that write something or start a program always ask you first.",
    row(t.enabled, (v) => ui.save({ tools: { enabled: v } }, true).then(() => ui.render()), "Let the model use tools",
      "(needs a model whose chat template supports them)"),
    h("div", { class: "grid two" },
      ui.field("Steps per answer", rounds, "How often the model may call tools before it has to answer.")),

    h("h4", { class: "sub" }, "Built in"),
    row(t.calculator, (v) => ui.save({ tools: { calculator: v } }, true), "Calculator", "exact arithmetic, nothing else"),
    row(t.datetime, (v) => ui.save({ tools: { datetime: v } }, true), "Clock", "today's date and time"),
    row(t.files, (v) => ui.save({ tools: { files: v } }, true).then(() => ui.render()), "Read files in one folder",
      "read only; the model can list and read text files below the folder you name, nothing outside it"),
    t.files ? ui.field("Folder", h("div", { class: "row" }, filesDir,
      h("button", { class: "btn", onclick: async () => { if (await ui.save({ tools: { files_dir: filesDir.value.trim() } })) refresh(); } }, "Save")),
    "") : null,
    t.files ? folderNote : null,
    row(t.web_search, (v) => ui.save({ tools: { web_search: v } }, true).then(() => ui.render()), "Web search",
      "off by default. The question the model searches for leaves this computer."),
    t.web_search ? row(t.web_search_confirm, (v) => ui.save({ tools: { web_search_confirm: v } }, true), "Ask before every search") : null,
    t.web_search ? ui.field("Search server", h("div", { class: "row" }, searchUrl,
      h("button", { class: "btn", onclick: () => ui.save({ tools: { web_search_url: searchUrl.value.trim() } }) }, "Save")),
    "A SearXNG address that answers with JSON. Empty: DuckDuckGo's plain web page.") : null,

    h("h4", { class: "sub" }, "MCP servers"),
    row(t.mcp_enabled, (v) => ui.save({ tools: { mcp_enabled: v } }, true).then(() => refresh()), "Use MCP servers",
      "programs and web addresses that offer tools in the Model Context Protocol"),
    ui.field("Servers", json, "The usual mcpServers JSON. A program (\"command\") is started on this computer when you save; a \"url\" must be a Streamable HTTP server."),
    h("div", { class: "row" },
      h("button", { class: "btn primary", onclick: async () => {
        if (!(await ui.save({ tools: { mcp_json: json.value } }))) return;
        t.mcp_json = json.value;
        try { draw(await post("/api/tools/reload")); } catch (e) { toast(e.message, "bad"); }
        refresh(true);
      } }, icon("link", "sm"), "Save and connect"),
      h("button", { class: "btn ghost", onclick: async () => { try { draw(await post("/api/tools/reload")); } catch (e) { toast(e.message, "bad"); } refresh(true); } }, "Connect again")),
    status,
    h("span", { class: "small faint" }, "Only add servers you trust: a program you list here runs with your rights, and what its tools return is read by the model."));
}
