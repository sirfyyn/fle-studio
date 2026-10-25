// Live: everything that goes through the /v1 address of FLE Studio, as a temporary chat that only lives in memory.
// The server joins requests that repeat the history into one conversation and keeps the newest 100 messages. A star
// copies a conversation (or everything up to one message) into the normal chat list, where it stays.

import { del, get, post } from "/js/api.js";
import { on, refreshChats, store, topbar } from "/js/app.js";
import { renderMarkdown } from "/js/md.js";
import { clear, confirmDialog, fmt, h, icon, toast } from "/js/ui.js";

function textOf(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.map((p) => {
    if (typeof p === "string") return p;
    if (p?.type === "image_url") return "[picture]";
    return typeof p?.text === "string" ? p.text : "";
  }).filter(Boolean).join("\n");
}

function when(t) {
  return new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function mount(main) {
  const unsub = [];
  let snap = { threads: [], count: 0, max_messages: 100, version: -1 };
  let alive = true;
  const list = h("div", { class: "live-list" });
  const clearBtn = h("button", { class: "btn", onclick: clearAll }, icon("trash", "sm"), "Clear");
  const note = h("p", null, "Everything that goes through the /v1 address of FLE Studio shows up here, from this page and from other programs. "
    + "It is kept in memory only: the newest 100 messages, gone when Studio closes. Star what you want to keep.");
  main.replaceChildren(topbar("Live", clearBtn), h("div", { class: "page" }, h("div", { class: "page-inner" },
    h("div", { class: "page-head" }, h("div", null, h("h2", null, "Live"), note)), list)));

  async function fetchAll() {
    try {
      snap = await get("/api/livechat");
    } catch (e) {
      toast(e.message, "bad");
      return;
    }
    if (alive) render();
  }

  unsub.push(on("livechat", () => {
    if ((store.livechat?.version ?? 0) !== snap.version) fetchAll();
  }));

  function render() {
    clearBtn.disabled = !snap.threads.length;
    if (!snap.threads.length) {
      clear(list, h("div", { class: "card pad-lg faint" }, "Nothing yet. Ask the model something, here or from another program."));
      return;
    }
    clear(list, h("div", { class: "small faint" }, `${snap.count} of ${snap.max_messages} messages`), ...snap.threads.map(thread));
  }

  function thread(t) {
    const star = h("button", { class: `btn ghost icon sm${t.starred ? " on" : ""}`, title: "Keep this conversation in your chats",
      "aria-label": "Keep this conversation", onclick: () => keep(t.id, null) }, icon("star", "sm"));
    const drop = h("button", { class: "btn ghost icon sm", title: "Remove from Live", "aria-label": "Remove from Live",
      onclick: () => remove(t.id) }, icon("trash", "sm"));
    return h("div", { class: "card live-thread" },
      h("div", { class: "card-head" }, h("div", { class: "grow" }, h("h3", { class: "live-title" }, t.title),
        h("p", null, `${t.model || "model"} · ${when(t.updated)}${t.cut ? " · the start was dropped (100 message limit)" : ""}`)), star, drop),
      h("div", { class: "live-messages" }, ...t.messages.map((m) => message(t, m))));
  }

  function message(t, m) {
    const body = h("div", { class: `live-body${m.role === "assistant" ? " md" : ""}` });
    const text = textOf(m.content);
    if (m.role === "assistant") body.innerHTML = renderMarkdown(text); // escaped first, see md.js
    else body.textContent = text;
    const extra = [];
    if (m.reasoning) extra.push(h("details", { class: "live-think" }, h("summary", null, "Thinking"), h("div", { class: "small" }, m.reasoning)));
    for (const c of m.tool_calls || []) {
      extra.push(h("div", { class: "small mono faint" }, `${c.function?.name || "tool"}(${String(c.function?.arguments || "").slice(0, 200)})`));
    }
    const speed = m.timings?.predicted_per_second;
    const star = h("button", { class: `btn ghost icon sm${m.starred ? " on" : ""}`, title: "Keep the conversation up to here",
      "aria-label": "Keep the conversation up to here", onclick: () => keep(t.id, m.id) }, icon("star", "sm"));
    return h("div", { class: `live-msg ${m.role}` },
      h("div", { class: "live-meta small faint" }, h("b", null, m.role), h("span", null, when(m.t)),
        speed ? h("span", null, `${fmt.rate(speed)} tok/s`) : null, h("span", { class: "grow" }), star),
      body, ...extra);
  }

  async function keep(id, message) {
    try {
      await post(`/api/livechat/${id}/star`, message ? { message } : {});
      toast("Kept in your chats", "good");
      refreshChats().catch(() => {});
    } catch (e) {
      toast(e.message, "bad");
    }
    fetchAll();
  }

  async function remove(id) {
    try {
      await del(`/api/livechat/${id}`);
    } catch (e) {
      toast(e.message, "bad");
    }
    fetchAll();
  }

  async function clearAll() {
    const ok = await confirmDialog({ title: "Clear Live?", body: "The conversations shown here are removed from memory. Starred ones stay in your chats.", ok: "Clear", danger: true });
    if (!ok) return;
    try {
      await del("/api/livechat");
    } catch (e) {
      toast(e.message, "bad");
    }
    fetchAll();
  }

  fetchAll();
  return { unmount() { alive = false; unsub.forEach((u) => u()); } };
}
