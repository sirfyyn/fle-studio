// The chat page: conversation, composer, and the live panel with the engine's numbers.

import { get, post, put, streamChat } from "./api.js";
import { emit, engineModelName, engineUp, navigate, on, refreshChats, store, topbar } from "./app.js";
import { renderMarkdown } from "./md.js";
import { clear, copyText, debounce, fmt, h, icon, mark, meter, openMenu, ring, sparkline, stateBadge, toast } from "./ui.js";

const SUGGESTIONS = [
  ["Explain mixture-of-experts", "and why only a few experts run for each token"],
  ["Write a Python script", "that hashes every file in a folder in parallel"],
  ["Summarize a text for me", "paste an article and get the key points"],
  ["Plan three days in Lisbon", "on a budget, with one day trip"],
];

export function mount(main, chatId) {
  let chat = newChat();
  let streaming = null;
  let pinned = true;
  let liveOpen = false;
  const unsub = [];

  // ------------------------------------------------------------------------------------------- layout

  const title = h("h1", { title: "Rename", style: { cursor: "text" } }, "New chat");
  title.addEventListener("click", rename);
  const modelChip = h("button", { class: "chip" });
  modelChip.addEventListener("click", () => modelMenu(modelChip));
  const liveBtn = h("button", { class: "btn ghost icon", title: "Live data", "aria-label": "Live data",
    onclick: () => { liveOpen = !liveOpen; live.classList.toggle("open", liveOpen); } }, icon("panel"));

  const inner = h("div", { class: "messages-inner" });
  const messages = h("div", { class: "messages" }, inner);
  messages.addEventListener("scroll", () => {
    pinned = messages.scrollHeight - messages.scrollTop - messages.clientHeight < 90;
  });
  messages.addEventListener("click", onMessagesClick);

  const ta = h("textarea", { rows: 1, placeholder: "Message the model", "aria-label": "Message" });
  const sendBtn = h("button", { class: "send", "aria-label": "Send", title: "Send (Enter)" }, icon("send"));
  const thinkChip = h("button", { class: "chip", title: "Let the model think before it answers" });
  const sampChip = h("button", { class: "chip", title: "Sampling settings" });
  const ctxMeter = h("span", { class: "ctx-meter num" });
  const composer = h("div", { class: "composer" }, ta,
    h("div", { class: "bar-row" }, thinkChip, sampChip, h("span", { class: "grow" }), ctxMeter, sendBtn));
  const note = h("div", { class: "composer-note" });
  const live = h("aside", { class: "live-panel", "aria-label": "Live data" });

  main.replaceChildren(
    topbar(title, modelChip, liveBtn),
    h("div", { class: "chat" },
      h("div", { class: "chat-col" }, messages, h("div", { class: "composer-wrap" }, composer, note)),
      live));

  ta.addEventListener("input", autosize);
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); submit(); }
    if (e.key === "Escape" && streaming) stop();
  });
  sendBtn.addEventListener("click", () => (streaming ? stop() : submit()));
  thinkChip.addEventListener("click", toggleThinking);
  sampChip.addEventListener("click", () => samplingMenu(sampChip));

  unsub.push(on("engine", () => { renderHeader(); renderComposer(); if (!chat.messages.length) renderMessages(); }));
  unsub.push(on("state", () => { renderComposer(); if (!chat.messages.length) renderMessages(); }));
  unsub.push(on("live", renderLive));

  if (chatId) load(chatId); else { renderMessages(); ta.focus(); }
  renderHeader();
  renderComposer();
  renderLive();

  // ------------------------------------------------------------------------------------------- data

  function newChat() {
    return { id: null, title: "", created: 0, model: "", system: "", messages: [] };
  }

  async function load(id) {
    try {
      chat = await get(`/api/chats/${id}`);
    } catch {
      toast("That chat no longer exists", "bad");
      navigate("/", { replace: true });
      return;
    }
    renderHeader();
    renderMessages();
    renderComposer();
    scrollBottom(true);
    ta.focus();
  }

  // Saves run one after another: the first one creates the chat, so a second one must wait for its id.
  let saving = Promise.resolve();
  function save() {
    saving = saving.then(saveNow, saveNow);
    return saving;
  }

  async function saveNow() {
    const body = { ...chat, model: chat.model || engineModelName() };
    try {
      const saved = chat.id ? await put(`/api/chats/${chat.id}`, body) : await post("/api/chats", body);
      // Keep our own message objects: an answer may still be streaming into them.
      Object.assign(chat, { id: saved.id, title: saved.title, created: saved.created, updated: saved.updated });
      if (location.pathname !== `/c/${chat.id}`) history.replaceState(null, "", `/c/${chat.id}`);
      renderHeader();
      refreshChats();
    } catch (e) {
      toast(`Could not save the chat: ${e.message}`, "bad");
    }
  }

  // ------------------------------------------------------------------------------------------- header

  function renderHeader() {
    title.textContent = chat.title || "New chat";
    const e = store.engine;
    const name = engineModelName();
    clear(modelChip, e && (e.state === "ready" || e.state === "attached") ? h("span", { class: "pill good", style: { height: "18px", padding: "0 6px" } }, h("span", { class: "dot" })) :
      e?.state === "starting" ? h("span", { class: "pill info live", style: { height: "18px", padding: "0 6px" } }, h("span", { class: "dot" })) : null,
    h("span", { style: { maxWidth: "260px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" } },
      name || "No model loaded"), icon("down", "sm"));
  }

  async function rename() {
    if (!chat.id) return;
    const input = h("input", { class: "input", value: chat.title, style: { height: "30px", maxWidth: "420px" } });
    title.replaceChildren(input);
    input.focus();
    input.select();
    const done = async (keep) => {
      input.removeEventListener("blur", onBlur);
      if (keep && input.value.trim() && input.value.trim() !== chat.title) {
        chat.title = input.value.trim();
        await save();
      }
      renderHeader();
    };
    const onBlur = () => done(true);
    input.addEventListener("blur", onBlur);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") input.blur();
      if (e.key === "Escape") done(false);
    });
  }

  function modelMenu(anchor) {
    const svc = store.engine?.service;
    if (svc) { // the service loads its own model with its own command
      openMenu(anchor, [{ head: svc.unit }, { label: "Start, stop and log: Engine page", icon: "engine", onclick: () => navigate("/engine") }]);
      return;
    }
    const models = (store.state?.models || []).filter((m) => m.kind === "model" && m.complete);
    const cur = store.engine?.model?.id;
    const items = [{ head: "Load a model" }];
    for (const m of models) {
      items.push({ label: m.name, hint: m.id === cur && engineUp() ? "loaded" : fmt.bytes(m.size), icon: m.id === cur ? "check" : "models",
        onclick: () => loadModel(m.id) });
    }
    if (!models.length) items.push({ label: "No models yet: get one", icon: "download", onclick: () => navigate("/models") });
    items.push("sep");
    if (engineUp() && store.engine.state === "ready") items.push({ label: "Eject the model", icon: "power", onclick: ejectModel });
    items.push({ label: "Engine and launch settings", icon: "engine", onclick: () => navigate("/engine") });
    openMenu(anchor, items);
  }

  // ------------------------------------------------------------------------------------------- messages

  function renderMessages() {
    if (!chat.messages.length) {
      clear(inner, welcome());
      return;
    }
    clear(inner, chat.messages.map((m, i) => msgEl(m, i)));
  }

  function welcome() {
    const e = store.engine || {};
    const st = store.state || {};
    const models = (st.models || []).filter((m) => m.kind === "model" && m.complete);
    const box = h("div", { class: "welcome" });
    if (e.state === "ready" || e.state === "attached") {
      box.append(
        h("div", { style: { width: "44px", height: "44px" } }, mark(44)),
        h("h2", null, "What should ", h("span", { class: "grad" }, engineModelName() || "the model"), " work on?"),
        h("p", { class: "lead" }, "Answers stream from the engine on this computer. Speed, MTP acceptance and the expert cache show in the live panel."),
        h("div", { class: "suggest" }, SUGGESTIONS.map(([a, b]) => h("button", { onclick: () => { ta.value = `${a} ${b}`; autosize(); submit(); } },
          h("b", null, a), h("span", null, b)))));
      return box;
    }
    if (e.state === "starting") {
      const p = e.load?.progress || 0;
      box.append(
        h("h2", null, "Loading ", h("span", { class: "grad" }, engineModelName())),
        h("p", { class: "lead" }, e.phase || "Starting the engine"),
        h("div", { class: "card" }, h("div", { class: "stack" },
          h("div", { class: `bar${p ? "" : " indet"}` }, h("i", { style: { width: `${(p * 100).toFixed(1)}%` } })),
          h("div", { class: "row between small muted num" },
            h("span", null, e.load?.total ? `${fmt.bytes(e.load.read)} of ${fmt.bytes(e.load.total)} read` : "Preparing"),
            h("span", null, e.load?.elapsed ? fmt.dur(e.load.elapsed) : "")),
          e.hint ? h("div", { class: "callout info" }, icon("info"), h("div", null, e.hint)) : null)));
      return box;
    }
    if (e.service) {
      const svc = e.service;
      box.append(
        h("div", { style: { width: "44px", height: "44px" } }, mark(44)),
        h("h2", null, h("span", { class: "grad" }, svc.unit), e.state === "stopping" ? " is stopping" : e.state === "crashed" ? " stopped with an error" : " is not running"),
        h("p", { class: "lead" }, "FLE Studio is connected to this service. Starting it runs the service's own command, which loads its model."),
        e.error ? h("div", { class: "callout bad" }, icon("alert"), h("div", null, e.error)) : null,
        h("div", { class: "row wrap", style: { justifyContent: "center" } },
          e.state === "stopping" ? null : h("button", { class: "btn primary", onclick: startService }, icon("play"), "Start the service"),
          h("a", { class: "btn", href: "/engine", "data-link": true }, icon("engine"), "Engine page and log")));
      return box;
    }
    const hasModel = models.length > 0;
    box.append(
      h("div", { style: { width: "44px", height: "44px" } }, mark(44)),
      h("h2", null, "Run ", h("span", { class: "grad" }, "Qwen3.8-Flash-Next"), " on your own machine"),
      h("p", { class: "lead" }, "FLE Studio downloads a model, starts the FLE engine with settings that fit your hardware, and shows what the engine is doing while you chat."));
    if (e.state === "crashed") {
      box.append(h("div", { class: "callout bad" }, icon("alert"), h("div", null,
        h("b", null, "The engine stopped. "), e.error || "", e.hint ? h("div", { class: "small", style: { marginTop: "4px" } }, e.hint) : null,
        h("div", { style: { marginTop: "8px" } }, h("a", { href: "/engine", "data-link": true }, "Open the engine log")))));
    }
    const last = models.find((m) => m.id === st.settings?.last_model) || models[0];
    box.append(h("div", { class: "steps" },
      step(1, hasModel ? "done" : "now", "Get a model", hasModel ? `${models.length} in your library` : "Pick an FLE quant that fits your memory, or any GGUF from Hugging Face.",
        hasModel ? null : h("a", { class: "btn primary", href: "/models", "data-link": true }, icon("download"), "Choose a model")),
      step(2, hasModel ? "now" : "", "Load it", hasModel ? last.name : "FLE Studio picks the launch settings for your GPU and RAM.",
        hasModel ? h("button", { class: "btn primary", onclick: () => loadModel(last.id) }, icon("play"), "Load") : null),
      step(3, "", "Chat", "Ask anything. Nothing leaves this computer.", null)));
    if (!hasModel) {
      box.append(h("p", { class: "small muted" }, "Already running a llama-server somewhere? ",
        h("a", { href: "/engine", "data-link": true }, "Connect to it"), " instead."));
    }
    return box;
  }

  function step(n, cls, titleText, sub, action) {
    return h("div", { class: `step ${cls}` }, h("div", { class: "n" }, cls === "done" ? icon("check", "sm") : n),
      h("div", null, h("b", null, titleText), h("span", null, sub)), action || h("span"));
  }

  function msgEl(m, i) {
    const last = i === chat.messages.length - 1;
    if (m.role === "user") {
      const bubble = h("div", { class: "bubble" }, m.content);
      return h("div", { class: `msg user${last ? " last" : ""}`, "data-i": i }, bubble,
        h("div", { class: "actions" },
          h("button", { class: "btn ghost icon sm", title: "Edit", "aria-label": "Edit", onclick: () => editUser(i) }, icon("edit", "sm")),
          h("button", { class: "btn ghost icon sm", title: "Copy", "aria-label": "Copy", onclick: (e) => copyText(m.content, e.currentTarget) }, icon("copy", "sm"))));
    }
    const isLive = streaming && streaming.msg === m;
    const body = h("div", { class: "body", style: { display: "grid", gap: "8px", minWidth: 0 } });
    body.append(h("div", { class: "who" }, m.model || "Assistant", m.t ? h("span", { class: "faint", style: { fontWeight: 400 } }, fmt.ago(m.t)) : null));
    if (m.reasoning || (isLive && streaming.thinking)) body.append(thinkEl(m, isLive));
    const md = h("div", { class: `md${isLive ? " streaming" : ""}` });
    md.innerHTML = renderMarkdown(m.content) + (isLive && !streaming.thinking ? '<span class="cursor"></span>' : "");
    if (!m.content && !isLive && !m.error) md.innerHTML = '<p class="faint">(no text)</p>';
    body.append(md);
    if (m.error) body.append(h("div", { class: "callout bad" }, icon("alert"), h("div", null, m.error)));
    if (m.stopped) body.append(h("div", { class: "small faint" }, "Stopped."));
    const stats = statsEl(m);
    if (stats) body.append(stats);
    if (!isLive) {
      body.append(h("div", { class: "actions" },
        h("button", { class: "btn ghost icon sm", title: "Copy", "aria-label": "Copy", onclick: (e) => copyText(m.content, e.currentTarget) }, icon("copy", "sm")),
        last ? h("button", { class: "btn ghost icon sm", title: "Regenerate", "aria-label": "Regenerate", onclick: () => regenerate() }, icon("refresh", "sm")) : null,
        h("button", { class: "btn ghost icon sm", title: "Delete", "aria-label": "Delete", onclick: () => removeMsg(i) }, icon("trash", "sm"))));
    }
    return h("div", { class: `msg assistant${last ? " last" : ""}`, "data-i": i },
      h("div", { class: "avatar" }, mark(20)), body);
  }

  function thinkEl(m, isLive) {
    const secs = m.think_s ?? (isLive && streaming.thinkStart ? (performance.now() - streaming.thinkStart) / 1000 : null);
    const label = isLive && streaming.thinking ? "Thinking" : secs ? `Thought for ${fmt.dur(secs)}` : "Thoughts";
    const d = h("details", { class: `think${isLive && streaming.thinking ? " live" : ""}` },
      h("summary", null, icon("chevron", "sm chev"), icon("brain", "sm"), h("span", { class: "lbl" }, label)),
      h("div", { class: "think-body md", html: renderMarkdown(m.reasoning || "") }));
    if (isLive && streaming.thinking) d.open = true;
    else if (streaming?.openThink?.has(m)) d.open = true;
    d.addEventListener("toggle", () => {
      if (!streaming) return;
      if (d.open) streaming.openThink.add(m); else streaming.openThink.delete(m);
    });
    return d;
  }

  function statsEl(m) {
    const s = m.stats;
    if (!s || !s.predicted_n) return null;
    const parts = [
      h("span", { class: "hot" }, h("b", null, fmt.rate(s.decode)), " tok/s"),
      h("span", null, h("b", null, fmt.int(s.predicted_n)), " tokens"),
    ];
    if (s.prefill && s.prompt_n) parts.push(h("span", null, "prefill ", h("b", null, fmt.rate(s.prefill)), " tok/s"));
    if (s.accept !== null && s.accept !== undefined) parts.push(h("span", null, "MTP ", h("b", null, fmt.pct(s.accept))));
    if (s.ttft_ms) parts.push(h("span", null, "first token ", h("b", null, fmt.dur(s.ttft_ms / 1000))));
    if (s.total_s) parts.push(h("span", null, h("b", null, fmt.dur(s.total_s))));
    return h("div", { class: "stats", title: "Measured by the engine for this answer" }, parts);
  }

  function onMessagesClick(e) {
    const b = e.target.closest("[data-copy]");
    if (!b) return;
    const code = b.closest(".codeblock")?.querySelector("code");
    if (code) copyText(code.textContent, b);
  }

  function scrollBottom(force = false) {
    if (force || pinned) messages.scrollTop = messages.scrollHeight;
  }

  // ------------------------------------------------------------------------------------------- sending

  function autosize() {
    ta.style.height = "auto";
    ta.style.height = `${Math.min(260, ta.scrollHeight)}px`;
    renderComposer();
  }

  function renderComposer() {
    const up = engineUp();
    const c = store.state?.settings?.chat || {};
    sendBtn.classList.toggle("stop", !!streaming);
    sendBtn.disabled = !streaming && (!up || !ta.value.trim());
    clear(sendBtn, icon(streaming ? "stop" : "send"));
    sendBtn.title = streaming ? "Stop (Esc)" : "Send (Enter)";
    ta.placeholder = up ? `Message ${engineModelName() || "the model"}` : "Load a model to start chatting";
    thinkChip.classList.toggle("on", c.thinking !== false);
    clear(thinkChip, icon("brain", "sm"), c.thinking !== false ? "Thinking on" : "Thinking off");
    clear(sampChip, icon("sliders", "sm"), c.sampling === "custom" ? `T ${c.temperature} · top-p ${c.top_p}` : "Model defaults");
    const lastStats = [...chat.messages].reverse().find((m) => m.stats?.predicted_n)?.stats;
    const nctx = store.engine?.n_ctx;
    if (lastStats && nctx) {
      const used = (lastStats.cache_n || 0) + (lastStats.prompt_n || 0) + (lastStats.predicted_n || 0);
      ctxMeter.textContent = `${fmt.tokens(used)} / ${fmt.ctx(nctx)} tokens`;
      ctxMeter.title = "Context used by this conversation, of the engine's context window";
    } else {
      ctxMeter.textContent = nctx ? `${fmt.ctx(nctx)} context` : "";
    }
    note.textContent = up ? "Runs on this computer. Answers can be wrong; check what matters." : "";
  }

  async function toggleThinking() {
    const cur = store.state?.settings?.chat?.thinking !== false;
    await saveChatSettings({ thinking: !cur });
  }

  async function saveChatSettings(patch) {
    try {
      const s = await post("/api/settings", { chat: patch });
      store.state.settings = s;
      emit("state");
    } catch (e) {
      toast(e.message, "bad");
    }
  }

  function samplingMenu(anchor) {
    const c = store.state?.settings?.chat || {};
    openMenu(anchor, [
      { head: "Sampling" },
      { label: "Model defaults", hint: c.sampling !== "custom" ? "in use" : "", icon: c.sampling !== "custom" ? "check" : "models",
        onclick: () => saveChatSettings({ sampling: "model" }) },
      { label: `My settings (T ${c.temperature}, top-p ${c.top_p}, top-k ${c.top_k})`, hint: c.sampling === "custom" ? "in use" : "",
        icon: c.sampling === "custom" ? "check" : "sliders", onclick: () => saveChatSettings({ sampling: "custom" }) },
      "sep",
      { label: "Change values and system prompt", icon: "settings", onclick: () => navigate("/settings") },
    ]);
  }

  function requestBody(upTo) {
    const c = store.state?.settings?.chat || {};
    const msgs = chat.messages.slice(0, upTo).filter((m) => m.content || m.role === "user")
      .map((m) => ({ role: m.role, content: m.content }));
    const system = chat.system || c.system_prompt;
    if (system) msgs.unshift({ role: "system", content: system });
    const body = { messages: msgs, stream_options: { include_usage: true }, chat_template_kwargs: { enable_thinking: c.thinking !== false } };
    if (c.sampling === "custom") {
      Object.assign(body, { temperature: c.temperature, top_p: c.top_p, top_k: c.top_k, min_p: c.min_p, presence_penalty: c.presence_penalty });
    }
    if (c.max_tokens > 0) body.max_tokens = c.max_tokens;
    return body;
  }

  async function submit() {
    const text = ta.value.trim();
    if (!text || streaming) return;
    if (!engineUp()) { toast("Load a model first", "bad"); return; }
    ta.value = "";
    autosize();
    chat.messages.push({ role: "user", content: text, t: Date.now() / 1000 });
    const asst = { role: "assistant", content: "", reasoning: "", model: engineModelName(), t: Date.now() / 1000 };
    chat.messages.push(asst);
    pinned = true;
    renderMessages();
    scrollBottom(true);
    save();
    await generate(asst);
  }

  async function regenerate() {
    const m = chat.messages[chat.messages.length - 1];
    if (!m || m.role !== "assistant" || streaming) return;
    if (!engineUp()) { toast("Load a model first", "bad"); return; }
    Object.assign(m, { content: "", reasoning: "", stats: null, error: "", stopped: false, think_s: null,
      model: engineModelName(), t: Date.now() / 1000 });
    renderMessages();
    scrollBottom(true);
    await generate(m);
  }

  function editUser(i) {
    if (streaming) return;
    const m = chat.messages[i];
    const el = inner.querySelector(`.msg[data-i="${i}"]`);
    const box = h("textarea", { class: "textarea", rows: 3, value: m.content });
    const form = h("div", { class: "edit-box" }, box, h("div", { class: "row end" },
      h("button", { class: "btn ghost", onclick: () => renderMessages() }, "Cancel"),
      h("button", { class: "btn primary", onclick: async () => {
        const text = box.value.trim();
        if (!text) return;
        if (!engineUp()) { toast("Load a model first", "bad"); return; }
        m.content = text;
        chat.messages.splice(i + 1);
        const asst = { role: "assistant", content: "", reasoning: "", model: engineModelName(), t: Date.now() / 1000 };
        chat.messages.push(asst);
        renderMessages();
        scrollBottom(true);
        await generate(asst);
      } }, "Send")));
    el.replaceChildren(form);
    box.focus();
  }

  async function removeMsg(i) {
    chat.messages.splice(i, 1);
    renderMessages();
    if (chat.id) await save();
  }

  async function generate(asst) {
    const upTo = chat.messages.indexOf(asst);
    const ctrl = new AbortController();
    streaming = { msg: asst, ctrl, t0: performance.now(), tFirst: 0, thinking: false, thinkStart: 0, raw: "", split: false,
      openThink: new Set(), frame: 0 };
    renderComposer();
    try {
      await streamChat(requestBody(upTo), { signal: ctrl.signal, onChunk });
    } catch (e) {
      if (e.name === "AbortError") asst.stopped = true;
      else asst.error = e.message;
    }
    const s = streaming;
    if (s.thinking) asst.think_s = (performance.now() - s.thinkStart) / 1000;
    const total = (performance.now() - s.t0) / 1000;
    if (asst.stats) {
      asst.stats.total_s = total;
      asst.stats.ttft_ms = s.tFirst ? s.tFirst - s.t0 : null;
    }
    streaming = null;
    cancelAnimationFrame(s.frame);
    renderMessages();
    renderComposer();
    scrollBottom();
    await save();
    ta.focus();
  }

  function onChunk(chunk) {
    const s = streaming;
    if (!s) return;
    const m = s.msg;
    const delta = chunk.choices?.[0]?.delta || {};
    const now = performance.now();
    if (delta.reasoning_content) {
      s.split = true;
      if (!s.tFirst) s.tFirst = now;
      if (!s.thinking && !m.content) { s.thinking = true; s.thinkStart = now; }
      m.reasoning += delta.reasoning_content;
    }
    if (delta.content) {
      if (!s.tFirst) s.tFirst = now;
      s.raw += delta.content;
      if (s.split) {
        m.content = s.raw;
      } else {
        const t = s.raw.match(/^\s*<think>([\s\S]*?)(<\/think>\s*|$)([\s\S]*)$/);
        if (t) {
          m.reasoning = t[1];
          m.content = t[3];
          if (!t[2] && !s.thinking) { s.thinking = true; s.thinkStart = s.thinkStart || now; }
        } else {
          m.content = s.raw;
        }
      }
      if (s.thinking && m.content.trim()) {
        s.thinking = false;
        m.think_s = (now - s.thinkStart) / 1000;
      }
    }
    if (chunk.timings) {
      const t = chunk.timings;
      m.stats = {
        decode: t.predicted_per_second, predicted_n: t.predicted_n, prefill: t.prompt_per_second, prompt_n: t.prompt_n,
        cache_n: t.cache_n, accept: t.draft_n ? (t.draft_n_accepted || 0) / t.draft_n : null, draft_n: t.draft_n || 0,
      };
    }
    if (!s.frame) {
      s.frame = requestAnimationFrame(() => {
        if (!streaming) return;
        streaming.frame = 0;
        paintStreaming();
      });
    }
  }

  function paintStreaming() {
    const s = streaming;
    const i = chat.messages.indexOf(s.msg);
    const old = inner.querySelector(`.msg[data-i="${i}"]`);
    const fresh = msgEl(s.msg, i);
    if (old) {
      const thinkBody = old.querySelector(".think-body");
      const keepScroll = thinkBody ? thinkBody.scrollTop : 0;
      old.replaceWith(fresh);
      const nb = fresh.querySelector(".think-body");
      if (nb) nb.scrollTop = s.thinking ? nb.scrollHeight : keepScroll;
    } else inner.append(fresh);
    scrollBottom();
  }

  function stop() {
    streaming?.ctrl.abort();
  }

  // ------------------------------------------------------------------------------------------- engine actions

  async function loadModel(id) {
    try {
      const st = await post("/api/engine/start", { model: id });
      store.engine = st;
      emit("engine");
      toast("Loading the model. This can take a few minutes for large files.");
    } catch (e) {
      toast(e.message, "bad", 8000);
      if (e.status === 400 || e.status === 409) navigate("/engine");
    }
  }

  async function startService() {
    try {
      store.engine = await post("/api/engine/service/start");
      emit("engine");
      toast("Starting the service. Loading a large model takes a few minutes.");
    } catch (e) {
      toast(e.message, "bad", 15000);
    }
  }

  async function ejectModel() {
    try {
      store.engine = await post("/api/engine/stop");
      emit("engine");
    } catch (e) { toast(e.message, "bad"); }
  }

  // ------------------------------------------------------------------------------------------- live panel

  function renderLive() {
    const hist = store.live.slice(-120);
    const lastS = hist[hist.length - 1] || {};
    const meta = store.liveMeta || {};
    const e = store.engine || {};
    const recent = (key, secs) => {
      const cut = (lastS.t || 0) - secs;
      for (let k = hist.length - 1; k >= 0; k -= 1) {
        if (hist[k].t < cut) break;
        if (hist[k][key] !== undefined && hist[k][key] !== null) return hist[k][key];
      }
      return null;
    };
    const lr = meta.last_response;
    const decodeNow = recent("decode", 3);
    const decode = decodeNow ?? lr?.predicted_per_second ?? null;
    const prefill = recent("prefill", 15) ?? lr?.prompt_per_second ?? null;
    let accept = recent("accept", 20);
    if (accept === null && meta.accept_logged !== null && meta.accept_logged !== undefined) accept = meta.accept_logged;
    if (accept === null && lr?.draft_n) accept = (lr.draft_n_accepted || 0) / lr.draft_n;
    const cacheHit = meta.cache?.hit_rate !== undefined && meta.cache?.hit_rate !== null ? meta.cache.hit_rate / 100 : null;
    const svc = e.service || null;
    const fle = (e.plan?.args || []).includes("--moe-expert-cache-inserts") || cacheHit !== null || !!svc?.expert_cache;

    const tiles = [];
    tiles.push(h("div", { class: "row between" }, h("h4", null, "Live"), stateBadge(e.state)));
    tiles.push(h("div", { class: "tile" },
      h("div", { class: "k" }, h("span", null, "Generation"), h("span", { class: "faint" }, decodeNow !== null ? "now" : lr ? "last answer" : "")),
      h("div", { class: "v" }, fmt.rate(decode), h("small", null, "tok/s")),
      sparkline(hist.map((x) => x.decode ?? null), { color: "var(--accent)", id: "dec" }),
      h("div", { class: "sub" }, "Tokens per second while answering, last two minutes")));
    tiles.push(h("div", { class: "tile" },
      h("div", { class: "k" }, h("span", null, "Prompt reading (prefill)")),
      h("div", { class: "v" }, fmt.rate(prefill), h("small", null, "tok/s")),
      sparkline(hist.map((x) => x.prefill ?? null), { color: "var(--violet)", id: "pre" })));
    tiles.push(h("div", { class: "tile" }, h("div", { class: "rings" },
      ring(accept, { color: "var(--cyan)", label: "MTP drafts accepted" }),
      ring(fle ? cacheHit : null, { color: "var(--fuchsia)", label: fle ? "Expert cache hits" : "Expert cache (FLE only)" }))));

    const gpus = lastS.gpus || [];
    if (gpus.length) {
      tiles.push(h("h4", null, "GPU"));
      for (const g of gpus) {
        tiles.push(h("div", { class: "tile gpu" },
          h("div", { class: "gpu-name" }, g.name, h("span", null, [g.temperature_c ? `${g.temperature_c} °C` : "", g.power_w ? `${Math.round(g.power_w)} W` : ""].filter(Boolean).join(" · "))),
          meter("Busy", g.utilization || 0, 100, "cyan", `${g.utilization ?? "–"} %`),
          meter("Memory", (g.memory_used || 0), (g.memory_total || 0), "violet")));
      }
    }
    tiles.push(h("h4", null, "Memory"));
    const memTile = h("div", { class: "tile gpu" });
    if (lastS.ram_total) memTile.append(meter("System RAM", lastS.ram_used, lastS.ram_total));
    if (lastS.engine_mem) memTile.append(h("div", { class: "row between small" }, h("span", null, "Engine process"), h("span", { class: "muted num" }, fmt.mem(lastS.engine_mem))));
    if (lastS.slots) memTile.append(h("div", { class: "row between small" }, h("span", null, "Slots busy"), h("span", { class: "muted num" }, `${lastS.slots_busy || 0} of ${lastS.slots}`)));
    if (meta.totals?.generated) memTile.append(h("div", { class: "row between small" }, h("span", null, "Generated since start"), h("span", { class: "muted num" }, `${fmt.int(meta.totals.generated)} tokens`)));
    tiles.push(memTile);
    if (meta.metrics_available === false) {
      tiles.push(h("div", { class: "callout" }, icon("info", "sm"), h("div", { class: "small" }, "This server runs without --metrics, so the numbers come from each finished answer.")));
    }
    if (svc?.expert_cache && !svc.cache_in_log && cacheHit === null) {
      tiles.push(h("div", { class: "callout" }, icon("info", "sm"), h("div", { class: "small" },
        `The expert-cache hit rate is a log line at level 4; ${svc.unit} logs up to level ${svc.log_level}. Add -lv 4 to its command to see it here.`)));
    }
    clear(live, tiles);
  }

  const resizeOnce = debounce(() => renderLive(), 150);
  window.addEventListener("resize", resizeOnce);

  return {
    unmount() {
      unsub.forEach((u) => u());
      window.removeEventListener("resize", resizeOnce);
      if (streaming) streaming.ctrl.abort();
    },
  };
}
