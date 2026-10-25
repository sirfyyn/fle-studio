// The recorder on every page (plugin `recorder`): the Live entry at the top of the chat list with the number of
// messages in it, and the Capture card in Settings. The Live page itself is live.js.

import { get } from "/js/api.js";
import { chatListHeads, emit, on, renderChatList, store } from "/js/app.js";
import { h, icon } from "/js/ui.js";
import { captureSection } from "./capture.js";

export function init(kit) {
  store.livechat = { count: 0, version: 0 }; // /api/livechat?summary=1, read again with every live sample

  chatListHeads.push((path) => {
    const count = store.livechat?.count || 0;
    return h("a", { class: `chat-item live${/^\/live\/?$/.test(path) ? " active" : ""}`, href: "/live", "data-link": true,
      title: "Everything that goes through /v1, kept in memory only" },
    icon("live", "sm"), h("span", { class: "t" }, "Live"), count ? h("span", { class: "badge", title: "Messages in Live" }, count) : null);
  });

  let busy = false;
  on("live", async () => { // the page's own one-second round: no timer of the plugin
    if (busy) return;
    busy = true;
    try {
      const lc = await get("/api/livechat?summary=1");
      const changed = lc.version !== store.livechat.version;
      store.livechat = lc;
      if (changed) { emit("livechat"); renderChatList(); }
    } catch { /* the next round asks again */ } finally { busy = false; }
  });
  renderChatList();

  kit.settingsSection(50, (s, ui) => captureSection(s, ui));
}
