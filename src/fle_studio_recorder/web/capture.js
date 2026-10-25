// Settings: Capture. Keep what you do with the model as files on this computer, to build your own corpus.

import { get, post } from "/js/api.js";
import { clear, fmt, h, toast } from "/js/ui.js";

// ui: the small helpers of settings.js ({ section, field, toggle, save, render }).
export function captureSection(s, ui) {
  const c = s.capture || { enabled: false, max_mb: 2048 };
  const info = h("div", { class: "small faint" }, "…");
  const limit = h("input", { class: "input num", type: "number", min: 0, step: 256, value: c.max_mb,
    onchange: (e) => ui.save({ capture: { max_mb: Math.max(0, Number(e.target.value) || 0) } }, true) });

  async function refresh() {
    try {
      const st = await get("/api/capture");
      clear(info, st.records
        ? h("span", null, `${st.records} recorded answers in ${st.files} file${st.files === 1 ? "" : "s"} (${fmt.bytes(st.bytes)}), ${st.first} to ${st.last}. Folder: `, h("span", { class: "mono" }, st.folder))
        : h("span", null, "Nothing recorded yet. Folder: ", h("span", { class: "mono" }, st.folder)));
    } catch (e) {
      clear(info, h("span", null, e.message));
    }
  }

  async function wipe() {
    if (!window.confirm("Delete every recorded answer on this computer? This cannot be undone.")) return;
    try {
      const r = await post("/api/capture/clear", { confirm: true });
      toast(`Deleted ${r.deleted} file${r.deleted === 1 ? "" : "s"}`, "good", 2500);
    } catch (e) {
      toast(e.message, "bad", 6000);
    }
    refresh();
  }

  refresh();

  return ui.section("folder", "Capture", "Keep what you do with the model, to build your own corpus (tests, calibration, training data for a better draft head).",
    h("label", { class: "row" },
      ui.toggle(c.enabled, (v) => ui.save({ capture: { enabled: v } }, true).then(() => ui.render())),
      h("span", null, "Record my chats and requests "), h("span", { class: "small faint" }, "(off by default)")),
    c.enabled ? h("p", { class: "small" }, "Every answer is written as one line of JSON to a file on this computer: the request, the answer, the speed and how many draft tokens were accepted. Pictures are left out. Nothing is sent anywhere.") : null,
    ui.field("Stop at (MB)", limit, "The recording stops at this size instead of filling the disk. 0 means no limit."),
    info,
    h("div", { class: "row" },
      h("button", { class: "btn", onclick: refresh }, "Refresh"),
      h("button", { class: "btn ghost", onclick: wipe }, "Delete all recordings")));
}
