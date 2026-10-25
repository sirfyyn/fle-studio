// The model switcher in the model menu of the chat (plugin `switch`). The menu of an attached service asks the
// providers in modelMenuProviders (app.js) for entries; this one lists the models of the switcher next to the service
// and follows a switch until it is done.

import { get, post } from "/js/api.js";
import { emit, modelMenuProviders, store } from "/js/app.js";
import { toast } from "/js/ui.js";

async function switchModel(t) {
  if (t.active) return;
  try {
    await post("/api/engine/switch", { target: t.id });
  } catch (e) {
    toast(e.message, "bad", 8000);
    return;
  }
  toast(`Switching to ${t.name}. The model stays loaded until you switch again.`, "info", 6000);
  const t0 = Date.now();
  for (;;) { // the switch waits for free slots and loads the model; follow it until it is done
    await new Promise((r) => setTimeout(r, 3000));
    let sw;
    try { sw = await get("/api/engine/switch"); } catch { sw = null; }
    if (sw && !sw.busy) {
      store.engine = await get("/api/engine");
      emit("engine");
      if (sw.error) toast(sw.error, "bad", 10000);
      else toast(`${t.name} is loaded.`, "good");
      return;
    }
    if (Date.now() - t0 > 20 * 60 * 1000) { toast("The switch is taking very long; see the Engine page.", "bad", 8000); return; }
  }
}

export function init() {
  modelMenuProviders.push(async (svc) => {
    const sw = await get("/api/engine/switch");
    if (!sw?.available || !sw.targets?.length) return null;
    const items = [{ head: `${svc.unit}: switch model` }];
    for (const t of sw.targets) {
      const switching = sw.busy && sw.target === t.id;
      items.push({ label: t.name, hint: switching ? "switching…" : t.active ? "loaded" : "", icon: t.active ? "check" : "models",
        onclick: () => switchModel(t) });
    }
    if (sw.error) items.push({ head: sw.error });
    return items;
  });
}
