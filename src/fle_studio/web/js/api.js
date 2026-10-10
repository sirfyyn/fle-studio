// Talking to the FLE Studio server. Every call that changes something carries the per-run token from the page.

const TOKEN = document.querySelector('meta[name="fle-token"]')?.content || "";

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

// FLE Studio was restarted and this page still holds the old run's token: reload once to get the new one.
// At most every 15 s, so a server that keeps refusing cannot put the page into a reload loop.
function reloadForNewSession(res) {
  if (res.status !== 403 || res.headers.get("X-FLE-Session") !== "expired") return false;
  let last = 0;
  try { last = Number(sessionStorage.getItem("fle-session-reload") || 0); } catch { /* storage blocked */ }
  if (Date.now() - last < 15000) return false;
  try { sessionStorage.setItem("fle-session-reload", String(Date.now())); } catch { /* reload anyway */ }
  location.reload();
  return true;
}

async function api(method, path, body) {
  const opts = { method, headers: { "X-FLE-Token": TOKEN } };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new ApiError(0, "FLE Studio is not reachable. Is it still running?");
  }
  if (res.status === 401) {
    location.href = "/login";
    throw new ApiError(401, "Sign in first");
  }
  if (reloadForNewSession(res)) throw new ApiError(403, "FLE Studio was restarted; reloading the page");
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = null; }
  if (!res.ok) throw new ApiError(res.status, data?.error?.message || `${res.status} ${res.statusText}`);
  return data;
}

export const get = (p) => api("GET", p);
export const post = (p, b = {}) => api("POST", p, b);
export const put = (p, b) => api("PUT", p, b);
export const del = (p) => api("DELETE", p);

// Streams an OpenAI-style chat completion through Studio's /v1 proxy. onDelta gets every parsed chunk.
export async function streamChat(body, { signal, onChunk }) {
  const res = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-FLE-Token": TOKEN, Accept: "text/event-stream" },
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  });
  if (!res.ok) {
    if (reloadForNewSession(res)) throw new ApiError(403, "FLE Studio was restarted; reloading the page");
    let msg = `${res.status} ${res.statusText}`;
    try { msg = (await res.json())?.error?.message || msg; } catch { /* keep status text */ }
    throw new ApiError(res.status, msg);
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let cut;
    while ((cut = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, cut).trim();
      buf = buf.slice(cut + 1);
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") return;
      let chunk;
      try { chunk = JSON.parse(data); } catch { continue; }
      if (chunk.error) throw new ApiError(500, chunk.error.message || "The engine reported an error");
      onChunk(chunk);
    }
  }
}
