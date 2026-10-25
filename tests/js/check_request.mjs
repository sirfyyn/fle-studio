// Checks of the browser-side request and tree code under Node (no DOM needed).
// Run: node --test tests/js/check_request.mjs
//
// The page's modules are loaded from a data: URL so that this works on every Node version without a package.json
// that marks the web folder as ES modules.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const web = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "src", "fle_studio", "web", "js");
const load = (name) =>
  import(`data:text/javascript;base64,${Buffer.from(readFileSync(join(web, name), "utf8")).toString("base64")}`);
const req = await load("chat-request.js");
const tree = await load("chat-tree.js");

const TOOLS = [{ type: "function", function: { name: "calculator", parameters: { type: "object" } } }];
const user = (content, extra = {}) => ({ role: "user", content, ...extra });

test("tools and a forced format never share a request", () => {
  for (const extra of [
    { response_format: { type: "json_object" } },
    { response_format: { type: "json_schema", json_schema: { name: "x", schema: {} } } },
    { json_schema: { type: "object" } },
    { grammar: "root ::= x" },
  ]) {
    const body = req.finalizeRequest({ messages: [], tools: TOOLS, tool_choice: "auto", ...extra });
    assert.equal("tools" in body, false, JSON.stringify(extra));
    assert.equal("tool_choice" in body, false);
    assert.deepEqual(Object.keys(extra).filter((k) => k in body), Object.keys(extra), "the format stays");
  }
});

test("tools stay when no format is forced", () => {
  for (const extra of [{}, { response_format: { type: "text" } }, { grammar: "" }, { json_schema: {} }]) {
    const body = req.finalizeRequest({ messages: [], tools: TOOLS, ...extra });
    assert.equal(body.tools.length, 1, JSON.stringify(extra));
  }
  assert.equal("tools" in req.finalizeRequest({ messages: [], tools: [] }), false);
});

test("buildRequest applies the rule through every route", () => {
  const body = req.buildRequest({ nodes: [user("hi")], tools: TOOLS, extra: { response_format: { type: "json_object" } } });
  assert.equal("tools" in body, false);
  assert.equal(req.buildRequest({ nodes: [user("hi")], tools: TOOLS }).tools.length, 1);
});

test("attachments become a block in front of the text", () => {
  const msg = user("Summarise", { attachments: [{ kind: "text", name: 'a "b".txt', text: "body" }] });
  const [out] = req.apiMessages([msg]);
  assert.equal(out.content, `<attachment name="a 'b'.txt">\nbody\n</attachment>\n\nSummarise`);
});

test("pictures travel as image parts, and only when loaded", () => {
  const msg = user("what is this", { attachments: [{ kind: "image", name: "c.png", blob: "h.png" }] });
  assert.equal(req.apiMessages([msg])[0].content, "what is this");
  const [out] = req.apiMessages([msg], new Map([["h.png", "data:image/png;base64,AAAA"]]));
  assert.equal(out.content[0].type, "text");
  assert.equal(out.content[1].image_url.url, "data:image/png;base64,AAAA");
  assert.deepEqual(req.usedImages([msg]), ["h.png"]);
});

test("a tool round expands to assistant tool_calls plus tool messages", () => {
  const asked = { role: "assistant", content: "", tool_calls: [{ id: "c1", name: "calculator", arguments: '{"expression":"1+1"}', result: "2" },
    { id: "c2", name: "get_current_time", arguments: "", result: "" }] };
  const out = req.apiMessages([user("q"), asked, { role: "assistant", content: "It is 2." }]);
  assert.deepEqual(out.map((m) => m.role), ["user", "assistant", "tool", "tool", "assistant"]);
  assert.equal(out[1].tool_calls[1].function.arguments, "{}");
  assert.equal(out[2].content, "2");
  assert.match(out[3].content, /interrupted/);
});

test("tools go out sorted by name, whatever order they arrive in", () => {
  const t = (name) => ({ type: "function", function: { name, parameters: { type: "object" } } });
  const a = req.buildRequest({ nodes: [user("hi")], tools: [t("web_search"), t("calculator"), t("mcp__x")] });
  const b = req.buildRequest({ nodes: [user("hi")], tools: [t("mcp__x"), t("calculator"), t("web_search")] });
  assert.deepEqual(a.tools.map((x) => x.function.name), ["calculator", "mcp__x", "web_search"]);
  assert.equal(JSON.stringify(a.tools), JSON.stringify(b.tools));
});

test("the last round keeps the tools and forbids the call", () => {
  const body = req.buildRequest({ nodes: [user("hi")], tools: TOOLS, noMoreTools: true });
  assert.equal(body.tools.length, 1);
  assert.equal(body.tool_choice, "none");
  assert.equal("tool_choice" in req.buildRequest({ nodes: [user("hi")], tools: TOOLS }), false);
  assert.equal("tool_choice" in req.buildRequest({ nodes: [user("hi")], tools: [], noMoreTools: true }), false);
});

test("thoughts travel back only after the last user message", () => {
  const think = (content, reasoning, extra = {}) => ({ role: "assistant", content, reasoning, ...extra });
  const nodes = [
    user("old question"), think("old answer", "old thoughts"),
    user("new question"),
    think("", "step one", { tool_calls: [{ id: "c1", name: "calculator", arguments: "{}", result: "2" }] }),
    think("It is 2.", "step two"),
  ];
  const out = req.apiMessages(nodes);
  assert.equal("reasoning_content" in out[1], false, "before the last user message: no thoughts");
  const asked = out.find((m) => m.tool_calls);
  assert.equal(asked.reasoning_content, "step one");
  assert.equal(out[out.length - 1].reasoning_content, "step two");
  assert.equal("reasoning_content" in req.apiMessages([user("q"), think("a", "")])[1], false, "nothing to send");
});

test("the token estimate matches the server's rule", () => {
  assert.equal(req.estimateTokens(""), 0);
  assert.equal(req.estimateTokens("a"), 1);
  assert.equal(req.estimateTokens("x".repeat(3600)), 1000);
  assert.equal(req.estimateTokens("ü".repeat(10)), 9);
  const body = req.buildRequest({ nodes: [user("x".repeat(360))], tools: TOOLS });
  assert.ok(req.estimateRequest(body) > 100);
});

// ---- the markdown of a streaming answer

const md = await load("md.js");
const DOC = [
  "# Title", "", "Intro with **bold** and `code`.", "", "- one", "- two", "", "  more of two", "", "- three", "",
  "Plain text after the list.", "", "Before the list:", "", "1. first", "", "2. second", "",
  "```py", "def f():", "", "    return 1", "```", "", "| a | b |", "|---|---|", "| 1 | 2 |", "",
  "> quoted", "> still", "", "$$", "x = 1", "", "y = 2", "$$", "", "The end.", "",
].join("\n");

test("a streamed answer renders like the whole text, at every point", () => {
  const r = md.createStreamRenderer();
  let shown = "";
  for (let n = 0; n <= DOC.length; n += 1) {
    const { reset, fresh, tail } = r.update(DOC.slice(0, n));
    shown = reset ? fresh : shown + fresh;
    assert.equal(shown + tail, md.renderMarkdown(DOC.slice(0, n)), `after ${n} characters`);
    assert.equal(r.html, shown);
  }
});

test("finished blocks are rendered once; a text that is not a continuation starts again", () => {
  const r = md.createStreamRenderer();
  const first = r.update("one\n\ntwo\n\nthree");
  assert.match(first.fresh, /one/);
  const again = r.update("one\n\ntwo\n\nthree and more\n\nfour\n\nfive");
  assert.doesNotMatch(again.fresh, /one/, "the first block is not rendered again");
  assert.equal(again.reset, false);
  const other = r.update("something else\n\nentirely\n\nnew");
  assert.equal(other.reset, true);
  assert.equal(other.fresh + other.tail, md.renderMarkdown("something else\n\nentirely\n\nnew"));
});

// ---- the tree

function sample() {
  const chat = { messages: [], leaf: null };
  const u1 = tree.addChild(chat, null, { role: "user", content: "one" });
  const a1 = tree.addChild(chat, u1, { role: "assistant", content: "A" });
  const u2 = tree.addChild(chat, a1, { role: "user", content: "two" });
  const a2 = tree.addChild(chat, u2, { role: "assistant", content: "B" });
  return { chat, u1, a1, u2, a2 };
}

test("editing adds a sibling and the switcher moves between them", () => {
  const { chat, a1, u2, a2 } = sample();
  const u2b = tree.addChild(chat, a1, { role: "user", content: "two, reworded" });
  const idx = tree.indexOf(chat);
  assert.deepEqual(tree.siblingsOf(u2b, idx).map((m) => m.id), [u2.id, u2b.id]);
  assert.equal(tree.pathOf(chat).length, 3);
  tree.switchTo(chat, u2);
  assert.equal(chat.leaf, a2.id, "the old branch comes back with its answer");
  assert.deepEqual(tree.pathOf(chat).map((m) => m.content), ["one", "A", "two", "B"]);
});

test("removing a branch shows its sibling, or the parent", () => {
  const { chat, a1, u2 } = sample();
  const u2b = tree.addChild(chat, a1, { role: "user", content: "other" });
  tree.removeBranch(chat, u2b);
  assert.equal(chat.leaf, tree.deepest(u2, tree.indexOf(chat)).id);
  tree.removeBranch(chat, u2);
  assert.equal(chat.leaf, a1.id);
  assert.equal(chat.messages.length, 2);
});

test("an old chat without ids becomes a chain", () => {
  const chat = { messages: [{ role: "user", content: "a" }, { role: "assistant", content: "b" }] };
  tree.ensureTree(chat);
  assert.equal(chat.messages[1].parent, chat.messages[0].id);
  assert.equal(chat.leaf, chat.messages[1].id);
  assert.equal(tree.pathOf(chat).length, 2);
});
