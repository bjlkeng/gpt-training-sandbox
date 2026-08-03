import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import test from "node:test";

const inspectionSource = await readFile(
  new URL("../../src/scratch_llm/web/static/inspection.js", import.meta.url),
  "utf8",
);
const inspectionModuleUrl = `data:text/javascript;base64,${Buffer.from(inspectionSource).toString("base64")}`;
const {
  downloadTranscript,
  formatMetrics,
  loadCheckpoint,
  populateSelect,
  renderDebug,
  selectRenderer,
} = await import(inspectionModuleUrl);

const rawAppSource = await readFile(
  new URL("../../src/scratch_llm/web/static/app.js", import.meta.url),
  "utf8",
);
const appSource = rawAppSource.replace("./inspection.js", inspectionModuleUrl);
const moduleUrl = `data:text/javascript;base64,${Buffer.from(appSource).toString("base64")}`;
const {
  applyControlState,
  applyServerEvent,
  appendSafeText,
  buildGenerateMessage,
  ChatController,
  contextLabel,
  normalizeSettings,
  resetConversation,
  websocketUrl,
} = await import(moduleUrl);


test("settings are bounded and serialized exactly once", () => {
  const input = (value, min, max) => ({value, min, max});
  const settings = normalizeSettings({
    temperature: input("0.25", "0", "10"),
    topK: input("7", "1", "100000"),
    maxNewTokens: input("32", "1", "4096"),
  });

  assert.deepEqual(settings, {
    temperature: 0.25,
    top_k: 7,
    max_new_tokens: 32,
  });
  assert.deepEqual(buildGenerateMessage("hello", settings, true), {
    protocol_version: "v1",
    type: "generate",
    message: "hello",
    debug: true,
    settings,
  });
  for (const values of [
    {
      temperature: input("-1", "0", "10"),
      topK: input("7", "1", "100000"),
      maxNewTokens: input("32", "1", "4096"),
    },
    {
      temperature: input("0.5", "0", "10"),
      topK: input("0", "1", "100000"),
      maxNewTokens: input("32", "1", "4096"),
    },
    {
      temperature: input("0.5", "0", "10"),
      topK: input("7.5", "1", "100000"),
      maxNewTokens: input("32", "1", "4096"),
    },
    {
      temperature: input("0.5", "0", "10"),
      topK: input("7", "1", "100000"),
      maxNewTokens: input("4097", "1", "4096"),
    },
  ]) {
    assert.throws(() => normalizeSettings(values), /generation settings/i);
  }
  assert.equal(
    normalizeSettings({
      temperature: input("0.5", "0", "10"),
      topK: input("", "1", "100000"),
      maxNewTokens: input("32", "1", "4096"),
    }).top_k,
    null,
  );
});


test("untrusted fragments use text nodes and preserve stream order", () => {
  const children = [];
  const element = {
    appendChild(child) {
      children.push(child);
    },
  };
  const documentRef = {
    createTextNode(data) {
      return {data, nodeType: 3};
    },
  };

  appendSafeText(element, "<img src=x onerror=alert(1)>", documentRef);
  appendSafeText(element, " & tail", documentRef);

  assert.equal(children.map((child) => child.data).join(""), "<img src=x onerror=alert(1)> & tail");
  assert.ok(children.every((child) => child.nodeType === 3));
});


test("server event reducer appends once and trusts server context only", () => {
  const calls = [];
  const view = {
    start() {
      calls.push(["start"]);
    },
    appendAssistant(text) {
      calls.push(["append", text]);
    },
    finish(kind, state) {
      calls.push(["finish", kind, state]);
    },
  };
  const state = {
    context: {
      prompt_tokens: 13,
      max_tokens: 64,
      dropped_turns: 2,
      truncated_user_tokens: 3,
    },
  };

  applyServerEvent(
    {protocol_version: "v1", type: "start", event: {text_delta: ""}},
    view,
  );
  applyServerEvent(
    {protocol_version: "v1", type: "token", event: {text_delta: "A"}},
    view,
  );
  applyServerEvent(
    {protocol_version: "v1", type: "token", event: {text_delta: "B"}},
    view,
  );
  applyServerEvent(
    {
      protocol_version: "v1",
      type: "done",
      event: {text_delta: ""},
      state,
    },
    view,
  );

  assert.deepEqual(calls, [
    ["start"],
    ["append", "A"],
    ["append", "B"],
    ["finish", "done", state],
  ]);
  assert.equal(
    contextLabel(state),
    "13 / 64 tokens · 2 older turns dropped · 3 user tokens truncated",
  );
  assert.equal(contextLabel({context: null}), "No checkpoint context available");
});


test("controls have deterministic idle, active, and resetting states", () => {
  const controls = Object.fromEntries(
    ["input", "send", "stop", "reset", "temperature", "topK", "maxNewTokens"].map(
      (name) => [name, {disabled: false}],
    ),
  );

  applyControlState(controls, "generating");
  assert.equal(controls.stop.disabled, false);
  assert.equal(controls.send.disabled, true);
  assert.equal(controls.reset.disabled, true);

  applyControlState(controls, "idle");
  assert.equal(controls.stop.disabled, true);
  assert.equal(controls.send.disabled, false);
  assert.equal(controls.reset.disabled, false);

  applyControlState(controls, "resetting");
  assert.ok(Object.values(controls).every((control) => control.disabled));
});


test("reset clears only after server acknowledgement", async () => {
  let resolveFetch;
  const fetchPromise = new Promise((resolve) => {
    resolveFetch = resolve;
  });
  const calls = [];
  const operation = resetConversation(
    () => fetchPromise,
    {
      clearConversation() {
        calls.push("clear");
      },
      updateContext(state) {
        calls.push(["context", state]);
      },
    },
  );

  await Promise.resolve();
  assert.deepEqual(calls, []);
  resolveFetch({
    ok: true,
    async json() {
      return {state: {context: {prompt_tokens: 0, max_tokens: 64}}};
    },
  });
  await operation;

  assert.deepEqual(calls, [
    "clear",
    ["context", {context: {prompt_tokens: 0, max_tokens: 64}}],
  ]);
});


test("WebSocket URL follows page security and can be recreated after close", () => {
  assert.equal(
    websocketUrl({protocol: "http:", host: "127.0.0.1:8000"}),
    "ws://127.0.0.1:8000/ws/generate",
  );
  assert.equal(
    websocketUrl({protocol: "https:", host: "localhost"}),
    "wss://localhost/ws/generate",
  );
  assert.notEqual(
    websocketUrl({protocol: "http:", host: "localhost:8000"}),
    websocketUrl({protocol: "http:", host: "localhost:8001"}),
  );
});


test("controller reconnects after a disconnect and stop is sent once", () => {
  class FakeElement {
    constructor() {
      this.children = [];
      this.listeners = new Map();
      this.value = "";
      this.min = "";
      this.max = "";
      this.disabled = false;
      this.textContent = "";
      this.parent = null;
      this.scrollHeight = 0;
      this.scrollTop = 0;
    }

    addEventListener(type, listener) {
      this.listeners.set(type, listener);
    }

    appendChild(child) {
      child.parent = this;
      this.children.push(child);
      return child;
    }

    append(...children) {
      for (const child of children) {
        this.appendChild(child);
      }
    }

    remove() {
      if (this.parent) {
        this.parent.children = this.parent.children.filter((child) => child !== this);
      }
    }

    replaceChildren() {
      this.children = [];
    }

    focus() {
      this.focused = true;
    }

    requestSubmit() {
      this.listeners.get("submit")?.({preventDefault() {}});
    }
  }

  const ids = [
    "chat-form",
    "chat-log",
    "connection-status",
    "context-status",
    "message-input",
    "temperature",
    "top-k",
    "max-new-tokens",
    "send-button",
    "stop-button",
    "reset-button",
    "checkpoint-select",
    "load-checkpoint-button",
    "renderer-select",
    "apply-renderer-button",
    "export-button",
    "debug-enabled",
    "debug-output",
    "generated-token-metric",
    "prefill-metric",
    "decode-metric",
    "throughput-metric",
    "memory-metric",
  ];
  const elements = Object.fromEntries(ids.map((id) => [id, new FakeElement()]));
  Object.assign(elements.temperature, {value: "0.25", min: "0", max: "10"});
  Object.assign(elements["top-k"], {value: "7", min: "1", max: "100000"});
  Object.assign(elements["max-new-tokens"], {
    value: "32",
    min: "1",
    max: "4096",
  });
  const documentRef = {
    getElementById(id) {
      return elements[id];
    },
    createElement() {
      return new FakeElement();
    },
    createTextNode(data) {
      return {data, nodeType: 3};
    },
  };

  class FakeSocket {
    static OPEN = 1;
    static instances = [];

    constructor(url) {
      this.url = url;
      this.readyState = 0;
      this.listeners = new Map();
      this.sent = [];
      FakeSocket.instances.push(this);
    }

    addEventListener(type, listener) {
      this.listeners.set(type, listener);
    }

    emit(type, event = {}) {
      if (type === "open") {
        this.readyState = FakeSocket.OPEN;
      }
      this.listeners.get(type)?.(event);
    }

    send(message) {
      this.sent.push(JSON.parse(message));
    }

    close() {
      this.readyState = 3;
    }
  }

  const controller = new ChatController({
    documentRef,
    fetchImpl() {
      throw new Error("not used");
    },
    WebSocketImpl: FakeSocket,
    locationRef: {protocol: "http:", host: "localhost:8000"},
  });
  controller.sessionReady = true;
  controller.transition("idle");
  elements["message-input"].value = "first";
  controller.submit();
  const first = FakeSocket.instances[0];
  first.emit("open");
  assert.equal(first.sent.length, 1);
  assert.equal(controller.state, "connecting");
  first.emit("close");
  assert.equal(controller.state, "idle");
  assert.equal(elements["chat-log"].children.length, 0);

  elements["message-input"].value = "second";
  controller.submit();
  const second = FakeSocket.instances[1];
  second.emit("open");
  second.emit("message", {
    data: JSON.stringify({protocol_version: "v1", type: "start"}),
  });
  assert.equal(controller.state, "generating");
  controller.stop();
  controller.stop();
  assert.equal(second.sent.length, 2);
  assert.deepEqual(second.sent[1], {protocol_version: "v1", type: "stop"});
});


test("metrics and raw debug values are formatted from terminal server data", () => {
  assert.deepEqual(
    formatMetrics({
      generated_tokens: 3,
      prefill_latency_seconds: 0.125,
      decode_latency_per_sampled_token_seconds: 0.02,
      tokens_per_second: 40,
      peak_memory_mib: 512.5,
    }),
    {
      generatedTokens: "3",
      prefill: "125.0 ms",
      decode: "20.0 ms/token",
      throughput: "40.00 tok/s",
      memory: "512.5 MiB",
    },
  );
  assert.deepEqual(formatMetrics(null), {
    generatedTokens: "—",
    prefill: "—",
    decode: "—",
    throughput: "—",
    memory: "—",
  });

  const output = {textContent: ""};
  renderDebug(
    output,
    {
      prompt_token_ids: [1, 2],
      generated_token_ids: [3],
      completion_reason: "stop_token",
      stop_token_id: 264,
    },
    {
      context: {
        prompt_tokens: 2,
        max_tokens: 64,
        dropped_turns: 1,
        truncated_user_tokens: 4,
      },
    },
  );
  assert.match(output.textContent, /"prompt_token_ids"/);
  assert.match(output.textContent, /"dropped_turns": 1/);
});


test("server catalogs populate safe options without HTML parsing", () => {
  const children = [];
  const select = {
    value: "",
    replaceChildren() {
      children.length = 0;
    },
    appendChild(child) {
      children.push(child);
    },
  };
  const documentRef = {
    createElement(tag) {
      assert.equal(tag, "option");
      return {value: "", textContent: ""};
    },
  };

  populateSelect(
    select,
    [{id: "safe.pt", name: "<img src=x onerror=alert(1)>"}],
    "safe.pt",
    documentRef,
  );

  assert.equal(children[0].textContent, "<img src=x onerror=alert(1)>");
  assert.equal(children[0].value, "safe.pt");
  assert.equal(select.value, "safe.pt");
});


test("checkpoint and renderer mutations commit UI only after acknowledgement", async () => {
  let resolveLoad;
  const pending = new Promise((resolve) => {
    resolveLoad = resolve;
  });
  const commits = [];
  const loading = loadCheckpoint(() => pending, "model.pt", (payload) => {
    commits.push(["checkpoint", payload.state]);
  });
  await Promise.resolve();
  assert.deepEqual(commits, []);
  resolveLoad({
    ok: true,
    async json() {
      return {state: {status: "ready", checkpoint_id: "model.pt"}};
    },
  });
  await loading;
  assert.deepEqual(commits, [
    ["checkpoint", {status: "ready", checkpoint_id: "model.pt"}],
  ]);

  await assert.rejects(
    loadCheckpoint(
      async () => ({
        ok: false,
        async json() {
          return {error: {message: "load failed"}};
        },
      }),
      "broken.pt",
      () => commits.push(["unexpected"]),
    ),
    /load failed/,
  );
  assert.equal(commits.length, 1);

  await selectRenderer(
    async () => ({
      ok: true,
      async json() {
        return {state: {renderer_id: "canonical"}, history_reset: false};
      },
    }),
    "canonical",
    (payload) => commits.push(["renderer", payload.history_reset]),
  );
  assert.deepEqual(commits.at(-1), ["renderer", false]);
});


test("transcript download uses the fixed endpoint and server filename", async () => {
  const requests = [];
  const anchors = [];
  const documentRef = {
    body: {
      appendChild(anchor) {
        anchors.push(anchor);
      },
    },
    createElement(tag) {
      assert.equal(tag, "a");
      return {
        click() {
          this.clicked = true;
        },
        remove() {
          this.removed = true;
        },
      };
    },
  };
  const urlApi = {
    createObjectURL() {
      return "blob:local";
    },
    revokeObjectURL(value) {
      this.revoked = value;
    },
  };
  await downloadTranscript(
    async (...args) => {
      requests.push(args);
      return {
        ok: true,
        headers: {
          get() {
            return 'attachment; filename="scratch-llm-transcript.jsonl"';
          },
        },
        async blob() {
          return {size: 10};
        },
      };
    },
    documentRef,
    urlApi,
  );

  assert.equal(requests[0][0], "/api/transcript");
  assert.equal(anchors[0].download, "scratch-llm-transcript.jsonl");
  assert.equal(anchors[0].href, "blob:local");
  assert.equal(anchors[0].clicked, true);
  assert.equal(anchors[0].removed, true);
  assert.equal(urlApi.revoked, "blob:local");
});
