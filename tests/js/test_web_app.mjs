import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import test from "node:test";

const source = await readFile(
  new URL("../../src/scratch_llm/web/static/app.js", import.meta.url),
  "utf8",
);
const moduleUrl = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
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
  assert.deepEqual(buildGenerateMessage("hello", settings), {
    protocol_version: "v1",
    type: "generate",
    message: "hello",
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
