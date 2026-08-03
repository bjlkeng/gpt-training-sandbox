const API_VERSION = "v1";


function numericValue(control, {integer, optional = false}, label) {
  const raw = control.value;
  if (typeof raw !== "string" || raw.trim() === "") {
    if (optional) {
      return null;
    }
    throw new Error(`Invalid generation settings: ${label} is required.`);
  }
  const value = Number(raw);
  const min = Number(control.min);
  const max = Number(control.max);
  if (
    !Number.isFinite(value) ||
    !Number.isFinite(min) ||
    !Number.isFinite(max) ||
    (integer && !Number.isInteger(value)) ||
    value < min ||
    value > max
  ) {
    throw new Error(`Invalid generation settings: ${label} is out of range.`);
  }
  return value;
}


export function normalizeSettings(values) {
  return {
    temperature: numericValue(
      values.temperature,
      {integer: false},
      "temperature",
    ),
    top_k: numericValue(
      values.topK,
      {integer: true, optional: true},
      "top k",
    ),
    max_new_tokens: numericValue(
      values.maxNewTokens,
      {integer: true},
      "max new tokens",
    ),
  };
}


export function buildGenerateMessage(message, settings) {
  return {
    protocol_version: API_VERSION,
    type: "generate",
    message,
    settings,
  };
}


export function appendSafeText(element, text, documentRef = document) {
  if (typeof text !== "string" || text === "") {
    return;
  }
  element.appendChild(documentRef.createTextNode(text));
}


function countLabel(value, singular, plural) {
  return `${value} ${value === 1 ? singular : plural}`;
}


export function contextLabel(state) {
  const context = state?.context;
  if (!context) {
    return "No checkpoint context available";
  }
  const details = [`${context.prompt_tokens} / ${context.max_tokens} tokens`];
  if (context.dropped_turns > 0) {
    details.push(
      `${countLabel(context.dropped_turns, "older turn", "older turns")} dropped`,
    );
  }
  if (context.truncated_user_tokens > 0) {
    details.push(
      `${countLabel(context.truncated_user_tokens, "user token", "user tokens")} truncated`,
    );
  }
  return details.join(" · ");
}


export function applyControlState(controls, state) {
  const active = ["connecting", "generating", "stopping"].includes(state);
  const resetting = state === "resetting";
  controls.input.disabled = active || resetting;
  controls.send.disabled = active || resetting;
  controls.stop.disabled = !active || state === "stopping" || resetting;
  controls.reset.disabled = active || resetting;
  controls.temperature.disabled = active || resetting;
  controls.topK.disabled = active || resetting;
  controls.maxNewTokens.disabled = active || resetting;
}


export function websocketUrl(locationRef) {
  const scheme = locationRef.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${locationRef.host}/ws/generate`;
}


export function applyServerEvent(payload, view) {
  if (!payload || payload.protocol_version !== API_VERSION) {
    throw new Error("Unsupported server protocol response.");
  }
  if (payload.type === "start") {
    view.start();
    return false;
  }
  if (payload.type === "token") {
    const text = payload.event?.text_delta;
    if (typeof text === "string" && text !== "") {
      view.appendAssistant(text);
    }
    return false;
  }
  if (payload.type === "done") {
    const text = payload.event?.text_delta;
    if (typeof text === "string" && text !== "") {
      view.appendAssistant(text);
    }
    view.finish("done", payload.state, payload);
    return true;
  }
  if (["cancelled", "busy", "error"].includes(payload.type)) {
    view.finish(payload.type, payload.state ?? null, payload);
    return true;
  }
  throw new Error("Unknown server event.");
}


export async function resetConversation(fetchImpl, view) {
  const response = await fetchImpl("/api/reset", {
    method: "POST",
    headers: {accept: "application/json"},
  });
  if (!response.ok) {
    throw new Error("The server did not reset the conversation.");
  }
  const payload = await response.json();
  view.clearConversation();
  view.updateContext(payload.state);
  return payload.state;
}


function messageNode(documentRef, role, text) {
  const article = documentRef.createElement("article");
  const heading = documentRef.createElement("h2");
  const body = documentRef.createElement("p");
  article.className = `message ${role}`;
  heading.textContent = role === "user" ? "You" : "Assistant";
  appendSafeText(body, text, documentRef);
  article.append(heading, body);
  return {article, body};
}


export class ChatController {
  constructor({documentRef, fetchImpl, WebSocketImpl, locationRef}) {
    this.document = documentRef;
    this.fetch = fetchImpl;
    this.WebSocket = WebSocketImpl;
    this.location = locationRef;
    this.form = documentRef.getElementById("chat-form");
    this.log = documentRef.getElementById("chat-log");
    this.status = documentRef.getElementById("connection-status");
    this.context = documentRef.getElementById("context-status");
    this.controls = {
      input: documentRef.getElementById("message-input"),
      temperature: documentRef.getElementById("temperature"),
      topK: documentRef.getElementById("top-k"),
      maxNewTokens: documentRef.getElementById("max-new-tokens"),
      send: documentRef.getElementById("send-button"),
      stop: documentRef.getElementById("stop-button"),
      reset: documentRef.getElementById("reset-button"),
    };
    this.state = "idle";
    this.socket = null;
    this.terminalReceived = false;
    this.stopSent = false;
    this.activeUser = null;
    this.activeAssistant = null;
    this.activeAssistantBody = null;
    this.bind();
    this.transition("idle");
  }

  bind() {
    this.form.addEventListener("submit", (event) => {
      event.preventDefault();
      this.submit();
    });
    this.controls.input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        this.form.requestSubmit();
      }
    });
    this.controls.stop.addEventListener("click", () => this.stop());
    this.controls.reset.addEventListener("click", () => this.reset());
  }

  transition(state) {
    this.state = state;
    applyControlState(this.controls, state);
  }

  setStatus(text) {
    this.status.textContent = text;
  }

  updateContext(state) {
    this.context.textContent = contextLabel(state);
  }

  addMessage(role, text) {
    const node = messageNode(this.document, role, text);
    this.log.appendChild(node.article);
    this.log.scrollTop = this.log.scrollHeight;
    return node;
  }

  submit() {
    if (this.state !== "idle") {
      return;
    }
    const message = this.controls.input.value;
    if (message.trim() === "") {
      this.setStatus("Enter a message first.");
      return;
    }
    let settings;
    try {
      settings = normalizeSettings(this.controls);
    } catch (error) {
      this.setStatus(error.message);
      return;
    }

    const user = this.addMessage("user", message);
    const assistant = this.addMessage("assistant", "");
    this.activeUser = user.article;
    this.activeAssistant = assistant.article;
    this.activeAssistantBody = assistant.body;
    this.terminalReceived = false;
    this.stopSent = false;
    this.transition("connecting");
    this.setStatus("Connecting…");

    const socket = new this.WebSocket(websocketUrl(this.location));
    this.socket = socket;
    socket.addEventListener("open", () => {
      if (this.socket !== socket) {
        return;
      }
      socket.send(JSON.stringify(buildGenerateMessage(message, settings)));
      this.controls.input.value = "";
      this.setStatus("Waiting for the server…");
    });
    socket.addEventListener("message", (event) => {
      if (this.socket !== socket) {
        return;
      }
      try {
        const payload = JSON.parse(event.data);
        applyServerEvent(payload, {
          start: () => {
            this.transition("generating");
            this.setStatus("Generating…");
          },
          appendAssistant: (text) => {
            appendSafeText(this.activeAssistantBody, text, this.document);
            this.log.scrollTop = this.log.scrollHeight;
          },
          finish: (kind, state, terminal) => this.finish(kind, state, terminal),
        });
      } catch (_error) {
        this.finish("error", null, {
          error: {message: "The server sent an invalid response."},
        });
      }
    });
    socket.addEventListener("error", () => {
      if (!this.terminalReceived) {
        this.setStatus("Connection error; this turn was not saved.");
      }
    });
    socket.addEventListener("close", () => {
      if (this.socket !== socket) {
        return;
      }
      this.socket = null;
      if (!this.terminalReceived) {
        this.removeActiveTurn();
        this.transition("idle");
        this.setStatus("Disconnected; submit again to reconnect.");
        this.controls.input.focus();
      }
    });
  }

  stop() {
    if (!["connecting", "generating"].includes(this.state) || this.stopSent) {
      return;
    }
    this.stopSent = true;
    this.transition("stopping");
    this.setStatus("Stopping…");
    if (this.socket?.readyState === this.WebSocket.OPEN) {
      this.socket.send(
        JSON.stringify({protocol_version: API_VERSION, type: "stop"}),
      );
    } else {
      this.socket?.close();
    }
  }

  finish(kind, state, terminal) {
    this.terminalReceived = true;
    if (kind === "done") {
      this.updateContext(state);
      this.clearActiveTurn();
      this.setStatus("Complete");
    } else {
      this.removeActiveTurn();
      this.setStatus(
        kind === "cancelled"
          ? "Generation stopped; this turn was not saved."
          : terminal?.error?.message ?? "Generation failed.",
      );
    }
    this.transition("idle");
    this.socket?.close();
    this.controls.input.focus();
  }

  removeActiveTurn() {
    this.activeUser?.remove();
    this.activeAssistant?.remove();
    this.clearActiveTurn();
  }

  clearActiveTurn() {
    this.activeUser = null;
    this.activeAssistant = null;
    this.activeAssistantBody = null;
  }

  async reset() {
    if (this.state !== "idle") {
      return;
    }
    this.transition("resetting");
    this.setStatus("Resetting…");
    try {
      await resetConversation(this.fetch, {
        clearConversation: () => {
          this.log.replaceChildren();
          this.clearActiveTurn();
        },
        updateContext: (state) => this.updateContext(state),
      });
      this.setStatus("Conversation reset.");
    } catch (error) {
      this.setStatus(error.message);
    } finally {
      this.transition("idle");
      this.controls.input.focus();
    }
  }
}


if (typeof document !== "undefined" && typeof window !== "undefined") {
  window.addEventListener("DOMContentLoaded", () => {
    new ChatController({
      documentRef: document,
      fetchImpl: window.fetch.bind(window),
      WebSocketImpl: window.WebSocket,
      locationRef: window.location,
    });
  });
}
